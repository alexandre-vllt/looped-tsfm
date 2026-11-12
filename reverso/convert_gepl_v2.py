"""GEP-L -> corpus v2. Garde ce que la v1 jetait.

La v1 a produit 406 452 chunks a partir de 1.66 G points de shards train, soit
57 %. Les 43 % perdus l'ont ete par QUATRE regles, toutes les notres :

  1. toute serie < 2304 points jetee          -> 260 968 series-canaux (55 %)
  2. tout chunk contenant UN non-fini jete    -> 95 849 chunks
  3. le reste apres decoupe jete              -> 17.7 % des points
  4. toute source < 32 chunks ignoree         -> 53 sources sur 87

La regle 1 est la pire : NEUF SOURCES ENTIERES sont a 100 % sous 2304 points et
n'existent donc pas dans notre corpus -- alibaba_cluster_trace, favorita_sales,
taxi_30min, kaggle_web_traffic_weekly, wiki-rolling_nips, m5, godaddy,
rideshare_with_missing, monash_m3_monthly. Du retail, du taxi, du trafic web :
exactement ce que contient GIFT-Eval. Et le papier, lui, les GARDE : section 2.1,
Pad() remplit les sequences courtes en recopiant le point le plus a gauche.

CE QUE CETTE VERSION CHANGE

  MIN_LEN = 1152        en dessous on jette, au-dessus on pad.
                        Mesure (survey_lengths.py) : 114 316 series recuperees,
                        27.8 % de contenu fabrique en moyenne -- sous le seuil ou
                        le garde de la section 9.1 jetterait la fenetre pour
                        platitude. A 512 on recupererait 188 670 series mais a
                        41.9 % de fabrique, ce qui ne vaut probablement rien.

  NAN_MAX = 0.10        un chunk a <=10 % de non-finis est interpole lineairement
                        plutot que jete. Au-dela il part. borg_cluster_data_2011
                        portait a lui seul 68 % des pertes NaN (88 % de ses chunks).

  chunk de queue        la fin de serie est reprise en RECOUVREMENT avec le chunk
                        precedent au lieu d'etre jetee. Deux chunks partagent
                        alors des points, ce qui n'arrivait pas en v1 : c'est une
                        deviation assumee, inscrite au manifeste.

  longueur PAR SOURCE   2304, 4608 ou 9216 selon ce que la source supporte.
                        36.1 % des series-canaux font 4608-9216 points a la
                        source, et les hacher a 2304 interdit mecaniquement le
                        downsampling multi-echelle d'Algorithm 4 (il faut
                        k x 2096 points). Stocker long le rend possible.
                        >>> EXIGE UNE MODIFICATION DE data.py : Corpus leve
                        >>> aujourd'hui sur des longueurs melangees. Voir --survey.

  MIN_CHUNKS = 8        au lieu de 32. Les petites sources sont precieuses pour
                        la diversite et serviront au split de validation.

  N_max = 100 000       inchange, c'est Reverso B.1.

USAGE
    python convert_gepl_v2.py --survey            # ce qu'il produirait, sans ecrire
    python convert_gepl_v2.py --convert --out $WORK/real_v2
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

SPAN = 2048 + 48
LENGTHS = (2304, 4608, 9216)      # longueurs de stockage autorisees
MIN_LEN = 1152                    # en dessous : jete
NAN_MAX = 0.10                    # au dela : chunk jete
MIN_CHUNKS = 8
N_MAX = 100_000
LONG_SHARE = 0.60                 # part de la source devant atteindre L pour le choisir


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))


def find_shards(hint: str):
    pats = []
    if hint:
        pats += [os.path.join(hint, "train-*.parquet"),
                 os.path.join(hint, "GEP-L", "train-*.parquet")]
    w = os.environ.get("WORK", "")
    if w:
        pats.append(os.path.join(w, ".hf", "**", "GEP-L", "train-*.parquet"))
    pats.append(os.path.expanduser("~/.cache/huggingface/**/GEP-L/train-*.parquet"))
    for p in pats:
        got = sorted(glob.glob(p, recursive=True))
        if got:
            return got
    return []


def pick_length(lengths: np.ndarray) -> int:
    """La plus grande longueur stockable qu'une part LONG_SHARE de la source atteint."""
    for L in sorted(LENGTHS, reverse=True):
        if (lengths >= L).mean() >= LONG_SHARE:
            return L
    return LENGTHS[0]


def repair(a: np.ndarray):
    """Interpole les trous courts. Retourne (serie, ok, part_non_finie)."""
    bad = ~np.isfinite(a)
    frac = float(bad.mean())
    if frac == 0.0:
        return a, True, 0.0
    if frac > NAN_MAX:
        return a, False, frac
    idx = np.arange(a.size)
    good = ~bad
    if good.sum() < 2:
        return a, False, frac
    a = a.copy()
    a[bad] = np.interp(idx[bad], idx[good], a[good])
    return a, True, frac


def chunks_of(a: np.ndarray, L: int):
    """Chunks de longueur L, avec un chunk de queue EN RECOUVREMENT, ou Pad()."""
    n = a.size
    if n < MIN_LEN:
        return []
    if n < L:
        out = np.empty(L, dtype=np.float32)
        out[: L - n] = a[0]          # Pad() de la section 2.1 : back-fill a gauche
        out[L - n:] = a
        return [out]
    full = n // L
    res = [a[i * L:(i + 1) * L] for i in range(full)]
    if n - full * L >= L // 4:       # queue assez longue -> chunk en recouvrement
        res.append(a[n - L:])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", default="")
    ap.add_argument("--survey", action="store_true")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--min-len", type=int, default=MIN_LEN)
    ap.add_argument("--nan-max", type=float, default=NAN_MAX)
    a = ap.parse_args()
    globals()["MIN_LEN"], globals()["NAN_MAX"] = a.min_len, a.nan_max

    if not (a.survey or a.convert):
        ap.error("choisir --survey ou --convert")
    if a.convert and not a.out:
        ap.error("--convert exige --out")

    shards = find_shards(a.shards)
    if not shards:
        raise SystemExit("aucun shard train-*.parquet trouve")
    print(f"{len(shards)} shards\n")

    # ---- passe 1 : longueurs par source, pour choisir L ------------------
    print("passe 1/2 : longueurs par source")
    lens = defaultdict(list)
    for i, sh in enumerate(shards, 1):
        pf = pq.ParquetFile(sh)
        for g in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(g, columns=["source", "target"])
            tgt = t.column("target").combine_chunks()
            nch = pc.list_value_length(tgt).to_numpy(zero_copy_only=False)
            L = pc.list_value_length(pc.list_flatten(tgt)).to_numpy(zero_copy_only=False)
            off = 0
            for s, n in zip(t.column("source").to_pylist(), nch):
                n = int(n)
                lens[safe(s)].append(L[off:off + n]); off += n
        print(f"  {i}/{len(shards)}", flush=True)

    plan = {}
    for k, v in lens.items():
        arr = np.concatenate(v)
        plan[k] = {"L": pick_length(arr), "n_series": int(arr.size),
                   "median": int(np.median(arr)),
                   "kept": int((arr >= MIN_LEN).sum()),
                   "lens": arr}

    print(f"\n=== plan : {len(plan)} sources ===")
    print(f"  {'source':<34}{'series':>8}{'>=min':>8}{'mediane':>9}{'L stocke':>10}")
    byL = Counter()
    for k in sorted(plan, key=lambda x: -plan[x]["n_series"]):
        p = plan[k]
        byL[p["L"]] += 1
        print(f"  {k[:34]:<34}{p['n_series']:>8,}{p['kept']:>8,}"
              f"{p['median']:>9,}{p['L']:>10,}")
    print(f"\n  longueurs retenues : "
          + ", ".join(f"{L} -> {n} sources" for L, n in sorted(byL.items())))
    if len(byL) > 1:
        print("\n  !! LONGUEURS MELANGEES : Corpus.__init__ de data.py leve aujourd'hui")
        print("     sur ce cas (\"one Corpus must hold one length\"). Le patch est court")
        print("     -- draw() doit lire self.lengths[g] au lieu de self.series_len --")
        print("     mais il doit etre fait AVANT d'entrainer sur ce corpus.")

    # ---- totaux (estimes : le rejet NaN n'est connu qu'en passe 2) -------
    def n_chunks(lens, L):
        """Mirror of chunks_of(), counted instead of materialised."""
        n = lens[lens >= MIN_LEN]
        short = int((n < L).sum())              # 1 chunk chacune, via Pad()
        lo = n[n >= L]
        full = lo // L
        tail = int(((lo - full * L) >= (L // 4)).sum())
        return short + int(full.sum()) + tail

    tot_src = tot_ch = tot_pts = 0
    for k, p in plan.items():
        c = min(n_chunks(p["lens"], p["L"]), N_MAX)
        if c < MIN_CHUNKS:
            continue
        tot_src += 1
        tot_ch += c
        tot_pts += c * p["L"]
    gb = tot_pts * 4 / 1e9

    print(f"\n=== totaux estimes ===")
    print(f"  {'sources retenues':<26}{tot_src:>12,} / {len(plan)}")
    print(f"  {'chunks':<26}{tot_ch:>12,}")
    print(f"  {'points':<26}{tot_pts:>12,}")
    print(f"  {'sur disque':<26}{gb:>11.2f} Go")
    print("  (avant rejet NaN > %.0f %%, qui n'est connu qu'en passe 2)"
          % (NAN_MAX * 100))
    print(f"\n  reference v1 : 406 575 chunks / 0.94 G points / 3.49 Go, "
          f"34 sources")
    if tot_ch:
        print(f"  gain         : x{tot_ch / 406575:.2f} chunks, "
              f"x{tot_pts / 0.94e9:.2f} points, x{tot_src / 34:.2f} sources")

    print(f"\n=== steps par epoch, batch 512, une fenetre par chunk ===")
    print(f"  {'p_real':<10}{'fenetres/epoch':>18}{'steps/epoch':>14}")
    for pr in (0.30, 0.25, 0.20, 0.15):
        win = tot_ch / pr
        print(f"  {pr:<10.2f}{win:>18,.0f}{win / 512:>14,.0f}")

    if not a.convert:
        print("\n  --survey : rien n'a ete ecrit.")
        return

    # ---- passe 2 : conversion -------------------------------------------
    os.makedirs(a.out, exist_ok=True)
    print(f"\npasse 2/2 : conversion vers {a.out}")
    per = defaultdict(list)
    st = Counter()
    t0 = time.time()
    for i, sh in enumerate(shards, 1):
        pf = pq.ParquetFile(sh)
        for g in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(g, columns=["source", "target"])
            for src, tgt in zip(t.column("source").to_pylist(),
                                t.column("target").to_pylist()):
                k = safe(src)
                if len(per[k]) >= N_MAX:
                    st["capped"] += 1
                    continue
                L = plan[k]["L"]
                for ch in tgt:
                    st["channel_series"] += 1
                    arr = np.asarray(ch, dtype=np.float32)
                    if arr.size < MIN_LEN:
                        st["too_short"] += 1
                        continue
                    for c in chunks_of(arr, L):
                        if len(per[k]) >= N_MAX:
                            break
                        c, ok, frac = repair(c)
                        if not ok:
                            st["nan_dropped"] += 1
                            continue
                        if frac > 0:
                            st["nan_repaired"] += 1
                        per[k].append(c.astype(np.float32))
                        st["kept"] += 1
        print(f"  shard {i}/{len(shards)}  chunks {st['kept']:,}  "
              f"{time.time()-t0:.0f}s", flush=True)

    print(f"\necriture de {len(per)} fichiers")
    gens, total = {}, 0
    for k in sorted(per):
        rows = per[k]
        if len(rows) < MIN_CHUNKS:
            st["sources_skipped"] += 1
            print(f"  skip {k:<30} {len(rows)} chunks")
            continue
        arr = np.stack(rows)
        p = os.path.join(a.out, f"{k}.npy")
        with open(p, "wb") as fh:
            np.save(fh, arr)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        gens[k] = {"generator": k, "n_series": int(arr.shape[0]),
                   "length": int(arr.shape[1]), "dtype": "float32",
                   "sha256": h, "bytes": int(arr.nbytes),
                   "source": "abel-lab/nanoTSFM-pretrain GEP-L train, v2: "
                             f"min_len {MIN_LEN}, Pad() 2.1, NaN<= {NAN_MAX}, "
                             "chunk de queue en recouvrement",
                   "median_source_len": plan[k]["median"]}
        total += int(arr.shape[0])

    corpus = hashlib.sha256("".join(
        f'{v["generator"]}:{v["sha256"]}' for v in sorted(gens.values(),
        key=lambda d: d["generator"])).encode()).hexdigest()
    json.dump({"created": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "total_series": total,
               "total_bytes": sum(v["bytes"] for v in gens.values()),
               "corpus_sha256": corpus,
               "policy": {"min_len": MIN_LEN, "nan_max": NAN_MAX,
                          "lengths": sorted({v["length"] for v in gens.values()}),
                          "min_chunks": MIN_CHUNKS, "n_max": N_MAX,
                          "overlapping_tail": True},
               "stats": dict(st), "generators": gens},
              open(os.path.join(a.out, "manifest.json"), "w"), indent=2)

    print(f"\n=== resultat ===")
    for k, v in sorted(st.items()):
        print(f"  {k:<22}{v:>10,}")
    print(f"  {'sources ecrites':<22}{len(gens):>10,}")
    print(f"  {'series':<22}{total:>10,}")
    print(f"  {'corpus_sha256':<22}{corpus[:16]}...")
    print(f"\n  v1 : 406 452 series, 34 sources")


if __name__ == "__main__":
    main()
