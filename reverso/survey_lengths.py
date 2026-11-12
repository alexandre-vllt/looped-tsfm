"""Q8 : la distribution des longueurs des series-canaux de GEP-L.

260 968 series-canaux sur 475 536 (55 %) ont ete jetees comme "trop courtes"
par convert_gepl.py, et personne n'a mesure si elles font 2000 points ou 50.
C'est ce qui rend l'estimation de recuperation large d'un facteur 3
(+38 % a +82 %). Ce script la remplace par un nombre.

Ne convertit rien, n'ecrit aucun corpus. Lit UNIQUEMENT les longueurs des
listes dans les parquet (pyarrow compute, sans materialiser les flottants),
donc quelques minutes et non quelques heures.

    PYTHONPATH=... $VENV/bin/python survey_lengths.py
    $VENV/bin/python survey_lengths.py --shards /chemin/vers/GEP-L

CE QU'IL FAUT EN TIRER
Pour chaque seuil de longueur minimale, le script donne le nombre de series
recuperees, les chunks qu'elles produisent, et surtout la FRACTION FABRIQUEE
par Pad() -- une serie de 500 points padee a 2304 porte 78 % de valeurs
recopiees. Le papier pad (section 2.1), mais une fenetre majoritairement
constante est exactement le cas degenere que le garde de la section 9.1 jette.
Le bon seuil est celui ou la fraction fabriquee reste acceptable.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

CHUNK = 2304
SPAN = 2048 + 48


def find_shards(hint: str) -> list:
    pats = []
    if hint:
        pats += [os.path.join(hint, "train-*.parquet"),
                 os.path.join(hint, "GEP-L", "train-*.parquet")]
    work = os.environ.get("WORK", "")
    if work:
        pats += [os.path.join(work, ".hf", "**", "GEP-L", "train-*.parquet")]
    pats += [os.path.expanduser("~/.cache/huggingface/**/GEP-L/train-*.parquet")]
    for p in pats:
        got = sorted(glob.glob(p, recursive=True))
        if got:
            return got
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    shards = find_shards(a.shards)
    if not shards:
        print("Aucun shard train-*.parquet trouve.", file=sys.stderr)
        print("  cherche dans $WORK/.hf, ~/.cache/huggingface, et --shards",
              file=sys.stderr)
        print("  si .hf a ete vide : python convert_gepl.py --download",
              file=sys.stderr)
        raise SystemExit(1)
    print(f"{len(shards)} shards, "
          f"{sum(os.path.getsize(s) for s in shards)/1e9:.2f} GB\n")

    lens = []
    by_source = defaultdict(list)
    for i, sh in enumerate(shards, 1):
        pf = pq.ParquetFile(sh)
        for g in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(g, columns=["source", "target"])
            tgt = t.column("target").combine_chunks()
            # longueurs SANS materialiser les flottants
            nch = pc.list_value_length(tgt).to_numpy(zero_copy_only=False)
            inner = pc.list_flatten(tgt)
            L = pc.list_value_length(inner).to_numpy(zero_copy_only=False)
            lens.append(L)
            srcs = t.column("source").to_pylist()
            off = 0
            for s, n in zip(srcs, nch):
                n = int(n)
                by_source[s].append(L[off:off + n])
                off += n
        print(f"  shard {i}/{len(shards)}", flush=True)

    L = np.concatenate(lens)
    n = len(L)
    print(f"\n{n:,} series-canaux\n")

    print("=== distribution des longueurs ===")
    edges = [0, 128, 256, 512, 1024, 1152, 2096, CHUNK, 4608, 9216, 10**9]
    names = ["<128", "128-256", "256-512", "512-1k", "1k-1152", "1152-2096",
             "2096-2304", "2304-4608", "4608-9216", ">9216"]
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        c = int(((L >= lo) & (L < hi)).sum())
        bar = "#" * int(c / n * 60)
        print(f"  {nm:<12}{c:>9,}{c/n*100:>7.1f}%  {bar}")

    short = L[L < CHUNK]
    print(f"\n  trop courtes (< {CHUNK}) : {len(short):,}"
          f"  ({len(short)/n*100:.1f} %)   mediane {int(np.median(short)):,} points")

    print(f"\n=== ce que Pad() rend, par seuil ===")
    print(f"  {'seuil':>8}{'series':>10}{'chunks':>10}"
          f"{'% du corpus actuel':>21}{'fraction fabriquee':>21}")
    KEPT = 406_452
    for thr in (256, 512, 1024, 1152, 1536, 2096):
        sel = short[short >= thr]
        if not len(sel):
            continue
        frac = 1.0 - sel.mean() / CHUNK
        print(f"  {thr:>8,}{len(sel):>10,}{len(sel):>10,}"
              f"{len(sel)/KEPT*100:>20.1f}%{frac*100:>20.1f}%")
    print("\n  'fraction fabriquee' = part moyenne du chunk qui serait du padding.")
    print("  Au-dela de ~50 %, la fenetre est majoritairement constante et le")
    print("  garde de la section 9.1 la jettera de toute facon.")

    print(f"\n=== les {a.top} sources ayant le plus de series courtes ===")
    rank = []
    for s, arrs in by_source.items():
        v = np.concatenate(arrs)
        sh = int((v < CHUNK).sum())
        rank.append((sh, s, len(v), float(np.median(v))))
    rank.sort(reverse=True)
    print(f"  {'source':<36}{'courtes':>9}{'total':>9}{'% courtes':>11}{'mediane':>10}")
    for sh, s, tot, med in rank[:a.top]:
        print(f"  {s[:36]:<36}{sh:>9,}{tot:>9,}{sh/tot*100:>10.1f}%{int(med):>10,}")


if __name__ == "__main__":
    main()
