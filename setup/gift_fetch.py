"""Survey and selectively download the GIFT-Eval pretraining set.

The full repo is ~975 GB. /raid has 2.4 TB free on a volume already at 82 %
shared by 61 users, so taking 40 % of the lab's remaining headroom for data you
will mostly never read is not a reasonable default.

It is also unnecessary. Reverso B.1 rebalances the corpus before training:

    "The dataset however is significantly imbalanced towards datasets such as
     Buildings900k and Era5. To resolve the imbalance, we precompute the strides
     on each dataset necessary to achieve a target (roughly uniform) fraction of
     time series sampled. For each dataset, we target a maximum of
     N_max = 100000 samples per epoch."

So the paper never trains on the bulk of those collections either. Of the ~140
directories in the repo, 68 are era5_<year> and cmip6_<year> slices; capping how
many of those you take is the same operation as the paper's stride rebalancing,
done earlier and cheaper.

    # look first -- downloads nothing, takes a few seconds
    python gift_fetch.py --survey

    # then fetch, with a hard budget
    python gift_fetch.py --download --max-gb 120 --era5 3 --cmip6 3

Everything lands under $HF_HOME (set by env_nala.sh to /raid/users/$USER/.hf).
Removing it later is one rm -rf; nothing here is irreversible.
"""
import argparse
import os
import re
import sys
from collections import defaultdict

REPO = "Salesforce/GiftEvalPretrain"
GB = 2 ** 30

# the three collections the paper names as dominating the corpus
SLICED = {
    "era5": re.compile(r"^era5_\d{4}$"),
    "cmip6": re.compile(r"^cmip6_\d{4}$"),
    "largest": re.compile(r"^largest_\d{4}$"),
}


def collection_of(path):
    return path.split("/")[0] if "/" in path else "(root)"


def survey():
    from huggingface_hub import HfApi
    api = HfApi()
    sizes, files = defaultdict(int), defaultdict(int)
    for f in api.list_repo_tree(REPO, repo_type="dataset", recursive=True):
        if getattr(f, "size", None) is None:
            continue
        c = collection_of(f.path)
        sizes[c] += f.size
        files[c] += 1
    return sizes, files


def classify(name):
    for fam, pat in SLICED.items():
        if pat.match(name):
            return fam
    if name.startswith("buildings_900k"):
        return "buildings_900k"
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--survey", action="store_true", help="list sizes, download nothing")
    p.add_argument("--download", action="store_true")
    p.add_argument("--max-gb", type=float, default=120.0,
                   help="hard budget; selection stops before exceeding it")
    p.add_argument("--era5", type=int, default=3, help="how many era5_<year> slices")
    p.add_argument("--cmip6", type=int, default=3, help="how many cmip6_<year> slices")
    p.add_argument("--largest", type=int, default=2, help="how many largest_<year> slices")
    p.add_argument("--buildings", action="store_true",
                   help="include buildings_900k (very large; off by default)")
    p.add_argument("--out", default=os.environ.get("HF_HOME", "") or None,
                   help="defaults to $HF_HOME")
    a = p.parse_args()

    if not (a.survey or a.download):
        p.error("pass --survey or --download")

    print(f"querying {REPO} ...", flush=True)
    sizes, files = survey()
    total = sum(sizes.values())
    print(f"{len(sizes)} collections, {total / GB:,.0f} GB, "
          f"{sum(files.values()):,} files\n")

    # ---- what the giants cost ------------------------------------------
    fam_tot = defaultdict(int)
    fam_n = defaultdict(int)
    for c, v in sizes.items():
        f = classify(c)
        if f:
            fam_tot[f] += v
            fam_n[f] += 1
    print(f"{'family':<18}{'dirs':>6}{'GB':>10}{'% of repo':>11}")
    for f in sorted(fam_tot, key=lambda k: -fam_tot[k]):
        print(f"{f:<18}{fam_n[f]:>6}{fam_tot[f] / GB:>10.0f}"
              f"{100 * fam_tot[f] / total:>10.0f}%")
    rest = total - sum(fam_tot.values())
    print(f"{'everything else':<18}{len(sizes) - sum(fam_n.values()):>6}"
          f"{rest / GB:>10.0f}{100 * rest / total:>10.0f}%")

    if a.survey:
        print(f"\n{'collection':<36}{'GB':>9}{'files':>8}")
        for c, v in sorted(sizes.items(), key=lambda kv: -kv[1])[:40]:
            print(f"{c:<36}{v / GB:>9.2f}{files[c]:>8}")
        print("\n(top 40 shown)  re-run with --download to fetch a selection")
        return

    # ---- build the selection -------------------------------------------
    keep, caps = [], {"era5": a.era5, "cmip6": a.cmip6, "largest": a.largest}
    taken = defaultdict(int)
    # small collections first, so the budget is spent on diversity not on slices
    for c, v in sorted(sizes.items(), key=lambda kv: kv[1]):
        fam = classify(c)
        if fam == "buildings_900k" and not a.buildings:
            continue
        if fam in caps:
            if taken[fam] >= caps[fam]:
                continue
            taken[fam] += 1
        keep.append(c)

    sel = sum(sizes[c] for c in keep)
    if sel / GB > a.max_gb:
        trimmed, run = [], 0
        for c in sorted(keep, key=lambda c: sizes[c]):
            if (run + sizes[c]) / GB > a.max_gb:
                break
            trimmed.append(c)
            run += sizes[c]
        print(f"\nselection was {sel / GB:.0f} GB, over the {a.max_gb:.0f} GB budget "
              f"-> trimmed to {len(trimmed)} collections, {run / GB:.0f} GB")
        keep, sel = trimmed, run

    print(f"\nselected {len(keep)} of {len(sizes)} collections, "
          f"{sel / GB:.1f} GB ({100 * sel / total:.1f} % of the repo)")
    for fam in caps:
        if taken[fam]:
            print(f"  {fam}: {taken[fam]} of {fam_n[fam]} slices")
    if not a.buildings:
        print("  buildings_900k: excluded (--buildings to include)")

    out = a.out
    if not out:
        sys.exit("\nno --out and $HF_HOME unset. source ~/env_nala.sh first.")
    free = os.statvfs(out if os.path.isdir(out) else "/")
    free_gb = free.f_bavail * free.f_frsize / GB
    print(f"\ntarget {out}  |  {free_gb:,.0f} GB free")
    if sel / GB > 0.5 * free_gb:
        sys.exit("refusing: selection is more than half the free space. "
                 "Lower --max-gb, or clear space first.")

    ans = input(f"\ndownload {sel / GB:.1f} GB? [y/N] ").strip().lower()
    if ans != "y":
        print("nothing downloaded.")
        return

    from huggingface_hub import snapshot_download
    snapshot_download(
        REPO, repo_type="dataset",
        allow_patterns=[f"{c}/*" for c in keep] + ["README.md", ".gitattributes"],
        max_workers=8,
    )
    print("\ndone. To remove later:  rm -rf $HF_HOME/datasets--Salesforce--GiftEvalPretrain")


if __name__ == "__main__":
    main()
