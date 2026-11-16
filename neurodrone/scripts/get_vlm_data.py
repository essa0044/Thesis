#!/usr/bin/env python3
"""Download the benchmark files the Jetson needs for the stage-1 accuracy check (about 2.2 GB), from the authors'
Hugging Face repositories - the same files the workstation used (Chapter 4).

    python scripts/get_vlm_data.py --root ~/nd_data              # RefDrone test + gRefCOCO val (+ its images)
    python scripts/get_vlm_data.py --root ~/nd_data --refcoco    # also RefCOCO testA/testB

Layout (what scripts/bench_grounding.py expects):  <root>/refdrone/, <root>/grefcoco/, <root>/refcoco/.
Files already present with the expected size are skipped. Needs `pip install huggingface_hub`.
"""
import argparse
import os
import shutil
import sys

REFDRONE = [("sunzc-sunny/RefDrone", "RefDrone_test_mdetr.json", 3_287_485),
            ("sunzc-sunny/RefDrone", "refdrone_test_base64.tsv", 884_067_043)]
GREFCOCO = [("FudanCVL/gRefCOCO", "grefs(unc).json", 58_950_386),
            ("FudanCVL/gRefCOCO", "instances.json", 120_213_916)]
# gRefCOCO's images are COCO train2014 pictures; bench_grounding.py takes them from RefCOCO's copies (val split)
# and only falls back to the COCO server for pictures that are not there
REFCOCO_VAL = [("lmms-lab-encoder/RefCOCO", f"data/val-0000{i}-of-00004.parquet", n) for i, n in
               enumerate((152_138_231, 217_257_428, 382_645_907, 390_523_008))]
REFCOCO_TEST = [("lmms-lab-encoder/RefCOCO", "data/testA-00000-of-00001.parquet", 129_502_844),
                ("lmms-lab-encoder/RefCOCO", "data/testB-00000-of-00001.parquet", 133_276_177)]


def size_ok(path, approx, tol=0.12):
    return os.path.exists(path) and abs(os.path.getsize(path) - approx) <= tol * approx


def fetch(repo, fname, sub, approx, root):
    from huggingface_hub import hf_hub_download
    dest_dir = os.path.join(root, sub)
    final = os.path.join(dest_dir, os.path.basename(fname))
    if size_ok(final, approx):
        print(f"ok (present)  {final}", flush=True)
        return final
    os.makedirs(dest_dir, exist_ok=True)
    print(f"downloading   {repo}/{fname} (~{approx / 1e6:.0f} MB)", flush=True)
    p = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset", local_dir=os.path.join(dest_dir, "_hf"))
    shutil.move(p, final)
    if not size_ok(final, approx):
        print(f"WARNING: {final} is {os.path.getsize(final)} bytes, expected about {approx}", flush=True)
    return final


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--refcoco", action="store_true", help="also RefCOCO testA and testB (260 MB)")
    a = ap.parse_args(argv)
    root = os.path.expanduser(a.root)
    jobs = [(r, f, "refdrone", n) for r, f, n in REFDRONE] + [(r, f, "grefcoco", n) for r, f, n in GREFCOCO]
    jobs += [(r, f, "refcoco", n) for r, f, n in REFCOCO_VAL + (REFCOCO_TEST if a.refcoco else [])]
    for job in jobs:
        fetch(*job, root)
    print(f"done: {root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
