#!/usr/bin/env python
"""One-time data preparation for the offline flow-DPO baseline on Sana.

Builds the human-labelled image pairs that correspond to the prompts the online NPO runs train on:
  - source: pickapic-anonymous/pickapic_v1, split train (jpg_0, jpg_1, label_0, caption, ...)
  - keep a row if its ranking_id is in CarperAI/pickapic_v1_no_images_training_sfw (the SFW subset
    the NPO runs take their prompts from) and label_0 is 0 or 1 (same "no split decisions" filter as
    train.py / train_nash.py).
Each source shard is filtered and written as one Arrow IPC file (random access, memory-mappable),
uploaded to --out (a gs:// prefix) and removed locally, so no large local disk is needed.
A manifest.json listing every file and its row count is uploaded last and marks completion.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq

SRC = "pickapic-anonymous/pickapic_v1"
SFW = "CarperAI/pickapic_v1_no_images_training_sfw"
COLS = ["caption", "jpg_0", "jpg_1", "label_0", "ranking_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="gs://... prefix for the prepared shards")
    ap.add_argument("--tmp", default="/tmp/pickapic_prep")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit_shards", type=int, default=0, help="debug: only the first N shards")
    args = ap.parse_args()
    os.makedirs(args.tmp, exist_ok=True)

    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    sfw = load_dataset(SFW, split="train")
    keep_ids = {rid for rid, l0 in zip(sfw["ranking_id"], sfw["label_0"]) if l0 in (0, 1)}
    print(f"SFW rows: {len(sfw)}, with a clear decision: {len(keep_ids)} unique ranking ids", flush=True)
    keep_arr = pa.array(sorted(keep_ids), type=pa.int64())

    files = sorted(f for f in HfApi().list_repo_files(SRC, repo_type="dataset") if f.startswith("data/train-"))
    if args.limit_shards:
        files = files[:args.limit_shards]
    print(f"{len(files)} source shards", flush=True)

    def work(item):
        i, f = item
        name = f"pairs-{i:05d}.arrow"
        for attempt in range(5):
            try:
                local = hf_hub_download(SRC, f, repo_type="dataset", local_dir=args.tmp)
                t = pq.read_table(local, columns=COLS)
                os.remove(local)
                rid = t["ranking_id"].cast(pa.int64())
                l0 = t["label_0"]
                mask = pa.compute.and_(pa.compute.is_in(rid, value_set=keep_arr),
                                       pa.compute.or_(pa.compute.equal(l0, 0), pa.compute.equal(l0, 1)))
                t2 = t.filter(mask)
                out = os.path.join(args.tmp, name)
                with pa.OSFile(out, "wb") as sink:
                    with pa.ipc.new_file(sink, t2.schema) as w:
                        w.write_table(t2, max_chunksize=64)
                r = subprocess.run(["gcloud", "storage", "cp", "--quiet", out, f"{args.out}/{name}"],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(r.stderr[-300:])
                os.remove(out)
                return name, t2.num_rows, t.num_rows
            except Exception as e:  # network hiccups: retry
                print(f"[retry {attempt}] shard {i}: {type(e).__name__}: {e}", flush=True)
                time.sleep(10 * (attempt + 1))
        raise RuntimeError(f"shard {i} failed")

    t0 = time.time()
    manifest, kept, seen = [], 0, 0
    with ThreadPoolExecutor(args.workers) as pool:
        for n, (name, rows, src_rows) in enumerate(pool.map(work, list(enumerate(files))), start=1):
            manifest.append({"file": name, "rows": rows})
            kept += rows
            seen += src_rows
            if n % 10 == 0 or n == len(files):
                print(f"{n}/{len(files)} shards, kept {kept}/{seen} rows, {time.time() - t0:.0f}s", flush=True)

    mpath = os.path.join(args.tmp, "manifest.json")
    with open(mpath, "w") as f:
        json.dump({"source": SRC, "filter": f"ranking_id in {SFW} and label_0 in (0,1)",
                   "rows": kept, "source_rows": seen, "files": manifest}, f)
    subprocess.run(["gcloud", "storage", "cp", "--quiet", mpath, f"{args.out}/manifest.json"], check=True)
    print(f"DONE: kept {kept} of {seen} rows ({len(keep_ids)} expected) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
