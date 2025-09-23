#!/usr/bin/env python3
import argparse
import os
import re
from glob import glob
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Top-level results dir (your --input_dir)")
    p.add_argument("--model", required=True, help="Model name string (used in folder path)")
    return p.parse_args()

def find_files(folder):
    pat = os.path.join(folder, "results_*_*.npz")
    files = glob(pat)
    if not files:
        raise FileNotFoundError(f"No chunked results found in {folder}")
    # sort by start index
    def key(path):
        m = re.search(r"results_(\d+)_([0-9]+|end)\.npz$", os.path.basename(path))
        return int(m.group(1)) if m else 10**12
    return sorted(files, key=key)

def stack_seqwise(obj_array):
    return np.stack(list(obj_array), axis=0)

def aggregate(folder):
    ess_chunks, tle_chunks, gride_chunks, loss_chunks = [], [], [], []
    for f in find_files(folder):
        data = np.load(f, allow_pickle=True)
        ess = stack_seqwise(data["ESS"])
        tle = stack_seqwise(data["TLE"])
        gride = stack_seqwise(data["GRIDE"])
        loss = np.array(list(data["loss"])).reshape(-1)
        ess_chunks.append(ess)
        tle_chunks.append(tle)
        gride_chunks.append(gride)
        loss_chunks.append(loss)
        print(f"Loaded {f}: {ess.shape[0]} sequences")

    ESS = np.concatenate(ess_chunks, axis=0)
    TLE = np.concatenate(tle_chunks, axis=0)
    GRIDE = np.concatenate(gride_chunks, axis=0)
    LOSS = np.concatenate(loss_chunks, axis=0)

    out_path = os.path.join(folder, "aggregated_results.npz")
    np.savez_compressed(out_path, ESS=ESS, TLE=TLE, GRIDE=GRIDE, loss=LOSS)

    print(f"\n✅ Saved merged results to {out_path}")
    print(f"ESS   : {ESS.shape}")
    print(f"TLE   : {TLE.shape}")
    print(f"GRIDE : {GRIDE.shape}")
    print(f"loss  : {LOSS.shape}")

def main():
    args = parse_args()
    folder = os.path.join(args.root, "Pile-Structured", args.model)
    print(f"Aggregating from {folder}")
    aggregate(folder)

if __name__ == "__main__":
    main()
