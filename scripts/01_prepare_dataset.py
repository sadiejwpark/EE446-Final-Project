"""
Stage 0/1 -- Dataset audit and split construction.

Usage:
    python scripts/01_prepare_dataset.py --data_dir data/speech_commands_v2

Downloads are NOT performed by this script (no internet access assumed).
Get the dataset first:
    wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
    mkdir -p data/speech_commands_v2
    tar -xzf speech_commands_v0.02.tar.gz -C data/speech_commands_v2

This script:
  1. Builds the official speaker-independent train/val/test split (Section 1
     of the plan -- required for the "avoided data leakage" rubric bullet).
  2. Reports exact per-class, per-split counts (don't hand-guess these numbers
     for your report -- use what this prints).
  3. Saves the split as a JSON manifest so every later stage uses the exact
     same split (important for fair comparison across Stage 1-4 models).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import dataset as ds_lib
from common import config as cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="path to extracted speech_commands_v2/")
    ap.add_argument("--out", default="results/split_manifest.json")
    args = ap.parse_args()

    splits = ds_lib.build_file_lists(args.data_dir)

    print(f"\nWake word: '{cfg.WAKE_WORD}'")
    print(f"{'Split':<8}{'target':<10}{'unknown':<10}{'silence':<10}{'total':<10}")
    manifest = {"wake_word": cfg.WAKE_WORD, "splits": {}}
    for split_name in ("train", "val", "test"):
        items = splits[split_name]
        counts = {"target": 0, "unknown": 0, "silence": 0}
        for _, label in items:
            key = "target" if label == cfg.WAKE_WORD else label
            counts[key] += 1
        print(f"{split_name:<8}{counts['target']:<10}{counts['unknown']:<10}"
              f"{counts['silence']:<10}{len(items):<10}")
        manifest["splits"][split_name] = [
            {"path": p, "label": l} for p, l in items
        ]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f)
    print(f"\nSaved split manifest to {args.out}")
    print("Use this exact file in 02_train_baseline.py, 04_mfcc_ablation.py, "
          "and 05_structured_pruning.py so every experiment is comparable.")


if __name__ == "__main__":
    main()
