"""
Stage 3 -- MFCC coefficient ablation.

This is explicitly a FEATURE-REPRESENTATION experiment, not model pruning
(this distinction is exactly what your instructor's feedback flagged as
wrong in the original proposal). It changes the INPUT tensor size, not the
model's internal filter counts.

Retrains the same architecture at each num_mfcc value in --sweep, from
scratch, and reports the accuracy/size tradeoff.

Usage:
    python scripts/04_mfcc_ablation.py --manifest results/split_manifest.json \
        --model ds_cnn --sweep 13 10 8 --epochs 20 --out results/stage3_mfcc_ablation
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", choices=["ds_cnn", "cnn_1d"], required=True)
    ap.add_argument("--sweep", type=int, nargs="+", default=[13, 10, 8])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    summary = []

    for num_mfcc in args.sweep:
        run_dir = os.path.join(args.out, f"mfcc_{num_mfcc}")
        print(f"\n=== Training {args.model} with num_mfcc={num_mfcc} ===")
        subprocess.run([
            sys.executable, os.path.join(os.path.dirname(__file__), "02_train_baseline.py"),
            "--manifest", args.manifest,
            "--model", args.model,
            "--num_mfcc", str(num_mfcc),
            "--epochs", str(args.epochs),
            "--out", run_dir,
        ], check=True)

        with open(os.path.join(run_dir, "metrics.json")) as f:
            metrics = json.load(f)
        model_size_kb = os.path.getsize(os.path.join(run_dir, "float32_model.keras")) / 1024
        summary.append({
            "num_mfcc": num_mfcc,
            "test_accuracy": metrics["test_accuracy"],
            "test_macro_f1": metrics["test_macro_f1"],
            "num_params": metrics["num_params"],
            "float32_size_kb": round(model_size_kb, 1),
        })

    with open(os.path.join(args.out, "ablation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== MFCC coefficient ablation summary (Stage 3) ===")
    print(f"{'num_mfcc':<10}{'test_acc':<12}{'macro_f1':<12}{'params':<10}{'size_kb':<10}")
    for row in summary:
        print(f"{row['num_mfcc']:<10}{row['test_accuracy']:<12.4f}"
              f"{row['test_macro_f1']:<12.4f}{row['num_params']:<10}{row['float32_size_kb']:<10}")
    print(f"\nSaved to {os.path.join(args.out, 'ablation_summary.json')}")
    print("Pick the smallest num_mfcc that doesn't meaningfully hurt accuracy -- ")
    print("that becomes the MFCC config you carry into Stage 4 (pruning) and deployment.")


if __name__ == "__main__":
    main()
