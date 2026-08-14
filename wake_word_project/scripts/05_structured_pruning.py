"""
Stage 4 -- Structured pruning, run SEPARATELY from Stage 3 (MFCC ablation)
and SEPARATELY from Stage 2 (quantization), per the instructor feedback.

IMPORTANT TECHNICAL NOTE (put a version of this in your report -- it shows
you understand the distinction the feedback is asking for):

  TFMOT's `prune_low_magnitude` performs UNSTRUCTURED (element-wise) magnitude
  pruning: it zeroes individual weights but keeps the same dense tensor shapes.
  This helps FILE SIZE after compression (zeros compress well) but does NOT
  reduce actual FLOPs, RAM, or inference latency on a Cortex-M4 running
  TFLite Micro, because TFLite Micro's dense kernels don't skip zeros.

  STRUCTURED pruning -- removing entire filters/channels so the tensor
  shapes themselves shrink -- is what actually reduces RAM, flash, and
  latency on this hardware. This script implements structured pruning as
  filter-width reduction (retrain the same architecture with fewer filters
  per layer via `width_multiplier`), and ALSO runs TFMOT unstructured
  pruning as a labeled comparison row so you can show you evaluated both
  and explain why you chose the one you deployed.

Usage:
    python scripts/05_structured_pruning.py --manifest results/split_manifest.json \
        --model ds_cnn --num_mfcc 10 --baseline_model results/stage1_ds_cnn/float32_model.keras \
        --widths 1.0 0.75 0.5 0.35 --epochs 20 --out results/stage4_pruning
"""
import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from common import config as cfg
from common import dataset as ds_lib
from common import features
from common import models


def load_manifest(path):
    with open(path) as f:
        manifest = json.load(f)
    splits = {}
    for name, items in manifest["splits"].items():
        splits[name] = [(it["path"] if it["path"] != "__SILENCE__" else None, it["label"])
                         for it in items]
    return splits


def build_mfcc_dataset(file_label_pairs, split_name, noise_pool_paths, num_mfcc):
    raw_ds = ds_lib.make_dataset(file_label_pairs, split_name, noise_pool_paths)

    def to_mfcc(wav_batch, label_batch):
        mfcc = features.batch_waveform_to_mfcc(wav_batch, num_mfcc=num_mfcc)
        return mfcc[..., tf.newaxis], label_batch

    return raw_ds.map(to_mfcc, num_parallel_calls=tf.data.AUTOTUNE)


def run_structured_width_sweep(splits, args):
    """Primary structured-pruning technique: retrain at reduced filter widths."""
    noise_pool = list({p for p, l in splits["train"] if l == "silence" and p is not None})
    train_ds = build_mfcc_dataset(splits["train"], "train", noise_pool, args.num_mfcc)
    val_ds = build_mfcc_dataset(splits["val"], "val", None, args.num_mfcc)
    test_ds = build_mfcc_dataset(splits["test"], "test", None, args.num_mfcc)

    results = []
    for width in args.widths:
        print(f"\n=== Structured pruning: width_multiplier={width} ===")
        model = models.build_ds_cnn(num_mfcc=args.num_mfcc, width_multiplier=width)
        model.compile(optimizer=tf.keras.optimizers.Adam(cfg.LEARNING_RATE),
                       loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                   callbacks=[tf.keras.callbacks.EarlyStopping(
                       monitor="val_accuracy", patience=6, restore_best_weights=True)],
                   verbose=2)
        test_eval = model.evaluate(test_ds, verbose=0, return_dict=True)

        run_dir = os.path.join(args.out, f"structured_width_{width}")
        os.makedirs(run_dir, exist_ok=True)
        model_path = os.path.join(run_dir, "model.keras")
        model.save(model_path)

        results.append({
            "technique": "structured_width_reduction",
            "width_multiplier": width,
            "num_params": model.count_params(),
            "float32_size_kb": round(os.path.getsize(model_path) / 1024, 1),
            "test_accuracy": test_eval["accuracy"],
            "model_path": model_path,
        })
    return results


def run_unstructured_tfmot_comparison(splits, args):
    """Comparison-only: unstructured magnitude pruning via TFMOT.
    Requires the tf_keras legacy API (TFMOT is not Keras-3 compatible)."""
    try:
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        import tf_keras
        import tensorflow_model_optimization as tfmot
    except ImportError:
        print("\n[skip] tf_keras / tensorflow_model_optimization not installed -- "
              "pip install tf_keras tensorflow-model-optimization to enable this comparison.")
        return []

    from tf_keras import layers as klayers, models as kmodels

    def build_tfkeras_ds_cnn(num_frames, num_mfcc, num_classes):
        inputs = klayers.Input(shape=(num_frames, num_mfcc, 1))
        x = klayers.Conv2D(64, (10, 4), strides=(2, 2), padding="same", use_bias=False)(inputs)
        x = klayers.BatchNormalization()(x)
        x = klayers.ReLU()(x)
        for _ in range(4):
            x = klayers.DepthwiseConv2D(3, padding="same", use_bias=False)(x)
            x = klayers.BatchNormalization()(x)
            x = klayers.ReLU()(x)
            x = klayers.Conv2D(64, 1, padding="same", use_bias=False)(x)
            x = klayers.BatchNormalization()(x)
            x = klayers.ReLU()(x)
        x = klayers.GlobalAveragePooling2D()(x)
        x = klayers.Dropout(0.2)(x)
        outputs = klayers.Dense(num_classes, activation="softmax")(x)
        return kmodels.Model(inputs, outputs)

    noise_pool = list({p for p, l in splits["train"] if l == "silence" and p is not None})
    train_ds = build_mfcc_dataset(splits["train"], "train", noise_pool, args.num_mfcc)
    val_ds = build_mfcc_dataset(splits["val"], "val", None, args.num_mfcc)
    test_ds = build_mfcc_dataset(splits["test"], "test", None, args.num_mfcc)

    # convert tf.data datasets to numpy for tf_keras simplicity (small dataset, fine)
    def ds_to_numpy(ds):
        xs, ys = [], []
        for x, y in ds:
            xs.append(x.numpy())
            ys.append(y.numpy())
        return np.concatenate(xs), np.concatenate(ys)

    x_train, y_train = ds_to_numpy(train_ds)
    x_val, y_val = ds_to_numpy(val_ds)
    x_test, y_test = ds_to_numpy(test_ds)

    base = build_tfkeras_ds_cnn(cfg.NUM_FRAMES, args.num_mfcc, cfg.NUM_CLASSES)

    end_step = (len(x_train) // cfg.BATCH_SIZE) * args.epochs
    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0, final_sparsity=args.target_sparsity,
            begin_step=0, end_step=end_step)
    }
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(base, **pruning_params)
    pruned_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])
    pruned_model.fit(x_train, y_train, validation_data=(x_val, y_val),
                      batch_size=cfg.BATCH_SIZE, epochs=args.epochs,
                      callbacks=[tfmot.sparsity.keras.UpdatePruningStep()], verbose=2)

    stripped = tfmot.sparsity.keras.strip_pruning(pruned_model)
    # strip_pruning returns an uncompiled model -- must recompile before evaluate/save
    stripped.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
    test_loss, test_acc = stripped.evaluate(x_test, y_test, verbose=0)

    run_dir = os.path.join(args.out, "unstructured_tfmot")
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "model.h5")
    stripped.save(model_path)

    return [{
        "technique": "unstructured_magnitude_pruning_tfmot",
        "target_sparsity": args.target_sparsity,
        "test_accuracy": float(test_acc),
        "model_path": model_path,
        "note": "Element-wise sparsity; helps compressed file size, NOT on-device "
                "latency/RAM on TFLite Micro without sparse kernel support.",
    }]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", choices=["ds_cnn"], default="ds_cnn",
                     help="structured width-pruning implemented for ds_cnn")
    ap.add_argument("--num_mfcc", type=int, default=cfg.NUM_MFCC)
    ap.add_argument("--widths", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.35])
    ap.add_argument("--target_sparsity", type=float, default=0.5,
                     help="for the TFMOT unstructured comparison run")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--skip_unstructured_comparison", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    splits = load_manifest(args.manifest)

    structured_results = run_structured_width_sweep(splits, args)

    unstructured_results = []
    if not args.skip_unstructured_comparison:
        unstructured_results = run_unstructured_tfmot_comparison(splits, args)

    all_results = structured_results + unstructured_results
    with open(os.path.join(args.out, "pruning_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== Stage 4 pruning summary ===")
    for r in all_results:
        print(r)
    print(f"\nSaved to {os.path.join(args.out, 'pruning_summary.json')}")
    print("\nPick the best structured_width row as your final pre-quantization model,")
    print("then run 03_quantize.py on it to get your FINAL DEPLOYED model, and re-run")
    print("03_quantize.py's evaluation to fill in Rubric Table B's 'final deployed model' row.")


if __name__ == "__main__":
    main()
