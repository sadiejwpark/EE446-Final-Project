"""
Stage 1 -- Float32 baseline training (DS-CNN and CNN-1D).

Usage:
    python scripts/02_train_baseline.py --manifest results/split_manifest.json \
        --model ds_cnn --out results/stage1_ds_cnn

Run twice (--model ds_cnn and --model cnn_1d) to get both comparison rows
for Rubric Table A. Whichever has better val/test performance becomes your
"selected baseline" for Stages 2-4.
"""
import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score

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


def build_mfcc_dataset(file_label_pairs, split_name, noise_pool_paths, num_mfcc, is_2d):
    raw_ds = ds_lib.make_dataset(file_label_pairs, split_name, noise_pool_paths)

    def to_mfcc(wav_batch, label_batch):
        mfcc = features.batch_waveform_to_mfcc(wav_batch, num_mfcc=num_mfcc)
        if is_2d:
            mfcc = mfcc[..., tf.newaxis]  # add channel dim for Conv2D
        return mfcc, label_batch

    return raw_ds.map(to_mfcc, num_parallel_calls=tf.data.AUTOTUNE)


def evaluate(model, test_ds):
    y_true, y_pred = [], []
    for x, y in test_ds:
        probs = model.predict(x, verbose=0)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())
    report = classification_report(y_true, y_pred, target_names=cfg.LABELS,
                                    output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(cfg.NUM_CLASSES)))
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"report": report, "confusion_matrix": cm.tolist(), "macro_f1": macro_f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", choices=["ds_cnn", "cnn_1d"], required=True)
    ap.add_argument("--num_mfcc", type=int, default=cfg.NUM_MFCC)
    ap.add_argument("--epochs", type=int, default=cfg.EPOCHS_BASELINE)
    ap.add_argument("--out", required=True, help="output dir for model + metrics")
    args = ap.parse_args()

    splits = load_manifest(args.manifest)

    # noise pool = the background-noise files referenced under the 'silence' label in train split
    noise_pool = list({p for p, l in splits["train"] if l == "silence" and p is not None})

    is_2d = (args.model == "ds_cnn")
    train_ds = build_mfcc_dataset(splits["train"], "train", noise_pool, args.num_mfcc, is_2d)
    val_ds = build_mfcc_dataset(splits["val"], "val", None, args.num_mfcc, is_2d)
    test_ds = build_mfcc_dataset(splits["test"], "test", None, args.num_mfcc, is_2d)

    if args.model == "ds_cnn":
        model = models.build_ds_cnn(num_mfcc=args.num_mfcc)
    else:
        model = models.build_cnn_1d(num_mfcc=args.num_mfcc)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    os.makedirs(args.out, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                          restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(args.out, "best.keras"),
                                            monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                         callbacks=callbacks)

    train_eval = model.evaluate(train_ds, verbose=0, return_dict=True)
    val_eval = model.evaluate(val_ds, verbose=0, return_dict=True)
    test_metrics = evaluate(model, test_ds)

    saved_model_path = os.path.join(args.out, "float32_model.keras")
    model.save(saved_model_path)

    results = {
        "model": args.model,
        "num_mfcc": args.num_mfcc,
        "num_params": model.count_params(),
        "train_accuracy": train_eval["accuracy"],
        "val_accuracy": val_eval["accuracy"],
        "test_accuracy": test_metrics["report"]["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_per_class": {
            label: test_metrics["report"][label] for label in cfg.LABELS
        },
        "confusion_matrix": test_metrics["confusion_matrix"],
        "labels_order": cfg.LABELS,
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Stage 1 baseline results ===")
    print(json.dumps({k: v for k, v in results.items() if k != "confusion_matrix"}, indent=2))
    print(f"\nSaved model to {saved_model_path}")
    print(f"Saved metrics to {os.path.join(args.out, 'metrics.json')}")
    print("\nThis JSON is what Rubric Table A gets filled in from -- keep it.")


if __name__ == "__main__":
    main()
