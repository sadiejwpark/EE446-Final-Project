"""
Stage 2 -- Post-training INT8 quantization of the Stage 1 float32 baseline.

Usage:
    python scripts/03_quantize.py --manifest results/split_manifest.json \
        --model_path results/stage1_ds_cnn/float32_model.keras \
        --num_mfcc 13 --is_2d --out results/stage2_ds_cnn_int8

Keep this as its OWN stage, separate from pruning and from MFCC-coefficient
changes (Point 3 of the instructor feedback) -- report its accuracy/size
delta versus the float32 baseline on its own row.
"""
import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from common import config as cfg
from common import dataset as ds_lib
from common import features


def load_manifest(path):
    with open(path) as f:
        manifest = json.load(f)
    splits = {}
    for name, items in manifest["splits"].items():
        splits[name] = [(it["path"] if it["path"] != "__SILENCE__" else None, it["label"])
                         for it in items]
    return splits


def representative_dataset_gen(train_pairs, num_mfcc, is_2d, num_samples=200):
    raw_ds = ds_lib.make_dataset(train_pairs, "train", None, batch_size=1)
    count = 0
    for wav, _ in raw_ds:
        if count >= num_samples:
            break
        mfcc = features.waveform_to_mfcc(wav[0], num_mfcc=num_mfcc)
        if is_2d:
            mfcc = mfcc[..., tf.newaxis]
        mfcc = tf.expand_dims(mfcc, 0)
        yield [mfcc.numpy().astype("float32")]
        count += 1


def quantize(keras_model_path, train_pairs, num_mfcc, is_2d):
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(
        train_pairs, num_mfcc, is_2d
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    # Arduino_TensorFlowLite@2.4.0-ALPHA's kernels don't reliably execute
    # per-channel quantized Conv2D/DepthwiseConv2D (TF's default since ~2.7).
    # Force per-tensor quantization instead so inference actually works on
    # this library version.
    converter._experimental_disable_per_channel_quantization = True
    tflite_model = converter.convert()
    return tflite_model, model


def evaluate_tflite(tflite_bytes, test_pairs, num_mfcc, is_2d):
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    in_scale, in_zero = input_details["quantization"]

    raw_ds = ds_lib.make_dataset(test_pairs, "test", None, batch_size=1)
    y_true, y_pred = [], []
    for wav, label in raw_ds:
        mfcc = features.waveform_to_mfcc(wav[0], num_mfcc=num_mfcc)
        if is_2d:
            mfcc = mfcc[..., tf.newaxis]
        mfcc = tf.expand_dims(mfcc, 0).numpy()
        mfcc_q = (mfcc / in_scale + in_zero).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], mfcc_q)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])
        y_pred.append(int(np.argmax(out)))
        y_true.append(int(label.numpy()[0]))

    report = classification_report(y_true, y_pred, target_names=cfg.LABELS,
                                    output_dict=True, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return report, macro_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model_path", required=True, help="path to Stage 1 .keras model")
    ap.add_argument("--num_mfcc", type=int, default=cfg.NUM_MFCC)
    ap.add_argument("--is_2d", action="store_true", help="set for ds_cnn, unset for cnn_1d")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    splits = load_manifest(args.manifest)
    os.makedirs(args.out, exist_ok=True)

    tflite_bytes, float_model = quantize(args.model_path, splits["train"], args.num_mfcc, args.is_2d)

    tflite_path = os.path.join(args.out, "model_int8.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)

    float_size_kb = os.path.getsize(args.model_path) / 1024
    int8_size_kb = len(tflite_bytes) / 1024

    report, macro_f1 = evaluate_tflite(tflite_bytes, splits["test"], args.num_mfcc, args.is_2d)

    results = {
        "float32_model_path": args.model_path,
        "float32_size_kb": round(float_size_kb, 1),
        "int8_size_kb": round(int8_size_kb, 1),
        "size_reduction_pct": round(100 * (1 - int8_size_kb / float_size_kb), 1),
        "int8_test_accuracy": report["accuracy"],
        "int8_test_macro_f1": macro_f1,
        "int8_per_class": {label: report[label] for label in cfg.LABELS},
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nSaved INT8 model to {tflite_path}")
    print("Next: scripts/06_export_c_array.py to turn this into a C header for Arduino,")
    print("or continue to 04_mfcc_ablation.py / 05_structured_pruning.py first.")


if __name__ == "__main__":
    main()
