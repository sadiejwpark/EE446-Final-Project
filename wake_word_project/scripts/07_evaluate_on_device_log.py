"""
Stage 6 -- Turn Serial Monitor logs from the Arduino sketch into the
rubric's required on-device evaluation tables (Section 4 of the plan /
Rubric Section 7).

The sketch prints two line types you need to capture (Arduino IDE ->
Tools -> Serial Monitor -> ... or better, use `arduino-cli monitor` /
a terminal redirect so you get a clean text file):

    INFER,<millis>,<pred_label>,<confidence>,<latency_us>
    TRIGGER,<millis>

Recommended test sessions (run each as its own recording + log file):
  1. "marvin" recall/FRR test: play >=15 isolated wake-word utterances in a
     quiet room, save the log, run this script with --session_type recall.
  2. Same again in a noisy room -- second recall/FRR number for comparison.
  3. False-accept test: play >=1 hour of continuous non-wake-word audio
     (other keywords + silence + background noise) in a quiet room, save the
     log, run with --session_type fa_hour --duration_hours 1.0.
  4. Repeat #3 in a noisy room.
  5. Per-class evaluation: run separate short sessions for each true class
     (marvin / unknown / silence) with >=10-15 samples each, run with
     --session_type per_class --true_label <label>, to build the confusion
     matrix and the required 10-instance table (log which TRIGGER/INFER
     lines correspond to which of your 10 chosen representative instances
     by timestamp).

Usage:
    python scripts/07_evaluate_on_device_log.py --log logs/quiet_marvin.txt \
        --session_type recall --true_label marvin --out results/eval_quiet_marvin.json

    python scripts/07_evaluate_on_device_log.py --log logs/quiet_fa.txt \
        --session_type fa_hour --duration_hours 1.0 --out results/eval_quiet_fa.json
"""
import argparse
import json
import re
import statistics
import sys

INFER_RE = re.compile(r"^INFER,(\d+),(\w+),([\-\d\.]+),(\d+)")
TRIGGER_RE = re.compile(r"^TRIGGER,(\d+)")


def parse_log(path):
    infers, triggers = [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = INFER_RE.match(line)
            if m:
                infers.append({
                    "millis": int(m.group(1)),
                    "pred_label": m.group(2),
                    "confidence": float(m.group(3)),
                    "latency_us": int(m.group(4)),
                })
                continue
            m = TRIGGER_RE.match(line)
            if m:
                triggers.append({"millis": int(m.group(1))})
    return infers, triggers


def latency_stats(infers):
    if not infers:
        return {}
    lat = sorted(i["latency_us"] for i in infers)
    n = len(lat)
    return {
        "n": n,
        "mean_us": round(statistics.mean(lat), 1),
        "median_us": round(statistics.median(lat), 1),
        "p95_us": lat[int(0.95 * (n - 1))],
        "max_us": lat[-1],
        "min_us": lat[0],
    }


def eval_recall(infers, triggers, true_label):
    """For a session where every played clip WAS the wake word: recall/FRR."""
    num_triggers = len(triggers)
    # crude but workable: assume one utterance -> at most one trigger, so
    # ask the user how many utterances were actually played via a manual
    # count; this script reports triggers observed, you supply n_played.
    return {
        "true_label": true_label,
        "num_triggers_observed": num_triggers,
        "latency": latency_stats(infers),
        "note": "Divide num_triggers_observed by the number of wake-word "
                "utterances you actually played to get recall; "
                "1 - recall = FRR. State the utterance count explicitly in your report.",
    }


def eval_fa_hour(infers, triggers, duration_hours):
    fa_per_hour = len(triggers) / duration_hours if duration_hours > 0 else float("nan")
    return {
        "duration_hours": duration_hours,
        "num_false_triggers": len(triggers),
        "false_accepts_per_hour": round(fa_per_hour, 3),
        "latency": latency_stats(infers),
    }


def eval_per_class(infers, true_label):
    from collections import Counter
    pred_counts = Counter(i["pred_label"] for i in infers)
    n = len(infers)
    correct = pred_counts.get(true_label, 0)
    return {
        "true_label": true_label,
        "n_samples": n,
        "pred_distribution": dict(pred_counts),
        "accuracy_this_class": round(correct / n, 4) if n else None,
        "latency": latency_stats(infers),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--session_type", choices=["recall", "fa_hour", "per_class"], required=True)
    ap.add_argument("--true_label", default=None)
    ap.add_argument("--duration_hours", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    infers, triggers = parse_log(args.log)
    print(f"Parsed {len(infers)} INFER lines and {len(triggers)} TRIGGER lines from {args.log}")

    if args.session_type == "recall":
        result = eval_recall(infers, triggers, args.true_label)
    elif args.session_type == "fa_hour":
        if args.duration_hours is None:
            sys.exit("--duration_hours is required for session_type=fa_hour")
        result = eval_fa_hour(infers, triggers, args.duration_hours)
    else:
        if args.true_label is None:
            sys.exit("--true_label is required for session_type=per_class")
        result = eval_per_class(infers, args.true_label)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
