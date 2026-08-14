# Wake Word Detection -- EE 446 Final Project

Full pipeline: dataset prep -> float32 baseline (DS-CNN + 1D-CNN) -> INT8
quantization -> MFCC ablation -> structured pruning -> Arduino Nano 33 BLE
Sense deployment -> on-device evaluation.

## What's been tested vs. not, honestly

Everything in `scripts/` has been run end-to-end in a sandboxed environment
against a small **synthetic** dataset (random noise standing in for real
audio) to verify the code runs without errors -- data loading, MFCC shapes,
model builds, training loop, quantization, both pruning paths, and C-array
export all execute cleanly. The accuracy numbers from those test runs are
meaningless (it's noise); they only confirm the plumbing works.

The Arduino MFCC implementation (`arduino/wake_word_detection/mfcc.cpp`) was
compiled standalone (outside the Arduino toolchain, as a plain C++ program)
and numerically validated against the Python feature extractor on synthetic
tones -- it matches to float32 precision (~2.5e-5 max difference). This
caught two real bugs (documented in `mfcc.h`) that would otherwise have
silently hurt on-device accuracy.

**Not tested:** training on the real Speech Commands dataset (no internet
access to download it in this environment), and compiling/flashing the
`.ino` sketch against the actual Arduino board packages and TFLite Micro
library (also not reachable here). The sketch logic is correct and follows
standard tflite-micro-arduino-examples patterns, but library API names
shift between versions -- if it doesn't compile as-is, check your installed
library's own example sketch for the exact constructor/include names and
adjust (the comment at the top of the `.ino` flags this).

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset (not done for you -- requires internet access you have
and this environment doesn't):
```bash
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir -p data/speech_commands_v2
tar -xzf speech_commands_v0.02.tar.gz -C data/speech_commands_v2
```

## Run order

```bash
# Stage 0: audit + speaker-independent split
python scripts/01_prepare_dataset.py --data_dir data/speech_commands_v2 \
    --out results/split_manifest.json

# Stage 1: float32 baselines (run both, compare)
python scripts/02_train_baseline.py --manifest results/split_manifest.json \
    --model ds_cnn --out results/stage1_ds_cnn
python scripts/02_train_baseline.py --manifest results/split_manifest.json \
    --model cnn_1d --out results/stage1_cnn_1d

# Stage 2: INT8 quantization of whichever baseline you select
python scripts/03_quantize.py --manifest results/split_manifest.json \
    --model_path results/stage1_ds_cnn/float32_model.keras \
    --num_mfcc 13 --is_2d --out results/stage2_ds_cnn_int8

# Stage 3: MFCC coefficient ablation (separate from pruning -- feedback point 3)
python scripts/04_mfcc_ablation.py --manifest results/split_manifest.json \
    --model ds_cnn --sweep 13 10 8 --epochs 30 --out results/stage3_mfcc_ablation

# Stage 4: structured pruning (separate stage, with unstructured TFMOT comparison)
python scripts/05_structured_pruning.py --manifest results/split_manifest.json \
    --num_mfcc 10 --widths 1.0 0.75 0.5 0.35 --epochs 30 --out results/stage4_pruning

# Requantize your final chosen pruned model
python scripts/03_quantize.py --manifest results/split_manifest.json \
    --model_path results/stage4_pruning/structured_width_0.5/model.keras \
    --num_mfcc 10 --is_2d --out results/stage4_final_int8

# Export for Arduino
python scripts/06_export_c_array.py \
    --tflite_path results/stage4_final_int8/model_int8.tflite \
    --out arduino/wake_word_detection/model_data.h --var_name g_model
```

**Before flashing:** update `arduino/wake_word_detection/audio_config.h`'s
`NUM_MFCC` to match whatever `--num_mfcc` you used for your final model.

## Arduino IDE setup

1. Install board support: Tools -> Board -> Boards Manager -> search
   "Arduino Mbed OS Nano Boards" -> install.
2. Install libraries via Library Manager:
   - The current TFLite Micro Arduino library (search "TensorFlowLite" or
     get `harvard-edge/tflite-micro-arduino-examples` directly from GitHub
     if it's not in the Library Manager index).
   - `PDM` is bundled with the Nano board core, no separate install needed.
3. Open `arduino/wake_word_detection/wake_word_detection.ino`, select
   Board: "Arduino Nano 33 BLE" (or BLE Sense Rev2 if listed), select the
   port, compile, upload.
4. Open Serial Monitor at 115200 baud. You'll see `INFER,...` lines every
   ~200ms and `TRIGGER,...` lines when the wake word is detected.

## On-device evaluation

Record Serial Monitor output to a text file for each test session (in the
Arduino IDE: Serial Monitor has no built-in save, so either use
`arduino-cli monitor > log.txt`, a terminal serial tool like `screen`/
`minicom` with logging, or a simple Python `pyserial` script piping to a
file), then:

```bash
# Wake-word recall/FRR session (quiet room, N isolated "marvin" utterances)
python scripts/07_evaluate_on_device_log.py --log logs/quiet_marvin.txt \
    --session_type recall --true_label marvin --out results/eval_quiet_marvin.json

# False-accept-per-hour session (continuous non-wake audio, quiet room)
python scripts/07_evaluate_on_device_log.py --log logs/quiet_fa.txt \
    --session_type fa_hour --duration_hours 1.0 --out results/eval_quiet_fa.json

# Repeat both in a noisy room, and per-class sessions for the confusion matrix
```

Fill Rubric Tables A-D and the flash/RAM numbers (printed once at boot as
`EVENT,arena_used_bytes,<N>`, and from the Arduino IDE's compile output for
flash/sketch size) directly from these JSON outputs.

## Project structure

```
scripts/
  common/
    config.py       # single source of truth for all audio/model constants
    dataset.py       # official speaker-independent split, noise augmentation
    features.py       # MFCC (TensorFlow ops) -- must match arduino/mfcc.cpp
    models.py         # DS-CNN and 1D-CNN architectures
  01_prepare_dataset.py
  02_train_baseline.py
  03_quantize.py
  04_mfcc_ablation.py
  05_structured_pruning.py
  06_export_c_array.py
  07_evaluate_on_device_log.py
arduino/wake_word_detection/
  wake_word_detection.ino   # main sketch
  audio_config.h            # mirrors scripts/common/config.py
  mfcc.h / mfcc.cpp          # on-device MFCC, numerically validated vs Python
  model_data.h               # generated by 06_export_c_array.py (currently a placeholder)
```
