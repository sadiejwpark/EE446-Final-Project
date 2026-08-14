// Mirrors scripts/common/config.py EXACTLY. If you change one, change both,
// or the model trained in Python will disagree with what's computed on-device.
#ifndef AUDIO_CONFIG_H_
#define AUDIO_CONFIG_H_

#define SAMPLE_RATE_HZ        16000
#define CLIP_DURATION_MS      1000
#define CLIP_LENGTH_SAMPLES   16000   // SAMPLE_RATE_HZ * CLIP_DURATION_MS / 1000

#define FRAME_LENGTH_MS       30
#define FRAME_STRIDE_MS       20
#define FRAME_LENGTH_SAMPLES  480     // SAMPLE_RATE_HZ * FRAME_LENGTH_MS / 1000
#define FRAME_STRIDE_SAMPLES  320     // SAMPLE_RATE_HZ * FRAME_STRIDE_MS / 1000
#define FFT_LENGTH             512    // next pow2 >= FRAME_LENGTH_SAMPLES
#define NUM_MEL_BINS            40
#define NUM_MFCC                10    // MUST match --num_mfcc used to train/export the deployed model
#define LOWER_EDGE_HERTZ       20.0f
#define UPPER_EDGE_HERTZ     4000.0f

#define NUM_FRAMES               49   // 1 + (CLIP_LENGTH_SAMPLES - FRAME_LENGTH_SAMPLES) / FRAME_STRIDE_SAMPLES

// Labels, in the SAME order as cfg.LABELS in Python (index = model output index)
#define NUM_CLASSES 3
static const char* const kLabels[NUM_CLASSES] = {"marvin", "unknown", "silence"};
#define WAKE_WORD_CLASS_INDEX 0

// Inference cadence: how often we run inference on the most recent 1s window.
// Smaller = lower latency to detect the word, higher = less CPU/power use.
#define INFERENCE_INTERVAL_MS 200

// Detection threshold + debounce, tuned during your on-device evaluation
// (Section 4 of the plan) -- start here and adjust based on your FA/FR tradeoff.
#define DETECTION_THRESHOLD    0.80f
#define DEBOUNCE_MS            1200

#endif  // AUDIO_CONFIG_H_
