"""
Single source of truth for every constant that affects the audio front-end
and label set. The Arduino sketch's mfcc.h mirrors these values exactly --
if you change anything here, change it there too, or train/deploy will
silently disagree with each other (the #1 cause of "works in Python, garbage
on-device" bugs in KWS projects).
"""

# ---- Audio / dataset ----
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
CLIP_LENGTH_SAMPLES = SAMPLE_RATE * CLIP_DURATION_MS // 1000  # 16000

# ---- MFCC front-end ----
FRAME_LENGTH_MS = 30
FRAME_STRIDE_MS = 20
FRAME_LENGTH_SAMPLES = SAMPLE_RATE * FRAME_LENGTH_MS // 1000   # 480
FRAME_STRIDE_SAMPLES = SAMPLE_RATE * FRAME_STRIDE_MS // 1000   # 320
FFT_LENGTH = 512                # next pow2 >= FRAME_LENGTH_SAMPLES
NUM_MEL_BINS = 40
NUM_MFCC = 13                    # default; 04_mfcc_ablation.py sweeps this
LOWER_EDGE_HERTZ = 20.0
UPPER_EDGE_HERTZ = 4000.0        # SAMPLE_RATE / 2 = 8000, keep margin

# number of MFCC frames produced per 1-second clip
NUM_FRAMES = 1 + (CLIP_LENGTH_SAMPLES - FRAME_LENGTH_SAMPLES) // FRAME_STRIDE_SAMPLES  # 49

# ---- Labels ----
WAKE_WORD = "marvin"
LABELS = [WAKE_WORD, "unknown", "silence"]
LABEL_TO_INDEX = {l: i for i, l in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

# the 34 non-target Speech Commands v2 keywords (v2 has 35 words total)
ALL_KEYWORDS = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five",
    "follow", "forward", "four", "go", "happy", "house", "learn", "left",
    "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila",
    "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes",
    "zero",
]
UNKNOWN_KEYWORDS = [w for w in ALL_KEYWORDS if w != WAKE_WORD]

# ---- Noise augmentation ----
SNR_LEVELS_DB = [20, 10, 5, 0]     # applied to TRAIN split only
NOISE_MIX_PROB = 0.8               # fraction of training clips that get noise mixed in

# ---- Unknown class sampling ----
# cap unknown class to roughly match target class size (proposal's original idea)
UNKNOWN_TO_TARGET_RATIO = 1.0

# ---- Training ----
BATCH_SIZE = 64
EPOCHS_BASELINE = 30
EPOCHS_FINETUNE = 10
LEARNING_RATE = 1e-3
