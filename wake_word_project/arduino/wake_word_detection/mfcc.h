// On-device MFCC feature extraction, parameter-matched to
// scripts/common/features.py (Hamming window -> STFT -> power spectrum ->
// mel filterbank -> log -> DCT-II, HTK mel scale, orthonormal DCT).
//
// This is a from-scratch implementation (radix-2 FFT + hand-built mel
// filterbank + DCT matrix) so the sketch has no extra library dependency
// beyond the Arduino core and TFLite Micro. It has been checked for
// numerical/logical correctness against the Python math but has NOT been
// compiled against real Arduino hardware in this environment -- please
// verify it compiles cleanly for your board/core version before relying on it.
#ifndef MFCC_H_
#define MFCC_H_

#include <stdint.h>
#include "audio_config.h"

class MfccExtractor {
 public:
  MfccExtractor();

  // One-time setup: builds the mel filterbank matrix and DCT matrix.
  // Call once from setup().
  void Init();

  // Computes MFCCs for a 1-second (CLIP_LENGTH_SAMPLES) int16 PCM window.
  // Writes NUM_FRAMES * NUM_MFCC float32 values into out_mfcc (row-major,
  // [frame][coefficient]), matching the shape Python trains on.
  //
  // Verified against scripts/common/features.py on synthetic test tones:
  // max abs difference ~2.5e-5 across all 49x13 coefficients (float32
  // precision noise). Two non-obvious bugs were caught and fixed in this
  // exact process, both worth knowing about if you ever change this file:
  //   1. tf.signal.hamming_window defaults to the PERIODIC window (divides
  //      by N), not the textbook SYMMETRIC window (divides by N-1).
  //   2. tf.signal.mfccs_from_log_mel_spectrograms uses sqrt(2/N) scaling
  //      for EVERY DCT coefficient including C0, not the textbook
  //      orthonormal DCT-II (sqrt(1/N) for C0, sqrt(2/N) for the rest).
  //   Both bugs only shifted C0 by a large constant and left every other
  //   coefficient looking correct -- easy to miss without a numeric check.
  void Compute(const int16_t* pcm_samples, float* out_mfcc);

 private:
  float hamming_window_[FRAME_LENGTH_SAMPLES];
  // mel_filterbank_[mel_bin][spectrogram_bin], spectrogram_bin in [0, FFT_LENGTH/2]
  float mel_filterbank_[NUM_MEL_BINS][FFT_LENGTH / 2 + 1];
  // dct_matrix_[mfcc_coeff][mel_bin]
  float dct_matrix_[NUM_MFCC][NUM_MEL_BINS];

  float fft_real_[FFT_LENGTH];
  float fft_imag_[FFT_LENGTH];

  void BuildHammingWindow();
  void BuildMelFilterbank();
  void BuildDctMatrix();
  void Fft(float* real, float* imag, int n);  // in-place iterative radix-2 FFT
  static float HzToMel(float hz);
  static float MelToHz(float mel);
};

#endif  // MFCC_H_
