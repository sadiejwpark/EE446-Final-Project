#include "mfcc.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

MfccExtractor::MfccExtractor() {}

void MfccExtractor::Init() {
  BuildHammingWindow();
  BuildMelFilterbank();
  BuildDctMatrix();
}

// NOTE: tf.signal.hamming_window defaults to periodic=True (divides by N,
// not N-1). Must match here or the C0/log-energy coefficient will be off by
// a constant per-frame offset (shape-only coefficients are unaffected since
// it's a pure amplitude/DC scaling of the window).
void MfccExtractor::BuildHammingWindow() {
  for (int i = 0; i < FRAME_LENGTH_SAMPLES; i++) {
    hamming_window_[i] =
        0.54f - 0.46f * cosf(2.0f * M_PI * i / FRAME_LENGTH_SAMPLES);
  }
}

float MfccExtractor::HzToMel(float hz) {
  return 1127.0f * logf(1.0f + hz / 700.0f);
}

float MfccExtractor::MelToHz(float mel) {
  return 700.0f * (expf(mel / 1127.0f) - 1.0f);
}

// Matches tf.signal.linear_to_mel_weight_matrix: triangular filters spaced
// evenly in MEL frequency, weights computed via linear interpolation in
// mel-space between (lower_edge, center, upper_edge) for each filter.
void MfccExtractor::BuildMelFilterbank() {
  const int num_spec_bins = FFT_LENGTH / 2 + 1;
  const float mel_low = HzToMel(LOWER_EDGE_HERTZ);
  const float mel_high = HzToMel(UPPER_EDGE_HERTZ);

  float band_edges_mel[NUM_MEL_BINS + 2];
  for (int i = 0; i < NUM_MEL_BINS + 2; i++) {
    band_edges_mel[i] = mel_low + (mel_high - mel_low) * i / (NUM_MEL_BINS + 1);
  }

  float bin_mel[FFT_LENGTH / 2 + 1];
  for (int j = 0; j < num_spec_bins; j++) {
    float bin_freq_hz = (float)j * SAMPLE_RATE_HZ / FFT_LENGTH;
    bin_mel[j] = HzToMel(bin_freq_hz);
  }

  for (int m = 0; m < NUM_MEL_BINS; m++) {
    float lower_edge = band_edges_mel[m];
    float center = band_edges_mel[m + 1];
    float upper_edge = band_edges_mel[m + 2];
    for (int j = 0; j < num_spec_bins; j++) {
      float lower_slope = (bin_mel[j] - lower_edge) / (center - lower_edge);
      float upper_slope = (upper_edge - bin_mel[j]) / (upper_edge - center);
      float weight = fminf(lower_slope, upper_slope);
      if (weight < 0.0f) weight = 0.0f;
      mel_filterbank_[m][j] = weight;
    }
  }
}

// DCT-II matching tf.signal.mfccs_from_log_mel_spectrograms EXACTLY.
// NOTE: this is NOT the textbook orthonormal DCT-II (which uses sqrt(1/N)
// for k=0 and sqrt(2/N) for k>0) -- TF's implementation uses sqrt(2/N) for
// EVERY coefficient, including k=0. Verified empirically against
// tf.signal.mfccs_from_log_mel_spectrograms; using the "textbook" formula
// here silently shifts C0 by a large constant and nothing else, which is a
// very easy bug to miss since every other coefficient still looks correct.
void MfccExtractor::BuildDctMatrix() {
  const int N = NUM_MEL_BINS;
  const float c_k = sqrtf(2.0f / N);
  for (int k = 0; k < NUM_MFCC; k++) {
    for (int n = 0; n < N; n++) {
      dct_matrix_[k][n] = c_k * cosf((float)M_PI / N * (n + 0.5f) * k);
    }
  }
}

// Iterative in-place radix-2 Cooley-Tukey FFT. n must be a power of 2.
void MfccExtractor::Fft(float* real, float* imag, int n) {
  // bit-reversal permutation
  int j = 0;
  for (int i = 0; i < n - 1; i++) {
    if (i < j) {
      float tr = real[i]; real[i] = real[j]; real[j] = tr;
      float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
    }
    int m = n >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }

  for (int len = 2; len <= n; len <<= 1) {
    float ang = -2.0f * (float)M_PI / len;
    float wr = cosf(ang), wi = sinf(ang);
    for (int i = 0; i < n; i += len) {
      float cur_wr = 1.0f, cur_wi = 0.0f;
      for (int k = 0; k < len / 2; k++) {
        int a = i + k;
        int b = i + k + len / 2;
        float tr = real[b] * cur_wr - imag[b] * cur_wi;
        float ti = real[b] * cur_wi + imag[b] * cur_wr;
        real[b] = real[a] - tr;
        imag[b] = imag[a] - ti;
        real[a] += tr;
        imag[a] += ti;
        float next_wr = cur_wr * wr - cur_wi * wi;
        float next_wi = cur_wr * wi + cur_wi * wr;
        cur_wr = next_wr;
        cur_wi = next_wi;
      }
    }
  }
}

void MfccExtractor::Compute(const int16_t* pcm_samples, float* out_mfcc) {
  const int num_spec_bins = FFT_LENGTH / 2 + 1;
  float power[FFT_LENGTH / 2 + 1];
  float mel_energies[NUM_MEL_BINS];
  float log_mel[NUM_MEL_BINS];

  for (int f = 0; f < NUM_FRAMES; f++) {
    int start = f * FRAME_STRIDE_SAMPLES;

    for (int i = 0; i < FRAME_LENGTH_SAMPLES; i++) {
      float sample = pcm_samples[start + i] / 32768.0f;
      fft_real_[i] = sample * hamming_window_[i];
      fft_imag_[i] = 0.0f;
    }
    for (int i = FRAME_LENGTH_SAMPLES; i < FFT_LENGTH; i++) {
      fft_real_[i] = 0.0f;
      fft_imag_[i] = 0.0f;
    }

    Fft(fft_real_, fft_imag_, FFT_LENGTH);

    for (int j = 0; j < num_spec_bins; j++) {
      power[j] = fft_real_[j] * fft_real_[j] + fft_imag_[j] * fft_imag_[j];
    }

    for (int m = 0; m < NUM_MEL_BINS; m++) {
      float sum = 0.0f;
      for (int j = 0; j < num_spec_bins; j++) {
        sum += power[j] * mel_filterbank_[m][j];
      }
      mel_energies[m] = sum;
      log_mel[m] = logf(mel_energies[m] + 1e-6f);
    }

    for (int k = 0; k < NUM_MFCC; k++) {
      float sum = 0.0f;
      for (int m = 0; m < NUM_MEL_BINS; m++) {
        sum += log_mel[m] * dct_matrix_[k][m];
      }
      out_mfcc[f * NUM_MFCC + k] = sum;
    }
  }
}
