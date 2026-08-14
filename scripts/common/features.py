"""
MFCC feature extraction (TensorFlow ops).

This MUST stay parameter-identical to arduino/wake_word_detection/mfcc.cpp.
Both implement: Hamming-windowed STFT -> power spectrum -> mel filterbank ->
log -> DCT-II, truncated to NUM_MFCC coefficients (including C0).
"""
import tensorflow as tf
from . import config as cfg


def waveform_to_mfcc(waveform, num_mfcc=None):
    """
    waveform: float32 tensor, shape [CLIP_LENGTH_SAMPLES], values in [-1, 1]
    returns: float32 tensor, shape [NUM_FRAMES, num_mfcc]
    """
    if num_mfcc is None:
        num_mfcc = cfg.NUM_MFCC

    stfts = tf.signal.stft(
        waveform,
        frame_length=cfg.FRAME_LENGTH_SAMPLES,
        frame_step=cfg.FRAME_STRIDE_SAMPLES,
        fft_length=cfg.FFT_LENGTH,
        window_fn=tf.signal.hamming_window,
    )
    power_spectrograms = tf.math.real(stfts * tf.math.conj(stfts))

    num_spectrogram_bins = stfts.shape[-1]
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=cfg.NUM_MEL_BINS,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=cfg.SAMPLE_RATE,
        lower_edge_hertz=cfg.LOWER_EDGE_HERTZ,
        upper_edge_hertz=cfg.UPPER_EDGE_HERTZ,
    )
    mel_spectrograms = tf.tensordot(power_spectrograms, mel_weight_matrix, 1)
    mel_spectrograms.set_shape(
        power_spectrograms.shape[:-1].concatenate(mel_weight_matrix.shape[-1:])
    )

    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
    mfccs = mfccs[..., :num_mfcc]
    return mfccs


def batch_waveform_to_mfcc(waveforms, num_mfcc=None):
    """waveforms: [B, CLIP_LENGTH_SAMPLES] -> [B, NUM_FRAMES, num_mfcc]"""
    return tf.map_fn(
        lambda w: waveform_to_mfcc(w, num_mfcc=num_mfcc),
        waveforms,
        fn_output_signature=tf.float32,
    )


if __name__ == "__main__":
    # quick self-check
    import numpy as np
    fake = tf.constant(np.random.uniform(-1, 1, cfg.CLIP_LENGTH_SAMPLES).astype("float32"))
    out = waveform_to_mfcc(fake)
    print("MFCC output shape:", out.shape, "expected:", (cfg.NUM_FRAMES, cfg.NUM_MFCC))
    assert out.shape[0] == cfg.NUM_FRAMES
    assert out.shape[1] == cfg.NUM_MFCC
    print("OK")
