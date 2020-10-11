import os
import tensorflow as tf
from tensorflow_io import experimental as tfex


def to_tensor(path, pad=True):
    # Create AudioTensor, normalized to (-1, 1)
    raw_audio = tf.io.read_file(path)
    tensor, sample_rate = tf.audio.decode_wav(raw_audio)
        
    # Flatten to one dimension
    tensor = tf.reshape(tensor, [-1])
    
    # Trim edges and zero-pad
    if pad:
        start, stop = tfex.audio.trim(tensor, axis=0, epsilon=0.005)
        tensor = tensor[:stop]
        tensor = tf.pad(tensor, tf.constant([[16_000 - tensor.shape[0], 0]]))
    return tensor


def to_label(path):
    return tf.strings.split(path, os.sep)[-2]


def to_binary_label(path, target='marvin'):
    return int(str.split(path, os.sep)[-2] == target)


def tensor_to_log_mel_spec(tensor):
    # Trim edges and zero-pad
    start, stop = tfex.audio.trim(tensor, axis=0, epsilon=0.005)
    tensor = tensor[:stop.numpy()]
    tensor = tf.pad(tensor, tf.constant([[16_000 - tensor.shape[0], 0]]))

    # Fade in/out
    tensor = tfex.audio.fade(tensor, fade_in=1000, fade_out=2000, mode="logarithmic")

    # Spectrogram (Discrete fourier transform)
    spectrogram = tfex.audio.spectrogram(tensor, nfft=512, window=512, stride=256)
    # Mel spectrogram
    mel_spectrogram = tfex.audio.melscale(spectrogram, rate=16000, mels=128, fmin=0, fmax=8000)

    # dB-scaled Mel spectrogram (axis0=time, axis1=freq)
    dbscale_mel_spectrogram = tfex.audio.dbscale(mel_spectrogram, top_db=80)
    return dbscale_mel_spectrogram.numpy()


def batch_to_log_mel_spec(batch):
    batch_size, num_samples, sample_rate = 10, 16000, 16000.0
    # A Tensor of [batch_size, num_samples] mono PCM samples in the range [-1, 1].
    #pcm = tf.random.normal([batch_size, num_samples], dtype=tf.float32)

    # A 1024-point STFT with frames of 64 ms and 75% overlap.
    stfts = tf.signal.stft(batch, frame_length=512, frame_step=256,
                           fft_length=512)
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 8000.0, 128
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    return log_mel_spectrograms
    

def batch_log_mel_spec_to_mfccs(batch):
    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
        batch)[..., :13]
    return mfccs
