import tensorflow as tf
from tensorflow_io import experimental as tfex


class Spectrogram(tf.keras.layers.Layer):
    """Converts batches of normalized audio tensors to spectrograms."""
    def __init__(self, params, hparams, **kwargs):
        super().__init__(**kwargs)
        all_params = {**params , **hparams}
        self.frame_size = all_params['frame_size']
        self.frame_step = all_params['frame_step']
        self.fft_size = all_params['fft_size']

    def call(self, audio_tensors):
        # Waveform input shape: (batch_size, num_samples) in the range [-1, 1]
        stfts = tf.signal.stft(
            audio_tensors, 
            frame_length=self.frame_size, 
            frame_step=self.frame_step,
            fft_length=self.fft_size
        )
        # The STFT returns a real (magnitude) and complex (phase) component
        return tf.abs(stfts)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'frame_size': self.frame_size,
            'frame_step': self.frame_step,
            'fft_size': self.fft_size,
        })
        return config


class MelSpectrogram(tf.keras.layers.Layer):
    def __init__(self, params, hparams, **kwargs):
        super().__init__(**kwargs)
        all_params = {**params , **hparams}
        self.sample_rate = all_params['sample_rate']
        self.fft_size = all_params['fft_size']
        self.mel_bins = all_params['mel_bins']
        self.min_freq = all_params['min_freq']
        self.max_freq = all_params['max_freq']
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.mel_bins,
            num_spectrogram_bins=self.fft_size // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.min_freq,
            upper_edge_hertz=self.max_freq
        )

    def build(self, input_shape):
        self.non_trainable_weights.append(self.mel_filterbank)
        super(MelSpectrogram, self).build(input_shape)

    def call(self, spectrograms):
        # Warp the linear scale spectrograms into the mel-scale.
        return tf.tensordot(spectrograms, self.mel_filterbank, 1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'fft_size': self.fft_size,
            'mel_bins': self.mel_bins,
            'sample_rate': self.sample_rate,
            'min_freq': self.min_freq,
            'max_freq': self.max_freq,
        })
        return config


class LogMelSpectrogram(tf.keras.layers.Layer):
    def __init__(self, params, hparams, **kwargs):
        super().__init__(**kwargs)

    def call(self, mel_spectrograms):
        return tf.math.log(mel_spectrograms + 1e-6)

    def get_config(self):
        return super().get_config().copy()


class DbMelSpectrogram(tf.keras.layers.Layer):
    def __init__(self, params, hparams, **kwargs):
        super().__init__(**kwargs)

    def call(self, mel_spectrograms):
        return tfex.audio.dbscale(mel_spectrograms, 80)

    def get_config(self):
        return super().get_config().copy()


class MFCC(tf.keras.layers.Layer):
    def __init__(self, params, hparams, **kwargs):
        super().__init__(**kwargs)
        all_params = {**params , **hparams}
        self.num_mfccs = all_params['num_mfccs']

    def call(self, log_mel_spectrograms):
        return tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :self.num_mfccs]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_mfccs': self.num_mfccs
        })
        return config


class TimeMask(tf.keras.layers.Layer):
    def __init__(self, params, hparams, **kwargs):
        super().__init__(**kwargs)
        all_params = {**params , **hparams}
        self.max_time_mask = all_params['max_time_mask']

    def call(self, spectrogram, training=None):
        # Apply data augmentation to training phase only
        if not training:
            return spectrogram

        time_max = tf.shape(spectrogram)[1]
        t = tf.random.uniform(shape=(), minval=0, maxval=self.max_time_mask, dtype=tf.dtypes.int32)
        t0 = tf.random.uniform(shape=(), minval=0, maxval=time_max - t, dtype=tf.dtypes.int32)
        indices = tf.reshape(tf.range(time_max), (1, -1, 1))
        condition = tf.math.logical_and(tf.math.greater_equal(indices, t0), tf.math.less(indices, t0 + t))
        return tf.where(condition, 0.0, spectrogram)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'max_time_mask': self.max_time_mask
        })
        return config


class FrequencyMask(tf.keras.layers.Layer):
    def __init__(self, params, hparams, **kwargs):
        super().__init__(**kwargs)
        all_params = {**params , **hparams}
        self.max_freq_mask = all_params['max_freq_mask']

    def call(self, spectrogram, training=None):
        # Apply data augmentation to training phase only
        if not training:
            return spectrogram

        freq_max = tf.shape(spectrogram)[2]
        f = tf.random.uniform(shape=(), minval=0, maxval=self.max_freq_mask, dtype=tf.dtypes.int32)
        f0 = tf.random.uniform(shape=(), minval=0, maxval=freq_max - f, dtype=tf.dtypes.int32)
        indices = tf.reshape(tf.range(freq_max), (1, 1, -1))
        condition = tf.math.logical_and(tf.math.greater_equal(indices, f0), tf.math.less(indices, f0 + f))
        return tf.where(condition, 0.0, spectrogram)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'max_freq_mask': self.max_freq_mask
        })
        return config
