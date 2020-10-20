import tensorflow as tf

class LMS(tf.keras.layers.Layer):
    def __init__(self, params, hparams, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = params['sample_rate']
        self.frame_size = hparams['frame_size']
        self.frame_step = hparams['frame_step']
        self.fft_size = hparams['fft_size']
        self.mel_bins = hparams['mel_bins']
        self.min_freq = params['min_freq']
        self.max_freq = params['max_freq']
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.mel_bins,
            num_spectrogram_bins=self.fft_size // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.min_freq,
            upper_edge_hertz=self.max_freq
        )

    def build(self, input_shape):
        self.non_trainable_weights.append(self.mel_filterbank)
        super(LMS, self).build(input_shape)

    def call(self, waveforms):
        # Waveform input shape: (batch_size, num_samples) in the range [-1, 1]
        stfts = tf.signal.stft(
            waveforms, 
            frame_length=self.fft_size, 
            frame_step=self.frame_step,
            fft_length=self.fft_size
        )

        # The STFT returns a real (magnitude) and complex (phase) component
        spectrograms = tf.abs(stfts)

        # Warp the linear scale spectrograms into the mel-scale.
        mel_spectrograms = tf.tensordot(spectrograms, self.mel_filterbank, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(self.mel_filterbank.shape[-1:]))

        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        return log_mel_spectrograms

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'sample_rate': self.sample_rate,
            'frame_size': self.frame_size,
            'frame_step': self.frame_step,
            'fft_size': self.fft_size,
            'mel_bins': self.mel_bins,
            'min_freq': self.min_freq,
            'max_freq': self.max_freq,
        })
        return config


class Spectrogram(tf.keras.layers.Layer):
    """Converts batches of normalized audio tensors to spectrograms."""
    def __init__(self, params, hparams, **kwargs):
        super().__init__(**kwargs)
        self.frame_size = hparams['frame_size']
        self.frame_step = hparams['frame_step']
        self.fft_size = hparams['fft_size']

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
        self.sample_rate = params['sample_rate']
        self.fft_size = hparams['fft_size']
        self.mel_bins = hparams['mel_bins']
        self.min_freq = params['min_freq']
        self.max_freq = params['max_freq']
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
        mel_spectrograms = tf.tensordot(spectrograms, self.mel_filterbank, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(self.mel_filterbank.shape[-1:]))
        return mel_spectrograms

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
        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        return tf.math.log(mel_spectrograms + 1e-6)

    def get_config(self):
        return super().get_config().copy()