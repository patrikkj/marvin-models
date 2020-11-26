import os
import tensorflow as tf
from tensorflow_io import experimental as tfex


def to_tensor(path, pad=False, pad_length=16_000):
    # Create AudioTensor, normalized to (-1, 1)
    raw_audio = tf.io.read_file(path)
    tensor, sample_rate = tf.audio.decode_wav(raw_audio)
        
    # Flatten to one dimension
    tensor = tf.reshape(tensor, [-1])
    
    # Trim edges and zero-pad
    if pad:
        start, stop = tfex.audio.trim(tensor, axis=0, epsilon=0.005)
        tensor = tensor[start:stop]
        if tensor.shape[0] > pad_length:
            print("ERROR: Tensor {tensor} is longer than pad_length, will be cropped.")
            tensor = tensor[:pad_length]
        tensor = tf.pad(tensor, tf.constant([[pad_length - tensor.shape[0], 0]]))
    return tensor


def to_label(path):
    return tf.strings.split(path, os.sep)[-2]


def to_binary_label(path, target='marvin'):
    return int(str.split(path, os.sep)[-2] == target)
