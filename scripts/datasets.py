import numpy as np
import tensorflow as tf


def build_dataset(pos_data, pos_labels, neg_data, neg_labels, split):
    split_array = np.array(split)

    # Actual and desired class distributions
    pos_size, neg_size = pos_labels.shape[0], neg_labels.shape[0]
    pos_indices = np.ceil(split_array.cumsum() * pos_size).astype(int)
    neg_indices = np.ceil(split_array.cumsum() * neg_size).astype(int)

    # Stratified sampling to handle class imbalances
    train_data_pos,  val_data_pos, test_data_pos = np.vsplit(pos_data, pos_indices[:-1])
    train_data_neg,  val_data_neg, test_data_neg = np.vsplit(neg_data, neg_indices[:-1])

    train_labels_pos,  val_labels_pos, test_labels_pos = np.split(pos_labels, pos_indices[:-1])
    train_labels_neg,  val_labels_neg, test_labels_neg = np.split(neg_labels, neg_indices[:-1])

    # Free up memory
    pos_data, pos_labels = None, None
    neg_data, neg_labels = None, None

    # Constants
    train_size = train_data_pos.shape[0] + train_data_neg.shape[0]
    val_size = val_data_pos.shape[0] + val_data_neg.shape[0]
    test_size = test_data_pos.shape[0] + test_data_neg.shape[0]

    train_data = (train_data_pos, train_data_neg)
    train_labels = (train_labels_pos, train_labels_neg)
    train_arrays =  (train_data, train_labels)

    val_data = (val_data_pos, val_data_neg)
    val_labels = (val_labels_pos, val_labels_neg)
    val_arrays =  (val_data, val_labels)

    test_data = (test_data_pos, test_data_neg)
    test_labels = (test_labels_pos, test_labels_neg)
    test_arrays =  (test_data, test_labels)
    return (train_arrays, val_arrays, test_arrays), (train_size, val_size, test_size)


def resample_dataset(arrays, batch_size, positive_weight=0.3): 
    # Unpack array reference
    train_arrays, val_arrays, test_arrays = arrays
    train_data, train_labels = train_arrays
    val_data, val_labels = val_arrays
    test_data, test_labels = test_arrays

    # Create rebalanced training set
    pos_dataset = tf.data.Dataset.from_tensor_slices((train_data[0], train_labels[0]))
    pos_dataset = pos_dataset.shuffle(1024).repeat()
    neg_dataset = tf.data.Dataset.from_tensor_slices((train_data[1], train_labels[1]))
    neg_dataset = neg_dataset.shuffle(1024).repeat()
    train_dataset = tf.data.experimental.sample_from_datasets([pos_dataset, neg_dataset], weights=[positive_weight, 1 - positive_weight])
    train_dataset = train_dataset.batch(batch_size).prefetch(1)

    # Create validation set
    val_data = np.vstack(val_data)
    val_labels = np.concatenate(val_labels)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(256)

    # Create test set
    test_data = np.vstack(test_data)
    test_labels = np.concatenate(test_labels)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(256)
    return train_dataset, val_dataset, test_dataset