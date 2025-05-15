import numpy as np
import tensorflow as tf

# Buat fungsi SCCE
def sparse_categorical_crossentropy(y_true, y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

def sparse_categorical_crossentropy_prime(y_true, y_pred):
    epsilon = 1e-15
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)  # tetap tensor, dtype sama
    y_true_one_hot = tf.one_hot(y_true, depth=y_pred.shape[1])
    return - y_true_one_hot / y_pred

