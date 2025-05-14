import numpy as np
import tensorflow as tf

# Buat fungsi SCCE
def sparse_categorical_crossentropy(y_true, y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

def sparse_categorical_crossentropy_prime(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.eye(y_pred.shape[1])[y_true] / y_pred

