import tensorflow as tf
from layer import Layer

class Softmax(Layer):
    def forward(self, input):
        exp_input = tf.exp(input - tf.reduce_max(input, axis=1, keepdims=True))  # stabilisasi numerik
        self.output = exp_input / tf.reduce_sum(exp_input, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Softmax grad: dL/dz = y_pred - y_true sudah dihitung di loss
        # Jadi grad sudah dihitung dari luar, tinggal diteruskan
        return output_gradient  # diasumsikan output_gradient = y_pred - y_true
    
class ReLU(Layer):
    def forward(self, input):
        self.input = input
        return tf.nn.relu(input)

    def backward(self, output_gradient, learning_rate):
        relu_grad = tf.cast(self.input > 0, tf.float32)
        return output_gradient * relu_grad