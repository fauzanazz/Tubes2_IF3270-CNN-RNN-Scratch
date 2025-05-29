import tensorflow as tf
import numpy as np
from simpleRNN import SimpleRNN
from bidirectional import Bidirectional
from dense import Dense
from dropout import Dropout
from tensorflow.python.keras.layers.embeddings import Embedding as KerasEmbedding

def newEmbeddingLayer(layer):
    """
    Create a new scratch Embedding layer from Keras Embedding layer
    """
    print("creating new embedding layer")
    config = layer.get_config()
    weights = layer.get_weights()
    new_layer = KerasEmbedding(
        input_dim=config['input_dim'],
        output_dim=config['output_dim'],
        embeddings_initializer=tf.constant_initializer(weights[0]),
        input_length=config.get('input_length', None),
        mask_zero=config.get('mask_zero', False),
        trainable=config.get('trainable', True)
    )
    return new_layer

def newSimpleRNNLayer(layer):
    """
    Create a new scratch SimpleRNN layer from Keras SimpleRNN layer
    """
    print("creating new simple RNN layer")
    weights = layer.get_weights()
    kernel, recurrent_kernel, bias = weights
    
    return SimpleRNN(kernel, recurrent_kernel, bias)

def newBidirectionalLayer(layer):
    """
    Create a new scratch Bidirectional layer from Keras Bidirectional layer
    """
    print("creating new bidirectional layer")
    forward_weights = layer.forward_layer.get_weights()
    backward_weights = layer.backward_layer.get_weights()
    
    f_kernel, f_recurrent, f_bias = forward_weights
    b_kernel, b_recurrent, b_bias = backward_weights
    
    forward_rnn = SimpleRNN(f_kernel, f_recurrent, f_bias)
    backward_rnn = SimpleRNN(b_kernel, b_recurrent, b_bias)
    
    return Bidirectional(forward_rnn, backward_rnn)

def newDropoutLayer(layer):
    """
    Create a new scratch Dropout layer from Keras Dropout layer
    """
    print("creating new dropout layer")
    rate = layer.rate
    return Dropout(rate)

def newDenseLayer(layer):
    """
    Create a new scratch Dense layer from Keras Dense layer
    """
    print("creating new dense layer")
    weights, bias = layer.get_weights()
    activation = layer.activation.__name__ if hasattr(layer.activation, "__name__") else "linear"
    
    return Dense(weights, bias, activation=activation)

def load_scratch_model(model):
    """
    Convert a Keras RNN model to a scratch implementation
    """
    from predictRNN import EmbeddingWrapper  # Import the wrapper class
    
    ScratchModel = []
    
    # Skip the first layer (TextVectorization) which will be handled separately
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.TextVectorization):
            continue
            
        layer_type = layer.__class__.__name__
        print(layer_type)
        
        if layer_type == "Embedding":
            new_layer = EmbeddingWrapper(layer)
        elif layer_type == "SimpleRNN":
            new_layer = newSimpleRNNLayer(layer)
        elif layer_type == "Bidirectional":
            new_layer = newBidirectionalLayer(layer)
        elif layer_type == "Dropout":
            new_layer = newDropoutLayer(layer)
        elif layer_type == "Dense":
            new_layer = newDenseLayer(layer)
        else:
            print(f"Warning: Layer type {layer_type} not supported")
            continue
            
        ScratchModel.append(new_layer)
    
    return ScratchModel