import tensorflow as tf
from convolutional import Convolutional
from pool import MaxPooling2D,AvgPooling2D
from flatten import Flatten
from dense import Dense


def newConvLayer(layer):
    print("creating new conv layer")

    config = layer.get_config()
    weights,biases = layer.get_weights()
    return Convolutional(filters=config['filters'],kernel_size=config['kernel_size'],activation=tf.nn.relu,kernels=weights,biases=biases)

def newMaxPoolLayer(layer):
    print("creating new max pool layer")

    config = layer.get_config()
    return MaxPooling2D(pool_size=tuple(config['pool_size']))


def newAvgPoolLayer(layer):
    print("creating new avg pool layer")
    config = layer.get_config()
    return AvgPooling2D(pool_size=tuple(config['pool_size']))


def newFlattenLayer(layer):
    print("creating new flatten layer")
    return Flatten()

def newDenseLayer(layer):
    print("creating new dense layer")
    config = layer.get_config()
    weights, biases = layer.get_weights()
    return Dense(
        input_size=weights.shape[0],
        output_size=weights.shape[1],
        weights=weights,
        biases=biases,
        activation=tf.nn.relu if config['activation'] == 'relu' else tf.nn.softmax
    )


def load_scratch_model(model: tf.keras.models):
    ScratchModel = []
    for layer in model.layers:
        layerType = layer.name
        print(layerType)
        if "conv2d" in layerType:
            newLayer = newConvLayer(layer)
        elif "max_pooling2d" in layerType:
            newLayer = newMaxPoolLayer(layer)
        elif "avg_pooling2d" in layerType:
            newLayer = newAvgPoolLayer(layer)
        elif "flatten" in layerType:
            newLayer = newFlattenLayer(layer)
        elif "dense" in layerType:
            newLayer = newDenseLayer(layer)

        ScratchModel.append(newLayer)
    return ScratchModel

        