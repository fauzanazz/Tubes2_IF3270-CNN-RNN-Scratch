import tensorflow as tf
import numpy as np

class EmbeddingWrapper:
    """Wrapper for Keras Embedding layer to provide forward() interface"""
    def __init__(self, keras_embedding):
        self.layer = keras_embedding
    
    def forward(self, inputs, training=False):
        # Ensure inputs are int32
        if isinstance(inputs, np.ndarray):
            inputs = tf.convert_to_tensor(inputs, dtype=tf.int32)
        
        # Get embeddings
        outputs = self.layer(inputs)
        
        # Convert to numpy and maintain consistency with other layers
        return outputs.numpy()

class DropoutWrapper:
    """Wrapper for Keras Dropout layer to provide forward() interface"""
    def __init__(self, keras_dropout):
        self.layer = keras_dropout
        self.rate = keras_dropout.rate
    
    def forward(self, inputs, training=False):
        # During inference (training=False), dropout does nothing
        if not training:
            return inputs
        
        if isinstance(inputs, np.ndarray):
            inputs = tf.convert_to_tensor(inputs)
        
        # Apply dropout
        outputs = self.layer(inputs, training=training)
        
        # Convert back to numpy
        return outputs.numpy()

def predict(network, inputs, batch_size=32):
    """
    Make predictions with the scratch RNN model.
    """
    # Add input validation
    if not isinstance(inputs, (list, np.ndarray)):
        raise ValueError("Input texts must be list or numpy array")
    
    # Get the vectorizer
    import sys
    module = sys.modules['__main__']
    if hasattr(module, 'vectorizer'):
        vectorizer = module.vectorizer
        print("Using vectorizer from global scope")
    else:
        raise ValueError("No vectorizer found in global scope")
    
    # Convert inputs to numpy array
    if hasattr(inputs, 'values'):
        text_inputs = inputs.values
    else:
        text_inputs = np.array(inputs)
    
    # Process in batches
    outputs = []
    n_batches = (len(text_inputs) + batch_size - 1) // batch_size
    
    for i in range(0, len(text_inputs), batch_size):
        print(f"Processing batch {i//batch_size + 1}/{n_batches}")
        batch_texts = text_inputs[i:i + batch_size]
        
        # Vectorize text
        current_output = vectorizer(batch_texts)
        
        # Pass through each layer
        for j, layer in enumerate(network):
            layer_name = layer.__class__.__name__
            print(f"  Layer {j+1}: {layer_name}")
            
            if isinstance(layer, EmbeddingWrapper):
                current_output = layer.forward(current_output)
            else:
                current_output = layer.forward(current_output)
        
  
        outputs.append(current_output)
    
    # Combine all batches
    final_output = np.vstack(outputs)
    print("Prediction complete")
    return final_output