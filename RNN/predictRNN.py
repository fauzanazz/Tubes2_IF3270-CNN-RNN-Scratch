import tensorflow as tf
import numpy as np

class EmbeddingWrapper:
    """Wrapper for Keras Embedding layer to provide forward() interface"""
    def __init__(self, keras_embedding):
        self.layer = keras_embedding
    
    def forward(self, inputs):
        return self.layer(inputs).numpy()

def predict(network, inputs, batch_size=32):
    """
    Make predictions with the scratch RNN model.
    """
    # Get the vectorizer from global scope
    import sys
    module = sys.modules['__main__']
    if hasattr(module, 'vectorizer'):
        vectorizer = module.vectorizer
        print("Using vectorizer from global scope")
    else:
        raise ValueError("No vectorizer found in global scope. Please ensure vectorizer is available.")
    
    # Convert inputs to numpy array if needed
    if hasattr(inputs, 'values'):
        text_inputs = inputs.values
    else:
        text_inputs = np.array(inputs)
    
    # Process inputs with vectorizer
    print("Preprocessing text inputs...")
    outputs = []
    
    # Process in batches
    for i in range(0, len(text_inputs), batch_size):
        batch_texts = text_inputs[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(text_inputs) + batch_size - 1)//batch_size}")
        
        # Vectorize the text
        current_output = vectorizer(batch_texts).numpy()
        
        # Pass through each layer
        for j, layer in enumerate(network):
            layer_name = layer.__class__.__name__
            print(f"Layer {j+1}: {layer_name}")
            
            if isinstance(layer, tf.keras.layers.Embedding):
                # Wrap Embedding layer if not already wrapped
                if not hasattr(layer, 'forward'):
                    layer = EmbeddingWrapper(layer)
                current_output = layer.forward(current_output)
            else:
                # For custom layers, use forward method directly
                current_output = layer.forward(current_output)
        
        outputs.append(current_output)
    
    # Concatenate all batch outputs
    final_output = np.vstack(outputs)
    print("Prediction complete")
    return final_output