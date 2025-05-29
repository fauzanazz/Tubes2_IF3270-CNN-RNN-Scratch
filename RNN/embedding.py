import numpy as np
from layer import Layer

class Embedding(Layer):
    def __init__(self, weights):
        """
        Initialize an embedding layer with pre-trained weights.
        
        Parameters:
        - weights: Pre-trained embedding matrix with shape (vocab_size, embedding_dim)
                  where each row is the embedding vector for a token
        """
        super().__init__()
        self.weights = weights
        self.vocab_size, self.embedding_dim = weights.shape
    
    def forward(self, inputs):
        """
        Maps token indices to their corresponding embedding vectors.
        
        Parameters:
        - inputs: Token indices with shape (...) or (batch_size, sequence_length)
        
        Returns:
        - Embedded vectors with shape (..., embedding_dim) or (batch_size, sequence_length, embedding_dim)
        """
        self.input = inputs
        
        # Convert TensorFlow tensor to numpy if needed
        if hasattr(inputs, 'numpy'):
            inputs = inputs.numpy()
        
        # Handle different input dimensions
        if isinstance(inputs, (int, np.integer)):
            # Single token case
            self.output = self.weights[inputs]
        elif inputs.ndim == 1:
            # Batch of tokens: (batch_size,)
            self.output = self.weights[inputs.astype(int)]
        else:
            # Batch of sequences: (batch_size, sequence_length)
            self.output = self.weights[inputs.astype(int)]
            
        return self.output
    
    def __str__(self):
        """String representation of the layer"""
        return f"Embedding(vocab_size={self.vocab_size}, embedding_dim={self.embedding_dim})"