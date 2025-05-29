import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

def preprocess_text(text):
    """
    Basic text preprocessing for Indonesian text
    
    Parameters:
    - text: String to preprocess
    
    Returns:
    - Preprocessed text
    """
    # Lowercase
    text = text.lower()
    
    # Remove special characters and extra spaces
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r'\s+', ' ', text)     
    text = text.strip()                 
    
    return text

def create_text_vectorizer(texts, max_tokens=10000, max_length=100):
    """
    Create a TextVectorization layer and adapt it to the given texts
    
    Parameters:
    - texts: List of text examples
    - max_tokens: Maximum number of tokens in the vocabulary
    - max_length: Maximum length of output sequences
    
    Returns:
    - vectorizer: Adapted TextVectorization layer
    """
    # Create and adapt TextVectorization layer
    vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_mode='int',
        output_sequence_length=max_length,
        standardize=preprocess_text
    )
    vectorizer.adapt(texts)
    
    return vectorizer