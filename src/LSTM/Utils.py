import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, classification_report, accuracy_score
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
import json

class Utils:
    @staticmethod
    def output_minus_target(output, target):
        target_array = np.zeros_like(output)
        target_array[target] = 1
        return output - target_array

def load_nusax_dataset(data_path: str, batch_size: int = 32) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    def load_split(split_name):
        file_path = os.path.join(data_path, f"{split_name}.csv")
        df = pd.read_csv(file_path)
        
        texts = df['text'].values
        labels = df['label'].map({'negative': 0, 'neutral': 1, 'positive': 2}).values
        
        dataset = tf.data.Dataset.from_tensor_slices((texts, labels))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    train_ds = load_split('train')
    val_ds = load_split('valid')
    test_ds = load_split('test')
    
    return train_ds, val_ds, test_ds


def create_text_vectorizer(train_dataset: tf.data.Dataset, 
                          max_tokens: int = 20000, 
                          sequence_length: int = 128) -> tf.keras.layers.TextVectorization:
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode='int',
        output_sequence_length=sequence_length
    )
    
    train_texts = train_dataset.map(lambda x, y: x)
    vectorizer.adapt(train_texts)
    
    return vectorizer


def build_lstm_model(vectorizer: tf.keras.layers.TextVectorization = None,
                    vocab_size: int = None,
                    embedding_dim: int = 128,
                    lstm_layers: List[int] = None,
                    bidirectional: bool = False,
                    dropout_rate: float = 0.5,
                    num_classes: int = 3,
                    mask_zero: bool = False,
                    learning_rate: float = 0.001) -> tf.keras.Model:
    
    if vectorizer is not None:
        vocab_size = vectorizer.vocabulary_size()
    
    inputs = tf.keras.Input(shape=(None,), dtype='int64')
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=mask_zero)(inputs)
    
    for i, units in enumerate(lstm_layers):
        return_sequences = (i < len(lstm_layers) - 1)
        lstm_layer = tf.keras.layers.LSTM(units, return_sequences=return_sequences)
        if bidirectional:
            lstm_layer = tf.keras.layers.Bidirectional(lstm_layer)
        x = lstm_layer(x)
        if return_sequences:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model


def evaluate_model(model: tf.keras.Model, 
                  test_dataset: tf.data.Dataset,
                  vectorizer: tf.keras.layers.TextVectorization = None) -> Dict[str, float]:
    test_texts = []
    test_labels = []
    
    for batch_texts, batch_labels in test_dataset:
        test_texts.extend(batch_texts.numpy())
        test_labels.extend(batch_labels.numpy())
    
    if vectorizer is not None:
        test_texts_vectorized = vectorizer(test_texts)
        predictions = model.predict(test_texts_vectorized)
    else:
        predictions = model.predict(test_dataset)
    
    predicted_labels = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(test_labels, predicted_labels)
    f1_macro = f1_score(test_labels, predicted_labels, average='macro')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'predictions': predicted_labels,
        'true_labels': test_labels
    }


def plot_training_history(history: tf.keras.callbacks.History, 
                         experiment_name: str,
                         save_path: str = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title(f'{experiment_name} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title(f'{experiment_name} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_comparison_results(results: Dict[str, Dict[str, float]], 
                          metric: str = 'f1_macro',
                          title: str = "Model Comparison",
                          save_path: str = None):
    """
    Plot comparison of different experiments.
    
    Args:
        results: Dictionary of experiment results
        metric: Metric to plot
        title: Plot title
        save_path: Path to save the plot
    """
    experiments = list(results.keys())
    values = [results[exp][metric] for exp in experiments]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(experiments, values)
    plt.title(title)
    plt.ylabel(metric.replace('_', ' ').title())
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_experiment_results(results: Dict[str, Any], 
                           experiment_name: str,
                           save_dir: str = "artifacts"):
    os.makedirs(save_dir, exist_ok=True)
    
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = value
    
    with open(os.path.join(save_dir, f"{experiment_name}_results.json"), 'w') as f:
        json.dump(serializable_results, f, indent=2)


def print_classification_report(y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              experiment_name: str):
    print(f"\n{experiment_name} Classification Report")
    print(classification_report(y_true, y_pred, 
                              target_names=['Negative', 'Neutral', 'Positive']))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted'):.4f}")