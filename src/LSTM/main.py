import sys
sys.dont_write_bytecode = True

# Import libraries
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score
import warnings
import random

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from Utils import (
    load_nusax_dataset, 
    create_text_vectorizer, 
    build_lstm_model, 
    evaluate_model,
    plot_loss_curves,
    save_experiment_results
)
from LSTMModel import LSTMModel
import tensorflow as tf

def main():
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    tf.keras.utils.set_random_seed(42)
    
    config = {
        'data_path': 'data/nusax_sentiment_id',
        'batch_size': 32,
        'max_tokens': 20000,
        'sequence_length': 128,
        'embedding_dim': 128,
        'lstm_layers': [64, 64],
        'bidirectional': True,
        'dropout_rate': 0.5,
        'num_classes': 3,
        'epochs': 10,
    }
    
    train_ds, val_ds, test_ds = load_nusax_dataset(
        data_path=config['data_path'],
        batch_size=config['batch_size']
    )
    print("Dataset loaded successfully")
    
    print("Creating text vectorizer")
    vectorizer = create_text_vectorizer(
        train_dataset=train_ds,
        max_tokens=config['max_tokens'],
        sequence_length=config['sequence_length']
    )
    vocab_size = vectorizer.vocabulary_size()
    print(f"Vectorizer created with vocabulary size: {vocab_size}")
    
    keras_model = build_lstm_model(
        vectorizer=vectorizer,
        embedding_dim=config['embedding_dim'],
        lstm_layers=config['lstm_layers'],
        bidirectional=config['bidirectional'],
        dropout_rate=config['dropout_rate'],
        num_classes=config['num_classes']
    )
    
    print(f"Keras model built with {keras_model.count_params():,} parameters")
    def vectorize_text(text, label):
        return vectorizer(text), label
    train_ds_vectorized = train_ds.map(vectorize_text)
    val_ds_vectorized = val_ds.map(vectorize_text)
    test_ds_vectorized = test_ds.map(vectorize_text)
    
    print("Training Keras LSTM")
    history = keras_model.fit(
        train_ds_vectorized,
        validation_data=val_ds_vectorized,
        epochs=config['epochs'],
        verbose=1
    )
    
    plot_loss_curves(
        history,
        save_path="artifacts/loss_curves.png"
    )
    
    print("Setting up custom LSTM model")
    custom_model = LSTMModel(
        vocab_size=vocab_size,
        embedding_dim=config['embedding_dim'],
        lstm_layers=config['lstm_layers'],
        bidirectional=config['bidirectional'],
        num_classes=config['num_classes'],
    )
    
    custom_model.set_weights_from_keras(keras_model)
    
    all_test_inputs = []
    all_test_labels = []
    
    for batch in test_ds_vectorized:
        batch_inputs, batch_labels = batch
        all_test_inputs.append(batch_inputs.numpy())
        all_test_labels.append(batch_labels.numpy())
    
    full_test_inputs = np.concatenate(all_test_inputs, axis=0)
    full_test_labels = np.concatenate(all_test_labels, axis=0)
    
    
    keras_full_pred = keras_model.predict(full_test_inputs, verbose=0)
    custom_full_pred = custom_model.forward(full_test_inputs)
    
    max_diff = np.max(np.abs(keras_full_pred - custom_full_pred))
    mean_diff = np.mean(np.abs(keras_full_pred - custom_full_pred))
    std_diff = np.std(np.abs(keras_full_pred - custom_full_pred))
    
    print(f"Dataset verification - Max difference: {max_diff:.8f}")
    print(f"Dataset verification - Mean difference: {mean_diff:.8f}")
    print(f"Dataset verification - Std difference: {std_diff:.8f}")
    
    keras_full_pred_classes = np.argmax(keras_full_pred, axis=1)
    custom_full_pred_classes = np.argmax(custom_full_pred, axis=1)
    full_prediction_agreement = np.mean(keras_full_pred_classes == custom_full_pred_classes)
    
    print(f"Dataset prediction agreement: {full_prediction_agreement:.4f}")
    
    for class_id in range(config['num_classes']):
        class_mask = full_test_labels == class_id
        if np.any(class_mask):
            class_diff = np.mean(np.abs(keras_full_pred[class_mask] - custom_full_pred[class_mask]))
            class_agreement = np.mean(keras_full_pred_classes[class_mask] == custom_full_pred_classes[class_mask])
            print(f"Class {class_id} - Mean diff: {class_diff:.6f}, Agreement: {class_agreement:.4f}")
    
    
    keras_results = evaluate_model(
        model=keras_model,
        test_dataset=test_ds_vectorized,
        vectorizer=None
    )
    
    custom_predicted_labels = custom_full_pred_classes
    custom_accuracy = accuracy_score(full_test_labels, custom_predicted_labels)
    custom_f1_macro = f1_score(full_test_labels, custom_predicted_labels, average='macro')
    
    custom_results = {
        'accuracy': custom_accuracy,
        'f1_macro': custom_f1_macro,
        'predictions': custom_predicted_labels,
        'true_labels': full_test_labels
    }
    
    print(f"Keras LSTM Results:")
    print(f"  Accuracy:    {keras_results['accuracy']:.4f}")
    print(f"  F1 Macro:    {keras_results['f1_macro']:.4f}")
    
    print(f"Custom LSTM Results:")
    print(f"  Accuracy:    {custom_results['accuracy']:.4f}")
    print(f"  F1 Macro:    {custom_results['f1_macro']:.4f}")
    
    acc_diff = abs(keras_results['accuracy'] - custom_results['accuracy'])
    f1_diff = abs(keras_results['f1_macro'] - custom_results['f1_macro'])
    
    print(f"Differences:")
    print(f"  Accuracy Difference:  {acc_diff:.4f}")
    print(f"  F1 Macro Difference:  {f1_diff:.4f}")
    
    if acc_diff < 0.01 and f1_diff < 0.01:
        print("Custom implementation matches Keras implementation!")
    else:
        print("Some differences detected between implementations")
        
        if acc_diff > 0.05 or f1_diff > 0.05:
            print("Detailed difference analysis:")
            keras_pred_dist = np.bincount(keras_results['predictions'], minlength=3) / len(keras_results['predictions'])
            custom_pred_dist = np.bincount(custom_predicted_labels, minlength=3) / len(custom_predicted_labels)
            
            print(f"Keras prediction distribution: {keras_pred_dist}")
            print(f"Custom prediction distribution: {custom_pred_dist}")
            
            prediction_agreement = np.mean(keras_results['predictions'] == custom_predicted_labels)
            print(f"Individual prediction agreement: {prediction_agreement:.4f}")
    
    print("Saving results")
    os.makedirs("artifacts", exist_ok=True)

    def convert_numpy_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_list(item) for item in obj]
        else:
            return obj
    
    experiment_results = {
        'config': config,
        'keras_results': convert_numpy_to_list(keras_results),
        'custom_results': convert_numpy_to_list(custom_results),
        'training_history': {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy']
        },
        'verification': {
            'dataset_max_diff': float(max_diff),
            'dataset_mean_diff': float(mean_diff),
            'dataset_std_diff': float(std_diff),
            'dataset_prediction_agreement': float(full_prediction_agreement),
            'accuracy_diff': float(acc_diff),
            'f1_diff': float(f1_diff)
        }
    }
    
    save_experiment_results(
        results=experiment_results,
        experiment_name="lstm_comparison",
        save_dir="artifacts"
    )
    
    try:
        keras_model.save("artifacts/keras_lstm_model.keras")
        print("Keras model saved successfully")
    except Exception as e:
        print(f"Warning: Could not save Keras model: {e}")
        try:
            keras_model.save("artifacts/keras_lstm_model.h5")
            print("Keras model saved in legacy format")
        except Exception as e2:
            print(f"Error: Could not save model in any format: {e2}")
    
    print("Results saved to artifacts directory")
    return experiment_results

if __name__ == "__main__":
    results = main()