from Layer.Layers import Embedding, Dense, Dropout, UnidirectionalLSTM, BidirectionalLSTM
import numpy as np

class LSTMModel:
    def __init__(self, vocab_size: int, embedding_dim: int, lstm_layers: list, 
                 bidirectional: bool = False, num_classes: int = 3,
                 mask_zero: bool = False, dropout_rate: float = 0.5,
                 recurrent_dropout: float = 0.0, return_sequences: bool = False,
                 return_state: bool = False, activation: str = 'tanh',
                 recurrent_activation: str = 'sigmoid'):
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        self.mask_zero = mask_zero
        self.dropout_rate = dropout_rate
        self.return_sequences = return_sequences
        self.return_state = return_state
        
        # Initialize layers
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero)
        self.lstm_cells = []
        self.dropout_layers = []
        
        # Create LSTM layers
        for i, units in enumerate(lstm_layers):
            layer_return_sequences = (i < len(lstm_layers) - 1) or return_sequences
            
            if bidirectional:
                lstm_cell = BidirectionalLSTM(
                    units, 
                    return_sequences=layer_return_sequences,
                    return_state=(i == len(lstm_layers) - 1) and return_state,
                    activation=activation,
                    recurrent_activation=recurrent_activation,
                    dropout=0.0,
                    recurrent_dropout=recurrent_dropout
                )
            else:
                lstm_cell = UnidirectionalLSTM(
                    units,
                    return_sequences=layer_return_sequences,
                    return_state=(i == len(lstm_layers) - 1) and return_state,
                    activation=activation,
                    recurrent_activation=recurrent_activation,
                    dropout=0.0,
                    recurrent_dropout=recurrent_dropout
                )
            
            self.lstm_cells.append(lstm_cell)
            
            if i < len(lstm_layers) - 1:
                self.dropout_layers.append(Dropout(dropout_rate))
        
        self.final_dropout = Dropout(dropout_rate)
        
        self.dense1 = Dense(64, 'relu')
        self.dense2 = Dense(num_classes, 'softmax')
    
    def set_weights_from_keras(self, keras_model):
        weights = keras_model.get_weights()
        weight_idx = 0
        
        # Set embedding weights
        self.embedding.set_weights(weights[weight_idx])
        weight_idx += 1
        
        # Set LSTM weights
        for i, lstm_cell in enumerate(self.lstm_cells):
            if self.bidirectional:
                lstm_weights = weights[weight_idx:weight_idx + 6]
                lstm_cell.set_weights(lstm_weights)
                weight_idx += 6
            else:
                lstm_weights = weights[weight_idx:weight_idx + 3]
                lstm_cell.set_weights(lstm_weights)
                weight_idx += 3
        
        self.dense1.set_weights(weights[weight_idx], weights[weight_idx + 1])
        weight_idx += 2
        self.dense2.set_weights(weights[weight_idx], weights[weight_idx + 1])
    
    def forward(self, input_ids: np.ndarray, training: bool = False, 
                initial_state: list = None) -> np.ndarray:
        # Embedding layer
        x, mask = self.embedding.forward(input_ids)
        
        # LSTM layers
        lstm_states = []
        current_state = initial_state
        
        for i, lstm_cell in enumerate(self.lstm_cells):
            layer_initial_state = None
            if current_state is not None:
                if self.bidirectional:
                    layer_initial_state = current_state[i*4:(i+1)*4] if len(current_state) > i*4 else None
                else:
                    layer_initial_state = current_state[i*2:(i+1)*2] if len(current_state) > i*2 else None
            
            # Forward pass through LSTM
            lstm_output = lstm_cell.forward(x, layer_initial_state, mask, training)
            
            # Handle return_state
            if lstm_cell.return_state:
                if self.bidirectional:
                    x, h_fwd, c_fwd, h_bwd, c_bwd = lstm_output
                    lstm_states.extend([h_fwd, c_fwd, h_bwd, c_bwd])
                else:
                    x, h, c = lstm_output
                    lstm_states.extend([h, c])
            else:
                x = lstm_output
            
            if i < len(self.lstm_cells) - 1:
                x = self.dropout_layers[i].forward(x, training)
        
        x = self.final_dropout.forward(x, training)
        
        x = self.dense1.forward(x)
        x = self.dense2.forward(x)
        
        if self.return_state and lstm_states:
            return x, lstm_states
        return x
    
    def predict(self, input_ids: np.ndarray, batch_size: int = 32) -> np.ndarray:
        if len(input_ids.shape) == 1:
            input_ids = np.expand_dims(input_ids, 0)
        
        num_samples = input_ids.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        predictions = []
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch = input_ids[start_idx:end_idx]
            
            batch_pred = self.forward(batch, training=False)
            if isinstance(batch_pred, tuple):
                batch_pred = batch_pred[0]
                
            predictions.append(batch_pred)
        
        return np.concatenate(predictions, axis=0)