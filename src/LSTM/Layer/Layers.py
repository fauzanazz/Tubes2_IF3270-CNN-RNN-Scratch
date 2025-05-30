import numpy as np
import tensorflow as tf


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, mask_zero: bool = False):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.mask_zero = mask_zero
        self.weights = None
    
    def set_weights(self, weights: np.ndarray):
        self.weights = weights
    
    def forward(self, input_ids: np.ndarray) -> tuple:
        if self.weights is None:
            raise ValueError("Weight belum di set.")
        
        embedded = self.weights[input_ids]
        mask = None
        if self.mask_zero:
            mask = (input_ids != 0).astype(np.float32)
            mask_expanded = np.expand_dims(mask, axis=-1)
            embedded = embedded * mask_expanded
        
        return embedded, mask


class Dense:
    def __init__(self, units: int, activation: str = 'linear', use_bias: bool = True):
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.weights = None
        self.bias = None
    
    def set_weights(self, weights: np.ndarray, bias: np.ndarray = None):
        self.weights = weights
        if self.use_bias and bias is not None:
            self.bias = bias
        elif not self.use_bias:
            self.bias = np.zeros(self.units)
    
    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'softmax':
            x_shifted = x - np.max(x, axis=-1, keepdims=True)
            exp_x = np.exp(np.clip(x_shifted, -500, 500))
            return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-10)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'linear' or self.activation is None:
            return x
        else:
            if hasattr(tf.nn, self.activation):
                return getattr(tf.nn, self.activation)(x).numpy()
            return x
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Weight belum di set.")
        
        output = np.dot(x, self.weights)
        if self.use_bias and self.bias is not None:
            output += self.bias
        
        return self._apply_activation(output)


class Dropout:
    def __init__(self, rate: float = 0.5, seed: int = None):
        self.rate = rate
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        if training and self.rate > 0:
            keep_prob = 1.0 - self.rate
            mask = np.random.binomial(1, keep_prob, x.shape) / keep_prob
            return x * mask
        return x

class LSTMCell:
    def __init__(self, units, activation="tanh", recurrent_activation="sigmoid",
                 use_bias=True,
                 unit_forget_bias=True,
                 dropout=0.0, recurrent_dropout=0.0):
        
        self.units = units
        self.activation = self._get_activation(activation)
        self.recurrent_activation = self._get_activation(recurrent_activation)
        self.use_bias = use_bias
        self.unit_forget_bias = unit_forget_bias
        
        # Initialize weights
        self.kernel = None  # Input weights [input_dim, 4 * units]
        self.recurrent_kernel = None  # Recurrent weights [units, 4 * units]
        self.bias = None  # Bias [4 * units]
        
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
    
    def _get_activation(self, activation):
        if activation == "tanh":
            return np.tanh
        elif activation == "sigmoid":
            return self._sigmoid
        elif activation == "relu":
            return lambda x: np.maximum(0, x)
        elif activation is None or activation == "linear":
            return lambda x: x
        else:
            return activation
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class LSTMLayer:    
    def __init__(self, lstm_units, return_sequences=False, bidirectional=False,
                 unit_forget_bias=True, activation="tanh", recurrent_activation="sigmoid",
                 use_bias=True, dropout=0.0, recurrent_dropout=0.0,
                 return_state=False, go_backwards=False, stateful=False,
                 unroll=False, **kwargs):
        
        self.lstm_units = lstm_units
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.bidirectional = bidirectional
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        
        # Create LSTM cell(s)
        self.cell = LSTMCell(
            lstm_units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            unit_forget_bias=unit_forget_bias,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            **kwargs
        )
        
        if bidirectional:
            self.cell_backward = LSTMCell(
                lstm_units,
                activation=activation,
                recurrent_activation=recurrent_activation,
                use_bias=use_bias,
                unit_forget_bias=unit_forget_bias,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                **kwargs
            )
        
        # Weights
        self.weights_W = None  # Input kernel
        self.weights_U = None  # Recurrent kernel
        self.biases_b = None   # Bias
        
        self.weights_W_bwd = None  # Backward input kernel
        self.weights_U_bwd = None  # Backward recurrent kernel
        self.biases_b_bwd = None   # Backward bias
        
        # Keras default clipping values
        self.recurrent_clip = 0.0  # No clipping by default
        self.cell_clip = 0.0      # No clipping by default
        
        # Keras default activation functions
        self.recurrent_activation = self.cell.recurrent_activation
        self.activation = self.cell.activation
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
    
    def set_weights(self, weights):
        if self.bidirectional:
            if len(weights) != 6:
                raise ValueError("Bidirectional LSTM requires 6 weight arrays")
            self.weights_W = weights[0].astype(np.float32)  # Forward kernel
            self.weights_U = weights[1].astype(np.float32)  # Forward recurrent kernel
            self.biases_b = weights[2].astype(np.float32)   # Forward bias
            self.weights_W_bwd = weights[3].astype(np.float32)  # Backward kernel
            self.weights_U_bwd = weights[4].astype(np.float32)  # Backward recurrent kernel
            self.biases_b_bwd = weights[5].astype(np.float32)   # Backward bias
        else:
            if len(weights) != 3:
                raise ValueError("Unidirectional LSTM requires 3 weight arrays")
            self.weights_W = weights[0].astype(np.float32)  # Kernel
            self.weights_U = weights[1].astype(np.float32)  # Recurrent kernel
            self.biases_b = weights[2].astype(np.float32)   # Bias
            
            if self.cell.unit_forget_bias:
                b_i, b_f, b_c, b_o = np.split(self.biases_b, 4)
                b_f += 1.0
                self.biases_b = np.concatenate([b_i, b_f, b_c, b_o])
        
        # Set weights in cells
        self.cell.kernel = self.weights_W
        self.cell.recurrent_kernel = self.weights_U
        self.cell.bias = self.biases_b
        
        if self.bidirectional:
            self.cell_backward.kernel = self.weights_W_bwd
            self.cell_backward.recurrent_kernel = self.weights_U_bwd
            self.cell_backward.bias = self.biases_b_bwd
    
    def _clip_value(self, x, clip_value):
        if clip_value > 0:
            return np.clip(x, -clip_value, clip_value)
        return x
    
    def call(self, inputs, initial_state=None, mask=None, training=False):
        return self.forward(inputs, initial_state, mask, training)
    
    def forward(self, input_sequences, initial_state=None, mask=None, training=False):
        if self.bidirectional:
            return self._forward_bidirectional_lstm(input_sequences, initial_state, mask, training)
        else:
            return self._forward_unidirectional_lstm(
                input_sequences, self.weights_W, self.weights_U, self.biases_b,
                initial_state, mask, training
            )
    
    def _forward_unidirectional_lstm(self, input_sequences, weights_W, weights_U, biases_b,
                                   initial_state=None, mask=None, training=False):
        input_sequences = input_sequences.astype(np.float32)
        batch_size, timesteps, _ = input_sequences.shape
        
        if initial_state is not None:
            h_prev = initial_state[0].astype(np.float32)
            C_prev = initial_state[1].astype(np.float32)
        else:
            h_prev = np.zeros((batch_size, self.lstm_units), dtype=np.float32)
            C_prev = np.zeros((batch_size, self.lstm_units), dtype=np.float32)
        
        if self.return_sequences:
            outputs = np.zeros((batch_size, timesteps, self.lstm_units), dtype=np.float32)
        
        time_steps = range(timesteps-1, -1, -1) if self.go_backwards else range(timesteps)
        
        for t in time_steps:
            x_t = input_sequences[:, t, :]
            
            if mask is not None:
                mask_t = mask[:, t:t+1].astype(np.float32)
                x_t = x_t * mask_t
            
            combined_input = np.dot(x_t, weights_W) + np.dot(h_prev, weights_U) + biases_b
            
            z_i = combined_input[:, :self.lstm_units]
            z_f = combined_input[:, self.lstm_units:2*self.lstm_units]
            z_c = combined_input[:, 2*self.lstm_units:3*self.lstm_units]
            z_o = combined_input[:, 3*self.lstm_units:]
            
            # Calculate gate values
            i_t = self.recurrent_activation(z_i)  # Input gate
            f_t = self.recurrent_activation(z_f)  # Forget gate
            o_t = self.recurrent_activation(z_o)  # Output gate
            
            C_tilde_t = self.activation(z_c)
            
            C_t = f_t * C_prev + i_t * C_tilde_t
            C_t = self._clip_value(C_t, self.cell_clip)
            
            h_t = o_t * self.activation(C_t)
            h_t = self._clip_value(h_t, self.recurrent_clip)
            
            if mask is not None:
                mask_t = mask[:, t:t+1].astype(np.float32)
                h_t = h_t * mask_t + h_prev * (1 - mask_t)
                C_t = C_t * mask_t + C_prev * (1 - mask_t)
            
            if self.return_sequences:
                if self.go_backwards:
                    outputs[:, timesteps-1-t, :] = h_t
                else:
                    outputs[:, t, :] = h_t
            
            h_prev = h_t
            C_prev = C_t
        
        # Return based on return_state flag
        if self.return_state:
            if self.return_sequences:
                return outputs, h_t, C_t
            else:
                return h_t, h_t, C_t
        else:
            if self.return_sequences:
                return outputs
            else:
                return h_t
    
    def _forward_bidirectional_lstm(self, input_sequences, initial_state=None, mask=None, training=False):
        # Forward pass
        if initial_state is not None and len(initial_state) >= 2:
            initial_state_fwd = initial_state[:2]
        else:
            initial_state_fwd = None
            
        forward_result = self._forward_unidirectional_lstm(
            input_sequences, self.weights_W, self.weights_U, self.biases_b,
            initial_state_fwd, mask, training
        )
        
        # Backward pass
        input_sequences_reversed = np.flip(input_sequences, axis=1)
        mask_reversed = np.flip(mask, axis=1) if mask is not None else None
        
        if initial_state is not None and len(initial_state) >= 4:
            initial_state_bwd = initial_state[2:4]
        else:
            initial_state_bwd = None
            
        backward_result = self._forward_unidirectional_lstm(
            input_sequences_reversed, self.weights_W_bwd, self.weights_U_bwd, self.biases_b_bwd,
            initial_state_bwd, mask_reversed, training
        )
        
        # Handle return_state
        if self.return_state:
            if self.return_sequences:
                output_fwd, h_fwd, c_fwd = forward_result
                output_bwd_rev, h_bwd, c_bwd = backward_result
                output_bwd = np.flip(output_bwd_rev, axis=1)
                output = np.concatenate([output_fwd, output_bwd], axis=-1)
                return output, h_fwd, c_fwd, h_bwd, c_bwd
            else:
                h_fwd, _, c_fwd = forward_result
                h_bwd, _, c_bwd = backward_result
                output = np.concatenate([h_fwd, h_bwd], axis=-1)
                return output, h_fwd, c_fwd, h_bwd, c_bwd
        else:
            if self.return_sequences:
                output_fwd = forward_result
                output_bwd_rev = backward_result
                output_bwd = np.flip(output_bwd_rev, axis=1)
                return np.concatenate([output_fwd, output_bwd], axis=-1)
            else:
                h_fwd = forward_result
                h_bwd = backward_result
                return np.concatenate([h_fwd, h_bwd], axis=-1)

    @property
    def units(self):
        return self.lstm_units


class UnidirectionalLSTM(LSTMLayer):
    def __init__(self, lstm_units, return_sequences=False, **kwargs):
        super().__init__(lstm_units, return_sequences, bidirectional=False, **kwargs)


class BidirectionalLSTM(LSTMLayer):
    def __init__(self, lstm_units, return_sequences=False, **kwargs):
        super().__init__(lstm_units, return_sequences, bidirectional=True, **kwargs)


def forward_unidirectional_lstm(input_sequences, weights_W, weights_U, biases_b, 
                               lstm_units, return_sequences=False, **kwargs):
    lstm = UnidirectionalLSTM(lstm_units, return_sequences, **kwargs)
    lstm.set_weights([weights_W, weights_U, biases_b])
    return lstm.forward(input_sequences)


def forward_bidirectional_lstm(input_sequences, W_fwd, U_fwd, b_fwd, W_bwd, U_bwd, b_bwd,
                              lstm_units, return_sequences=False, **kwargs):
    lstm = BidirectionalLSTM(lstm_units, return_sequences, **kwargs)
    lstm.set_weights([W_fwd, U_fwd, b_fwd, W_bwd, U_bwd, b_bwd])
    return lstm.forward(input_sequences) 