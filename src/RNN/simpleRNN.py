import numpy as np

class SimpleRNN:
    def __init__(self, kernel, recurrent_kernel, bias, return_sequences=False):
        self.kernel = kernel  # Input weights
        self.recurrent_kernel = recurrent_kernel  # Recurrent weights
        self.bias = bias  # Bias
        self.units = bias.shape[0]  # Number of units from bias shape
        self.return_sequences = return_sequences
        self.states = None  # Store states for each timestep
        
    def forward(self, inputs):
        """
        Forward pass for SimpleRNN - Keras compatible implementation
        Args:
            inputs: Input array of shape (batch_size, timesteps, input_features)
        Returns:
            Array of shape (batch_size, timesteps, units) if return_sequences=True
            else (batch_size, units)
        """
        batch_size = inputs.shape[0]
        time_steps = inputs.shape[1]
        
        h_t = np.zeros((batch_size, self.units), dtype=inputs.dtype)
        all_states = []
        
        for t in range(time_steps):
            x_t = inputs[:, t, :]
            linear_output = np.dot(x_t, self.kernel) + np.dot(h_t, self.recurrent_kernel) + self.bias
            h_t = np.tanh(linear_output)
            all_states.append(h_t.copy())
        self.states = np.stack(all_states, axis=1)
        
        if self.return_sequences:
            return self.states
        return all_states[-1] 