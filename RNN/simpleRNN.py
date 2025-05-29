import numpy as np

class SimpleRNN:
    def __init__(self, kernel, recurrent_kernel, bias, return_sequences=False):
        self.kernel = kernel  # Input weights
        self.recurrent_kernel = recurrent_kernel  # Recurrent weights
        self.bias = bias  # Bias term
        self.units = bias.shape[0]  # Number of units from bias shape
        self.return_sequences = return_sequences
        self.states = None  # Will store states for each timestep
        
    def forward(self, inputs):
        """
        Forward pass for SimpleRNN
        Args:
            inputs: Input array of shape (batch_size, timesteps, input_features)
        Returns:
            Array of shape (batch_size, timesteps, units) if return_sequences=True
            else (batch_size, units)
        """
        batch_size = inputs.shape[0]
        time_steps = inputs.shape[1]
        
        # Initialize states
        h_t = np.zeros((batch_size, self.units))  # Initial hidden state
        all_states = []

        # Process each time step
        for t in range(time_steps):
            # Get current input
            x_t = inputs[:, t, :]
            
            # Calculate current state
            # h_t = tanh(x_t @ W + h_(t-1) @ U + b)
            h_t = np.tanh(
                np.dot(x_t, self.kernel) +
                np.dot(h_t, self.recurrent_kernel) +
                self.bias
            )
            
            all_states.append(h_t)
        
        # Stack all states
        self.states = np.stack(all_states, axis=1)
        
        # Return either all states or just the last state
        if self.return_sequences:
            return self.states
        return all_states[-1]  # Return only the last state