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
        # Input normalization
        inputs = (inputs - np.mean(inputs)) / (np.std(inputs) + 1e-8)
        
        batch_size = inputs.shape[0]
        time_steps = inputs.shape[1]
        
        # Initialize states with zeros and small noise (Keras-like initialization)
        noise_scale = 0.001
        h_t = np.zeros((batch_size, self.units)) + np.random.normal(0, noise_scale, (batch_size, self.units))
        all_states = []


        # More conservative gradient clipping
        MAX_GRAD = 0.25
        EPSILON = 1e-7
        
        for t in range(time_steps):
            x_t = inputs[:, t, :]
            
            # Per-timestep input normalization
            x_t = (x_t - np.mean(x_t, axis=1, keepdims=True)) / (np.std(x_t, axis=1, keepdims=True) + EPSILON)
            
            h_prev = h_t  # Store previous state
            state_input = (np.dot(x_t, self.kernel) + 
                        np.dot(h_prev, self.recurrent_kernel) + 
                        self.bias)
            state_input = np.clip(state_input, -MAX_GRAD, MAX_GRAD)
            h_t = np.tanh(state_input)
            
            # Add state normalization
            h_t = (h_t - np.mean(h_t, axis=1, keepdims=True)) / (np.std(h_t, axis=1, keepdims=True) + EPSILON)
            
            all_states.append(h_t)
        
        # Stack all states
        self.states = np.stack(all_states, axis=1)
        
        # Return either all states or just the last state
        if self.return_sequences:
            return self.states
        return all_states[-1]  # Return only the last state