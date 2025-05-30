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
        batch_size = inputs.shape[0]
        time_steps = inputs.shape[1]
        
        h_t = np.zeros((batch_size, self.units))
        all_states = []

        MAX_GRAD = 0.25
        EPSILON = 1e-7
        
        for t in range(time_steps):
            x_t = inputs[:, t, :]
            h_prev = h_t
            
            state_input = (np.dot(x_t, self.kernel) + 
                        np.dot(h_prev, self.recurrent_kernel) + 
                        self.bias)
            state_input = np.clip(state_input, -MAX_GRAD, MAX_GRAD)
            h_t = np.tanh(state_input)
            all_states.append(h_t)
        
        # Stack all states
        self.states = np.stack(all_states, axis=1)
        
        # Return either all states or just the last state
        if self.return_sequences:
            return self.states
        return all_states[-1]  # Return only the last state