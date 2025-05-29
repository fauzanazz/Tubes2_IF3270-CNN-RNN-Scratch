from layer import Layer
import numpy as np

class SimpleRNN(Layer):
    def __init__(self, units, return_sequences=False, kernel=None, recurrent_kernel=None, bias=None):
        self.type = "rnn"
        self.units = units
        self.return_sequences = return_sequences
        self.kernel = kernel
        self.recurrent_kernel = recurrent_kernel
        self.bias = bias
        

    def forward(self, input):
        self.input = input
        batch_size, time_steps, features = input.shape
        
        # Initialize output array
        if self.return_sequences:
            output = np.zeros((batch_size, time_steps, self.units))
        else:
            output = np.zeros((batch_size, self.units))
            
        # Initialize hidden state
        h_prev = np.zeros((batch_size, self.units))
        
        for i in range(time_steps):
            # Current input at time step i
            x_t = input[:, i, :]
            
            # Calculate weighted input
            u_xt = np.dot(x_t, self.kernel) if self.kernel is not None else 0
            w_ht = np.dot(h_prev, self.recurrent_kernel) if self.recurrent_kernel is not None else 0
            
            # Output for hidden state
            ht = np.tanh(u_xt + w_ht + self.bias if self.bias is not None else 0)
            
            # Storing output
            if self.return_sequences:
                output[:, i, :] = ht
            
            # Update state for next time step
            h_prev = ht            
            
        
        # If not returning sequences, return only the last output
        if not self.return_sequences:
            output = h_prev
            
        self.output = output
        
        return output
        