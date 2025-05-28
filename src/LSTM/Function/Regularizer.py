import numpy as np
from enums import RegularizationType

class Regularizer:
    def __init__(self, regularization_type: RegularizationType = RegularizationType.NONE,
                 lambda_reg=0.01, lambda_reg_l1=0.01, lambda_reg_l2=0.01, epsilon=1e-8):
        self.regularization_type = regularization_type
        self.lambda_reg = lambda_reg
        self.lambda_reg_l1 = lambda_reg_l1
        self.lambda_reg_l2 = lambda_reg_l2
        self.epsilon = epsilon

    def compute_grad(self, weights):
        if self.regularization_type == RegularizationType.L1:
            return self.lambda_reg * np.sign(weights)

        elif self.regularization_type == RegularizationType.L2:
            return self.lambda_reg * weights

        elif self.regularization_type == RegularizationType.ELASTIC_NET:
            l1_grad = self.lambda_reg_l1 * np.sign(weights)
            l2_grad = self.lambda_reg_l2 * weights
            return l1_grad + l2_grad

        elif self.regularization_type == RegularizationType.NONE:
            return np.zeros_like(weights)

        else:
            raise ValueError("Unknown regularization type")
