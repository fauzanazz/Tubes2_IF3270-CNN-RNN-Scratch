import numpy as np
import torch

from Function import LossFunction

if __name__ == "__main__":
    # Binary test data
    y_true_binary = np.array([1, 0, 0, 1, 1], dtype=np.float32)
    y_pred_binary = np.array([0.9, 0.1, 0.2, 0.8, 0.9], dtype=np.float32)

    # PyTorch tensors data
    y_true_binary_torch = torch.tensor(y_true_binary, dtype=torch.float32)
    y_pred_binary_torch = torch.tensor(y_pred_binary, dtype=torch.float32)
    y_pred_binary_torch.requires_grad = True

    # Multi-class test data
    y_true_multi = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=np.float32)
    y_pred_multi = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.9, 0.05, 0.05],
        [0.3, 0.6, 0.1]
    ], dtype=np.float32)

    # PyTorch multi-class tensors
    y_true_multi_torch = torch.tensor(y_true_multi, dtype=torch.float32)
    y_pred_multi_torch = torch.tensor(y_pred_multi, dtype=torch.float32)
    y_pred_multi_torch.requires_grad = True

    # Test MSE derivatives
    print("MSE Derivatives:")
    mse_deriv_np = LossFunction.mean_squared_error_derivative(y_pred_binary, y_true_binary)

    # PyTorch native implementation using autograd
    mse_loss = torch.nn.functional.mse_loss(y_pred_binary_torch, y_true_binary_torch)
    mse_loss.backward()
    mse_deriv_torch_native = y_pred_binary_torch.grad.clone()
    y_pred_binary_torch.grad.zero_()

    print(f"  NumPy result:              {mse_deriv_np}")
    print(f"  PyTorch autograd result:   {mse_deriv_torch_native.numpy()}")
    print(f"  Max difference:            {np.max(np.abs(mse_deriv_np - mse_deriv_torch_native.numpy()))}")

    # Test BCE derivatives
    print("\nBCE Derivatives:")
    bce_deriv_np = LossFunction.binary_cross_entropy_derivative(y_pred_binary, y_true_binary)

    # PyTorch native implementation using autograd
    bce_loss = torch.nn.functional.binary_cross_entropy(y_pred_binary_torch, y_true_binary_torch)
    bce_loss.backward()
    bce_deriv_torch_native = y_pred_binary_torch.grad.clone()
    y_pred_binary_torch.grad.zero_()

    print(f"  NumPy result:              {bce_deriv_np}")
    print(f"  PyTorch autograd result:   {bce_deriv_torch_native.numpy()}")
    print(f"  Max difference:            {np.max(np.abs(bce_deriv_np - bce_deriv_torch_native.numpy()))}")

    # Test CCE derivatives (multi-class)
    print("\nCCE Derivatives (Multi-class):")
    cce_deriv_np = LossFunction.categorical_cross_entropy_derivative(y_pred_multi, y_true_multi)

    # Create fresh tensor with requires_grad
    y_pred_multi_torch = torch.tensor(y_pred_multi, dtype=torch.float32, requires_grad=True)
    y_true_multi_torch = torch.tensor(y_true_multi, dtype=torch.float32)

    custom_loss = torch.sum(0.5 * torch.sum((y_pred_multi_torch - y_true_multi_torch) ** 2, dim=1))
    custom_loss.backward()
    cce_deriv_torch_auto = y_pred_multi_torch.grad.clone().numpy()

    print(f"  NumPy result:              {cce_deriv_np}")
    print(f"  PyTorch autograd result:    {cce_deriv_torch_auto}")
    print(f"  Max difference (autograd):  {np.max(np.abs(cce_deriv_np - cce_deriv_torch_auto))}")
