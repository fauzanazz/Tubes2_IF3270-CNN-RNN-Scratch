from Function import ActivationFunction
import numpy as np
import torch


if __name__ == "__main__":
    print("Comparing NumPy and PyTorch Activation Functions")
    print("=" * 50)

    # Test data
    x_np = np.linspace(-5, 5, 10).astype(np.float32)
    x_torch = torch.tensor(x_np, requires_grad=True)

    # 2D test data for softmax
    x_2d_np = np.array([
        [-2.0, 1.0, 3.0],
        [0.5, 1.5, -1.0],
        [3.0, -2.0, 0.5]
    ], dtype=np.float32)
    x_2d_torch = torch.tensor(x_2d_np, requires_grad=True)

    # 1. ReLU Comparison
    print("\nReLU Function:")

    # NumPy output
    relu_np = ActivationFunction.relu(x_np)
    relu_deriv_np = ActivationFunction.derivative_relu(x_np)

    # PyTorch output
    relu_torch = torch.relu(x_torch)
    # PyTorch derivative using autograd
    relu_torch.sum().backward()
    relu_deriv_torch = x_torch.grad.clone().numpy()
    x_torch.grad.zero_()

    print(f"Input: {x_np}")
    print(f"NumPy ReLU:          {relu_np}")
    print(f"PyTorch ReLU:        {relu_torch.detach().numpy()}")
    print(f"NumPy Derivative:    {relu_deriv_np}")
    print(f"PyTorch Derivative:  {relu_deriv_torch}")
    print(f"Max Difference:      {np.max(np.abs(relu_np - relu_torch.detach().numpy()))}")
    print(f"Max Deriv Diff:      {np.max(np.abs(relu_deriv_np - relu_deriv_torch))}")

    # 2. Sigmoid Comparison
    print("\nSigmoid Function:")
    # Reset tensor with requires_grad
    x_torch = torch.tensor(x_np, requires_grad=True)

    # NumPy output
    sigmoid_np = ActivationFunction.sigmoid(x_np)
    sigmoid_deriv_np = ActivationFunction.derivative_sigmoid(x_np)

    # PyTorch output
    sigmoid_torch = torch.sigmoid(x_torch)
    # PyTorch derivative using autograd
    sigmoid_torch.sum().backward()
    sigmoid_deriv_torch = x_torch.grad.clone().numpy()
    x_torch.grad.zero_()

    print(f"Input: {x_np}")
    print(f"NumPy Sigmoid:       {sigmoid_np}")
    print(f"PyTorch Sigmoid:     {sigmoid_torch.detach().numpy()}")
    print(f"NumPy Derivative:    {sigmoid_deriv_np}")
    print(f"PyTorch Derivative:  {sigmoid_deriv_torch}")
    print(f"Max Difference:      {np.max(np.abs(sigmoid_np - sigmoid_torch.detach().numpy()))}")
    print(f"Max Deriv Diff:      {np.max(np.abs(sigmoid_deriv_np - sigmoid_deriv_torch))}")

    # 3. Tanh Comparison
    print("\nTanh Function:")
    # NumPy output
    tanh_np = ActivationFunction.tanh(x_np)
    tanh_deriv_np = ActivationFunction.derivative_tanh(x_np)

    # Reset tensor with requires_grad
    x_torch = torch.tensor(x_np, requires_grad=True)

    # PyTorch output
    tanh_torch = torch.tanh(x_torch)
    # PyTorch derivative using autograd
    tanh_torch.sum().backward(retain_graph=True)
    tanh_deriv_torch = x_torch.grad.numpy().copy()
    x_torch.grad.zero_()

    print(f"Input: {x_np}")
    print(f"NumPy Tanh:          {tanh_np}")
    print(f"PyTorch Tanh:        {tanh_torch.detach().numpy()}")
    print(f"NumPy Derivative:    {tanh_deriv_np}")
    print(f"PyTorch Derivative:  {tanh_deriv_torch}")
    print(f"Max Difference:      {np.max(np.abs(tanh_np - tanh_torch.detach().numpy()))}")
    print(f"Max Deriv Diff:      {np.max(np.abs(tanh_deriv_np - tanh_deriv_torch))}")

    # 4. Softmax Comparison (2D case)
    print("\nSoftmax Function (2D):")
    # NumPy output
    softmax_np = ActivationFunction.softmax(x_2d_np)

    # PyTorch output
    softmax_torch = torch.nn.functional.softmax(x_2d_torch, dim=1)

    print(f"Input:\n{x_2d_np}")
    print(f"NumPy Softmax:\n{softmax_np}")
    print(f"PyTorch Softmax:\n{softmax_torch.detach().numpy()}")
    print(f"Max Difference:      {np.max(np.abs(softmax_np - softmax_torch.detach().numpy()))}")

    # 5. Softmax derivative
    print("\nSoftmax Derivative:")
    for i in range(len(x_2d_np)):
        print(f"\nSample {i + 1}:")
        # Get the numpy Jacobian
        single_input = x_2d_np[i].reshape(1, -1)
        jacobian_np = ActivationFunction.derivative_softmax(single_input)[0]

        # Get PyTorch Jacobian through manual computation
        x_single = torch.tensor(x_2d_np[i], requires_grad=True)
        softmax_single = torch.nn.functional.softmax(x_single, dim=0)

        # Manually compute Jacobian by taking derivative for each output dimension
        jacobian_torch = np.zeros((len(x_single), len(x_single)))
        for j in range(len(x_single)):
            if x_single.grad is not None:
                x_single.grad.zero_()

            # Compute gradient for jth output
            softmax_single[j].backward(retain_graph=True)
            jacobian_torch[:, j] = x_single.grad.numpy()

        print(f"NumPy Jacobian:\n{jacobian_np}")
        print(f"PyTorch Jacobian:\n{jacobian_torch}")
        print(f"Max Difference:      {np.max(np.abs(jacobian_np - jacobian_torch))}")