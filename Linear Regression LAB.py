import numpy as np

# Set random seed for reproducibility
np.random.seed(10)

# Data Generation
# Generate 200 random x values between 0 and 10
x = np.random.rand(200) * 10

# True relationship: y = 3.5 * x + 7 + noise
true_slope = 3.5
true_bias = 7
noise = np.random.randn(200)  # standard normal noise
y = true_slope * x + true_bias + noise

# Initialize parameters
slope = 0.0
bias = 0.0

# Hyperparameters
learning_rate = 0.0005
epochs = 1000

# Arrays to store the history of MSE, slope, and bias
mse_history = []
slope_history = []
bias_history = []

# Gradient Descent
for i in range(epochs):
    # Predictions
    y_pred = slope * x + bias
    
    # Compute error
    error = y_pred - y
    
    # Compute Mean Squared Error (MSE)
    mse = np.mean(error ** 2)
    
    # Store values
    mse_history.append(mse)
    slope_history.append(slope)
    bias_history.append(bias)
    
    # Compute gradients
    slope_grad = (2 / len(x)) * np.dot(error, x)
    bias_grad = (2 / len(x)) * np.sum(error)
    
    # Update parameters
    slope -= learning_rate * slope_grad
    bias -= learning_rate * bias_grad

# Find optimal values (lowest MSE)
min_mse_index = np.argmin(mse_history)
optimal_slope = slope_history[min_mse_index]
optimal_bias = bias_history[min_mse_index]

# Output optimal values
print(f"Optimal Slope: {optimal_slope:.4f}")
print(f"Optimal Bias: {optimal_bias:.4f}")
