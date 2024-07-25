import numpy as np
import matplotlib.pyplot as plt

# Data preparation: true values and predicted values
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

# Calculate RMS Error
mse = np.mean((y_true - y_pred) ** 2)  # Mean Squared Error
rms_error = np.sqrt(mse)  # Root Mean Square Error
print(f"RMS Error: {rms_error}")

# Visualization
plt.figure(figsize=(10, 5))

# Plot true and predicted values
plt.plot(y_true, 'o-', label='True Values')
plt.plot(y_pred, 's--', label='Predicted Values')

# Plot error terms
for i in range(len(y_true)):
    plt.plot([i, i], [y_true[i], y_pred[i]], 'r-')

# Graph settings
plt.title('True vs Predicted Values with RMS Error')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
