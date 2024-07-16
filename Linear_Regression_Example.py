# Import necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression

# Create training data
X = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
Y = np.array([5, 20, 14, 32, 22, 38])

# Create and train the model
model = LinearRegression()
model.fit(X, Y)

# Print the coefficients and intercept of the model
print('Coefficient:', model.coef_)
print('Intercept:', model.intercept_)

# Make a prediction
X_new = np.array([60]).reshape((-1, 1))
Y_new = model.predict(X_new)
print('Prediction:', Y_new)
