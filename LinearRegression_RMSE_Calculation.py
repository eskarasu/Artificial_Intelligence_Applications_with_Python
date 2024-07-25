from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

# Create a simple dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]) # RMSE: 0.0
# y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 21]) # RMSE: 0.3535533905932738

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the RMSE
rmse = sqrt(mean_squared_error(y_test, y_pred))

print(f'RMSE: {rmse}')
