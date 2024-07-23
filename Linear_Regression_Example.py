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

"""

Linear Regression is a statistical analysis method used for predicting how one variable is likely to respond to changes in other variables. The variable you want to predict is called the dependent variable, and the variable you are using to predict the other variable's value is called the independent variable.

This form of analysis estimates the coefficients of the linear equation, involving one or more independent variables, that best predict the value of the dependent variable. Linear regression fits a straight line that minimizes the discrepancies between predicted and actual output values.

There are simple linear regression calculators that use a “least squares” method to discover the best-fit line for a set of paired data. You then estimate the value of X (dependent variable) from Y (independent variable).

Linear regression models are relatively simple and provide an easy-to-interpret mathematical formula that can generate predictions. Linear regression can be applied to various areas in business and academic study. You’ll find that linear regression is used in everything from biological, behavioral, environmental and social sciences to business.

Linear regression models have become a proven way to scientifically and reliably predict the future. Because linear regression is a long-established statistical procedure, the properties of linear-regression models are well understood and can be trained very quickly.

In summary, Linear Regression is a powerful and flexible model that can be used for a wide range of prediction tasks, making it a valuable tool in the field of machine learning.

"""
