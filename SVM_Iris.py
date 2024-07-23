#Support Vector Machines - SVM

# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42) # Accuracy Rate: 1.0
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=64) # Accuracy Rate: 0.9666666666666667

# Create an SVM model
model = svm.SVC(kernel='linear') # We are using a linear kernel

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the accuracy rate
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy Rate: {accuracy}')

"""

Support Vector Machines (SVMs) are a set of supervised learning methods used for classification, regression, and outlier detection. They were developed in the 1990s by Vladimir N. Vapnik and his colleagues.

In the context of classification, one of the most common uses of SVMs, they work by finding an optimal hyperplane that separates data points of different classes. The hyperplane is chosen to maximize the margin between the closest data points of opposite classes, known as support vectors. This maximization of the margin allows the SVM to generalize well to new data and make accurate classification predictions.

SVMs can handle both linear and nonlinear classification tasks. When the data is not linearly separable, kernel functions are used to transform the data into a higher-dimensional space where it becomes linearly separable. This application of kernel functions is known as the "kernel trick".

SVMs are widely used in machine learning due to their flexibility in being applied to a wide variety of tasks, including structured prediction problems. They are used in various fields such as image recognition, text categorization, bioinformatics, and even in stock market analysis.

In summary, SVMs are powerful and flexible models that can be used for a wide range of classification and regression tasks, making them a valuable tool in the field of machine learning.

"""