# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Example dataset creation (Replace this with your actual dataset)
# Assume X contains features and y contains the target variable
# X, y = your_data, your_target_variable

# For demonstration, we use a dummy dataset
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=10000, n_features=11, random_state=42, n_classes=2)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Generate classification report
report = classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"])
print("Classification Report:")
print(report)

# Feature Importance (coefficients for logistic regression)
feature_importance = pd.DataFrame({
    'Feature': [f'Feature {i}' for i in range(X.shape[1])],
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("Feature Importance:")
print(feature_importance)
