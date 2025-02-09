# Re-import necessary libraries after execution state reset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Reload dataset since execution state reset removed uploaded files
file_path = "/content/4th feb market research(Sheet4).csv"  # Update if file path differs

# Load dataset assuming it has a comma delimiter (change if different)
df = pd.read_csv(file_path, delimiter=',') # Changed delimiter to ','

# Encode categorical variables using Label Encoding
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for reference

# Define features (X) and target variable (y)
X = df.drop(columns=['y'])  # Features
y = df['y']  # Target

# Standardizing numerical variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate model performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Get feature importance using absolute coefficients
feature_importance = np.abs(model.coef_[0])
important_features = sorted(zip(X.columns, feature_importance), key=lambda x: x[1], reverse=True)[:3]

# Convert to DataFrame for display
important_features_df = pd.DataFrame(important_features, columns=["Feature", "Importance"])

# Display important variables
# Instead of using ace_tools, just display the dataframe using pandas
print("Top 3 Important Variables:")
print(important_features_df) # Print the dataframe

# Print model accuracy
print(f"Accuracy: {accuracy}")
