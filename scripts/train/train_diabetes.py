import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the dataset
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(here, "..", ".."))
data = pd.read_csv(os.path.join(root, "data", "raw", "diabetes.csv"))

# Split the data into features and labels
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Diabetes Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
with open(os.path.join(root, "saved_models", "diabetes_model.pkl"), "wb") as f:
    pickle.dump(model, f)
