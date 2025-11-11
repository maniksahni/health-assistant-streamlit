import pickle

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the dataset
data = pd.read_csv('parkinsons.csv')

# Split the data into features and labels
# 'status' column is the target (0 = healthy, 1 = Parkinson's)
# Drop non-numeric identifier column 'name' if present
drop_cols = ['status']
if 'name' in data.columns:
    drop_cols.append('name')
X = data.drop(drop_cols, axis=1)
y = data['status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM model
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Parkinson’s Model Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
with open('parkinsons_model.sav', 'wb') as f:
    pickle.dump(model, f)