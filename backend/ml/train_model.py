import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset

data = pd.read_csv("training_data.csv")

# Separate features and labels

X = data.drop("stress", axis=1)
y = data["stress"]

# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

# Create model

model = DecisionTreeClassifier()

# Train model

model.fit(X_train, y_train)

# Test model

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Save model

joblib.dump(model, "stress_model.pkl")

print("Model saved as stress_model.pkl")
