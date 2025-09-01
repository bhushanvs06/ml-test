import pickle, json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
#hiiiiiiii aniket
# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Evaluate model
acc = accuracy_score(y_test, model.predict(X_test))

# Save metadata
metadata = {"accuracy": acc, "dataset": "Iris"}
with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("✅ Model trained and saved")
print("Accuracy:", acc)
