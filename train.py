import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
with open('data_aug.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["label"])

print("Keys in dataset:", data_dict.keys())

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

# Train RandomForest Model with all CPU cores
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(x_train, y_train)

# Predictions & Accuracy
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save trained model
with open("RandomForest_aug.p", 'wb') as f:
    pickle.dump({"model": model}, f)

print("Model saved successfully!")
