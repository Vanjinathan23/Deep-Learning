import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Dataset
data = {
    'Battery': [95, 90, 85, 80, 40, 35, 30, 25],
    'Status': ['Good', 'Good', 'Good', 'Good', 'Low', 'Low', 'Low', 'Low']
}

df = pd.DataFrame(data)

X = df[['Battery']]
y = df['Status']

# Create model
model = KNeighborsClassifier(n_neighbors=3)

# Train model
model.fit(X, y)

# Predict battery status for 50%
prediction = model.predict([[50]])

print("Predicted Battery Status:", prediction[0])
