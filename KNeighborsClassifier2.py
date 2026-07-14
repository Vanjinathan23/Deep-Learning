import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Dataset
data = {
    'Price': [10000, 12000, 15000, 18000, 50000, 55000, 60000, 70000],
    'Category': ['Budget', 'Budget', 'Budget', 'Budget',
                 'Premium', 'Premium', 'Premium', 'Premium']
}

df = pd.DataFrame(data)

X = df[['Price']]
y = df['Category']

# Create model
model = KNeighborsClassifier(n_neighbors=3)

# Train model
model.fit(X, y)

# Predict category for ₹20,000
prediction = model.predict([[20000]])

print("Predicted Category:", prediction[0])
