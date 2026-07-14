import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Dataset
data = {
    'Marks': [35, 40, 45, 50, 75, 80, 85, 90],
    'Result': ['Fail', 'Fail', 'Fail', 'Pass', 'Pass', 'Pass', 'Pass', 'Pass']
}

df = pd.DataFrame(data)

X = df[['Marks']]
y = df['Result']

# Create model
model = KNeighborsClassifier(n_neighbors=3)

# Train model
model.fit(X, y)

# Predict for 48 marks
prediction = model.predict([[48]])

print("Predicted Result:", prediction[0])
