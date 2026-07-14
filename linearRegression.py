import pandas as pd
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'Hours':[1,2,3,4,5,6],
    'Marks':[35,45,55,65,75,85]
}

df = pd.DataFrame(data)

X = df[['Hours']]
y = df['Marks']

# Create model
model = LinearRegression()

# Train model
model.fit(X,y)

# Predict marks for 7 hours
prediction = model.predict([[7]])

print("Predicted Marks:",prediction[0])
