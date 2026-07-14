import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'Hours': [0, 1, 2, 3, 4, 5, 6],
    'Battery_Percentage': [100, 92, 84, 76, 68, 60, 52]
}

df = pd.DataFrame(data)

X = df[['Hours']]

y = df['Battery_Percentage']

model = LinearRegression()

model.fit(X, y)

prediction = model.predict([[3]])

print("Predicted Battery Percentage after 3 hours:", prediction[0])
