import pandas as pd
from sklearn.neural_network import MLPClassifier

data = {
    "Study_Hours": [2, 3, 5, 6, 8],
    "Attendance": [60, 65, 80, 85, 95],
    "Assignment": [50, 55, 70, 80, 90],
    "Result": [0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["Study_Hours", "Attendance", "Assignment"]]

y = df["Result"]

model = MLPClassifier(hidden_layer_sizes=(4,),
                      max_iter=2000,
                      random_state=42)

model.fit(X, y)

prediction = model.predict([[4, 75, 65]])

if prediction[0] == 1:
    print("Student Will Pass")
else:
    print("Student Will Fail")
