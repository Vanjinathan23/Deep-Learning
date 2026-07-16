from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


X = [
    [8, 90, 1],   # Pass
    [3, 65, 0],   # Fail
    [6, 80, 1],   # Pass
    [2, 60, 0],   # Fail
    [7, 85, 1]    # Pass
]


y = [1, 0, 1, 0, 1]


model = DecisionTreeClassifier(random_state=42)


model.fit(X, y)


test_student = [[6, 82, 1]]


prediction = model.predict(test_student)

if prediction[0] == 1:
    print("Prediction: Pass")
else:
    print("Prediction: Fail")


y_pred = model.predict(X)

print("\nAccuracy:", accuracy_score(y, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))
