import numpy as np
X = np.array([
    [85, 78, 92, 88],
    [70, 65, 75, 72],
    [95, 90, 98, 96],
    [60, 55, 58, 62]
], dtype=float)

print("Original Student Marks:")

X_norm = X / 100

W_encoder = np.array([
    [0.5, 0.2],
    [0.4, 0.3],
    [0.6, 0.5],
    [0.3, 0.7]
])

encoded = np.dot(X_norm, W_encoder)

print("\nEncoded Features (2 Features):")
print(np.round(encoded, 3))

W_decoder = np.array([
    [0.6, 0.5, 0.7, 0.4],
    [0.3, 0.4, 0.2, 0.8]
])

decoded = np.dot(encoded, W_decoder)
decoded_marks = decoded * 100

print("\nReconstructed Student Marks:")
print(np.round(decoded_marks, 2))
