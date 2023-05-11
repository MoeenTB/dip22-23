import numpy as np

B = np.array([[-1, 1, -1, 1],
              [0, 0, 0, 1],
              [1, 1, 1, 1],
              [8, 4, 2, 1]])

B2 = np.array([[8, 4, 2, 1],
              [27, 9, 3, 1],
              [64, 16, 4, 1],
              [125, 25, 5, 1]])

F = np.array([[0, 0, 7, 5],
              [4, 6, 6, 7],
              [1, 1, 7, 3],
              [0, 0, 0, 0]])

A = np.linalg.inv(B)*F*(np.linalg.inv(B).T)
A2 = np.linalg.inv(B2)*F*(np.linalg.inv(B2).T)

print(A)
print(A2)