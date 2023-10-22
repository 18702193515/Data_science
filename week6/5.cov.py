import numpy as np

data = np.array([[1, 2, 1],[-1, 1, 3],[4, 3, -1]])
C = np.cov(data, rowvar=False)
print(C)