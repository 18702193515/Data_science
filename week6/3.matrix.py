import numpy as np
matrix = np.array([[2, 1], [4, 5]])
evalue, evector = np.linalg.eig(matrix)
print("特征值：")
for a in evalue:
    print(a)
print("特征向量：")
for a in evector:
    print(a)