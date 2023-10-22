import numpy as np

def power_iteration(matrix, init_vector, num):
    vector = init_vector.copy()
    for a in range(num):
        vector = matrix.dot(vector)
        vector = vector / np.linalg.norm(vector)
    evalue = vector.T.dot(matrix.dot(vector))
    return evalue[0, 0]

matrix = np.array([[2, 1], [4, 5]])
init_vector = np.array([[1],[1]])
max_eigenvalue = power_iteration(matrix, init_vector, num=100)
print("最大特征值:", max_eigenvalue)