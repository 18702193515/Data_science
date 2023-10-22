import numpy as np

def qr_iteration(matrix, num):
    n = matrix.shape[0]
    evalues = np.zeros(n)
    evectors = np.eye(n)
    for a in range(num):
        Q, R = np.linalg.qr(matrix)
        matrix = R.dot(Q)
        evectors = evectors.dot(Q)
        evalues = np.diag(matrix)
    return evalues, evectors

data = np.array([[1, 2, 1],[-1, 1, 3],[4, 3, -1]])
C = np.cov(data, rowvar=False)
evalue, evector = np.linalg.eig(C)
print("eig特征值：")
for a in evalue:
    print(a)
print("eig特征向量：")
for a in evector:
    print(a)

evalues, evectors = qr_iteration(C, num=10000)
print("qr特征值:")
for a in evalues:
    print(a)
print("qr特征向量:")
for a in evectors:
    print(a)