def construct_array(A):
    n = len(A)
    pre = [1] * n
    back = [1] * n

    for i in range(1, n):
        pre[i] = pre[i - 1] * A[i - 1]

    for i in range(n - 2, -1, -1):
        back[i] = back[i + 1] * A[i + 1]

    B = [pre[i] * back[i] for i in range(n)]

    return B

A = [1, 2, 3, 4, 5]
B = construct_array(A)
print("原数组：",A)
print(B)