def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

# 获取用户输入的常数 n
n = int(input("请输入一个正整数: "))

# 调用函数计算阶乘
result = factorial(n)

# 输出结果
print(f"{n} 的阶乘是: {result}")