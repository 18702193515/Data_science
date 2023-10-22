def cube_root(num):
    left = 0.0  # 左边界
    right = num  # 右边界
    epsilon = 0.0001  # 精度要求

    while True:
        mid = (left + right) / 2  # 计算中点

        if abs(mid**3 - num) < epsilon:  # 判断是否满足精度要求
            return mid

        if mid**3 < num:
            left = mid  # 更新左边界
        else:
            right = mid  # 更新右边界

# 获取用户输入的数字
num = float(input("请输入一个数字: "))

# 调用函数求立方根
result = cube_root(num)

# 输出结果
print("自建函数求立方根:", result)

result = pow(num, 1/3)

print("调用系统函数求立方根:", result)
