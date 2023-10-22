def calculate_sqrt2(precision,g):
    #g = 2000.0  # 初始猜测值
    while True:
        g_new = 0.5 * (g + 2 / g)  # 更新猜测值
        if abs(g_new - g) < precision:  # 判断精度是否满足要求
            break
        g = g_new  # 更新猜测值
    return g_new

precision = 0.0001  # 设置所需的精度
result = calculate_sqrt2(precision,2)
print("c初始值为2时，根号2的近似值为:", result)
result = calculate_sqrt2(precision,1/2)
print("c初始值为1/2时，根号2的近似值为:", result)