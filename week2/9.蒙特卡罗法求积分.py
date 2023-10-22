import random
import math

def monte_carlo_integration(f, a, b, num_samples):
    integral_sum = 0

    for _ in range(num_samples):
        x = random.uniform(a, b)
        integral_sum += f(x)

    integral = (b - a) * (integral_sum / num_samples)
    return integral

def f(x):
    return x**2 + 4 * x * math.sin(x)

# 定义积分区间和采样点数量
a = 2
b = 3
num_samples = 1000000

# 使用蒙特卡洛方法计算积分
integral = monte_carlo_integration(f, a, b, num_samples)

# 输出结果
print(f"定积分的近似值: {integral:.6f}")