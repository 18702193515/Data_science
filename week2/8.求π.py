import math
import random
import time

# 雅可比-马基诺公式
def jacobi_madhava_formula(n):
    result = 0
    for k in range(n):
        result += ((-1) ** k) / (2 * k + 1)
    return result * 4

# 蒙特卡罗方法
def monte_carlo_method(num_points):
    points_inside_circle = 0
    total_points = num_points

    for _ in range(num_points):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        distance = math.sqrt(x ** 2 + y ** 2)

        if distance <= 1:
            points_inside_circle += 1

    return (points_inside_circle / total_points) * 4

# 雅克布尼-托勒密公式
def jacobi_totally_ptolemaic_formula():
    pi_approx = 0
    pi_approx += 12 * math.atan(1/49)
    pi_approx += 32 * math.atan(1/57)
    pi_approx -= 5 * math.atan(1/239)
    pi_approx *= 4
    return pi_approx

# 比较不同方法的效率
num_iterations = 1000000
num_points = 1000000

# 雅可比-马基诺公式
start_time = time.time()
pi_jacobi = jacobi_madhava_formula(num_iterations)
end_time = time.time()
execution_time_jacobi = end_time - start_time

# 蒙特卡罗方法
start_time = time.time()
pi_monte_carlo = monte_carlo_method(num_points)
end_time = time.time()
execution_time_monte_carlo = end_time - start_time

# 雅克布尼-托勒密公式
start_time = time.time()
pi_approx = jacobi_totally_ptolemaic_formula()
end_time = time.time()
execution_time_approx = end_time - start_time

# 输出结果及运行时间
print(f"π 的近似值（雅可比-马基诺公式）: {pi_jacobi:.10f}")
print(f"运行时间（雅可比-马基诺公式）: {execution_time_jacobi:.6f} 秒")

print(f"π 的近似值（蒙特卡罗方法）: {pi_monte_carlo:.10f}")
print(f"运行时间（蒙特卡罗方法）: {execution_time_monte_carlo:.6f} 秒")

print(f"π 的近似值（雅克布尼-托勒密公式）: {pi_approx:.10f}")
print(f"运行时间（雅克布尼-托勒密公式）: {execution_time_approx:.6f} 秒")