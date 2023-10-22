def newton_cubic_root(c, epsilon=1e-6, max_iterations=100):
    x = c / 2  # 初始近似解
    
    for i in range(max_iterations):
        f = x**3 - c
        f_prime = 3 * x**2
        
        if abs(f) < epsilon:
            break
        
        x = x - f / f_prime
    
    return x

# 示例使用
c = 10
result = newton_cubic_root(c)
print(c,"的三次方根近似值为:", result)