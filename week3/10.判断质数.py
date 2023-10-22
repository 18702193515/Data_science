def is_prime(a):
    if a <= 1:
        return False

    for i in range(2, int(a ** 0.5) + 1):
        if a % i == 0:
            return False
    return True

num = int(input("请输入一个整数："))
if is_prime(num):
    print(f"{num} 是质数")
else:
    print(f"{num} 不是质数")