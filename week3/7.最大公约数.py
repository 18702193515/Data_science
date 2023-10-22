def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a


num1 =int( input("请输入第一个数："))
num2 =int( input("请输入第二个数："))
result = gcd(num1,num2)
##result = gcd(min(num1,num2), max(num1,num2))
print("最大公约数是:", result)