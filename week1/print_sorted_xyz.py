# 接受用户输入
x = float(input("请输入第一个数 x："))
y = float(input("请输入第二个数 y："))
z = float(input("请输入第三个数 z："))

# 将输入的数排序
numbers = [x, y, z]
numbers.sort()

# 打印排序后的数
print("从小到大排序后的数为：")
for number in numbers:
    print(number)