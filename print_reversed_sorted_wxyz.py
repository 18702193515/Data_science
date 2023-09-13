# 接受用户输入
w = float(input("请输入第一个数 w："))
x = float(input("请输入第二个数 x："))
y = float(input("请输入第三个数 y："))
z = float(input("请输入第四个数 z："))

# 将输入的数排序
numbers = [w, x, y, z]
numbers.sort(reverse=True)

# 打印排序后的数
print("从大到小排序后的数为：")
for number in numbers:
    print(number)