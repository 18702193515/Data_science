def remove_spaces(s):
    return s.replace(" ", "")

# 获取用户输入的字符串
s = input("请输入一个字符串: ")

# 调用函数去除空格
result = remove_spaces(s)

# 输出结果
print("去除空格后的字符串:", result)