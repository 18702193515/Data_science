def check_duplicate_substring(s):
    for i in range(len(s) - 2):
        if s[i] == s[i+1] == s[i+2]:
            return True
    return False

# 获取用户输入的字符串
s = input("请输入一个字符串: ")

# 调用函数进行判断
if check_duplicate_substring(s):
    print("输入的字符串中包含连续相同字符组成的子字符串")
else:
    print("输入的字符串中不包含连续相同字符组成的子字符串")