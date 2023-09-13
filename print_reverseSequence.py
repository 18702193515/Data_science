# 获取用户输入的序列
sequence_str = input("请输入一个序列，以逗号分隔: ")

# 将输入的序列转换为列表
sequence = sequence_str.split(',')

# 使用for循环倒序输出
reversed_sequence_for = []
for i in range(len(sequence)-1, -1, -1):
    reversed_sequence_for.append(sequence[i])

print("倒排序输出（使用for循环）:", end=" ")
for num in reversed_sequence_for:
    print(num, end=" ")

# 使用while循环倒序输出
reversed_sequence_while = []
i = len(sequence) - 1
while i >= 0:
    reversed_sequence_while.append(sequence[i])
    i -= 1

print("\n倒排序输出（使用while循环）:", end=" ")
index = 0
while index < len(reversed_sequence_while):
    print(reversed_sequence_while[index], end=" ")
    index += 1