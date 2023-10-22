score = float(input("请输入考试成绩："))
if score < 60:
    print("不合格")
elif score < 75:
    print("合格")
elif score < 90:
    print("良好")
else:
    print("优秀")
