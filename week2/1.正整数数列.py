n=int(input("请输入一个正整数："))
if n%3==1:
    if n>=4:
        print("2,2"+(n-4)//3*",3")
    elif n==1:
        print("1")
elif n%3==2:
    print("2"+(n-2)//3*",3")
else:
    print("3"+(n-3)//3*",3")
    

