dec = float(input("输入数字："))
bstr=str(bin(int(dec)))
if(dec-int(dec)!=0):
    bstr+='.'
    dec=dec-int(dec)
    while(dec>0):
        dec*=2
        if dec>=1:
            dec-=1
            bstr+='1'
        else:
            bstr+='0'
print("二进制表示为"+bstr)
