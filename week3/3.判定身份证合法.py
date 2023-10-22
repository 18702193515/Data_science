import re

def validate_id_number(id_number):
    pattern = r'^[1-9]\d{5}(18|19|20)\d{2}((0[1-9])|(1[0-2]))(([0-2][1-9])|(3[0-1]))\d{3}[0-9Xx]$'
    if re.match(pattern, id_number):
        return '这是一个合法身份证号'
    else:
        return '这不是一个合法身份证号'
    
id=input("请输入一个身份证号码：")
print(validate_id_number(id))