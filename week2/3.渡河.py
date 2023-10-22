th = [' ', '狼', '羊', '菜']

def check(status):
    return status[0] == status[2] or (status[2] ^ status[1] and status[2] ^ status[3])

def cal(status):
    return status[0] * 1000 + status[1] * 100 + status[2] * 10 + status[3]

def cross(status1, existed, way, printed):
    status=status1.copy()
    if status == [0, 0, 0, 0]:
        if way not in printed:
            print(way)
            printed.append(way)
        return

    status[0] = 1 - status[0]
    if check(status):
        if cal(status) not in existed:
            newex = existed.copy()
            newex.add(cal(status))
            cross(status, newex, way + "农夫过河 ", printed)
            

    for i in range(1, 4):
        if status[0] + status[i] == 1:
            newst = status.copy()
            newst[i] = 1 - newst[i]
            if check(newst):
                if cal(newst) not in existed:


                    newex = existed.copy()
                    newex.add(cal(newst))
                    cross(newst, newex, way + th[i] + "和农夫过河 ", printed)
                    

status = [1, 1, 1, 1]
existed = set()
way = ""
printed = []
cross(status, existed, way, printed)