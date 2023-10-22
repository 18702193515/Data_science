def Statistic(file):
    f = open(file,'r', encoding='utf-8')
    dictionary = {}
    for line in f.readlines():
        if len(line)>10:
            mark =[ ',',':','.','\'s',';','?','(',')'] 
            for m in mark:
                line = line.replace(m,' ')
            #print (line)
            lineattr = line.strip().split(" ")
            for char in lineattr:
                if char.strip() != '':
                    if char not in dictionary:
                        dictionary[char]=1
                    else:dictionary[char]+=1
    a = sorted(dictionary.items(),key = lambda x:x[1],reverse = True)
    return a
def printWords(file,n):
    a=Statistic(file)
    for i in range(min(n, len(a))):
        print (a[i],end='')
printWords('hamlet.txt',20)