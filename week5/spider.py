import requests
from lxml import etree
import csv
import pymysql

# 建立数据库连接
db = pymysql.connect(host='localhost', user='root', password='123456', db='mydb', port=3306, charset='utf8')
cursor = db.cursor()

# 准备插入SQL语句
sql = "INSERT INTO douban_spider (name,rating) VALUES (%s,%s)"

urls = ['https://movie.douban.com/top250?start={}&filter='.format(str(i)) for i in range(0, 250, 25)]
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 18.8; Win4; x64) Appleebkit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

for url in urls:
    html = requests.get(url, headers=headers)
    selector = etree.HTML(html.text)
    infos = selector.xpath("//ol[@class='grid_view']/li") # visit all the URLs and get information
    # 解析网页内容
    for info in infos:
        name = info.xpath(".//div[@class='info']//div[@class='hd']//a/span[1]/text()")
        rating = info.xpath(".//div[@class='info']//div[@class='star']/span[2]/text()")
        if name:
            name = name[0]
            try:
                # 执行SQL语句
                cursor.execute(sql, (name,rating))
                #print(sql, (name,rating))
                db.commit() # 进行数据库提交，写入数据库
            except Exception as e:
                print('写入失败:', e)
                db.rollback() # 数据回滚，多次操作要么都执行，要么都不执行

# 关闭游标连接
cursor.close()
# 关闭数据库连接
db.close()

print('写入成功！')