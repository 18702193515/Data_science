# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface

import pymysql
from itemadapter import ItemAdapter


class MyspiderPipeline:
    def process_item(self, item, spider):
        conn = pymysql.connect(host='localhost', user='root', password='123456', db='mydb', port=3306, charset='utf8')
        for i in range(0, len(item["title"])):
            title = item["title"][i]
            price = item["price"][i]
            author = item["author"][i]
            link = item["link"][i]
            sql = "insert into csbook(title,price,author,link) values ('"+title+"','"+price+"','"+author+"','"+link+"');"
            try:
               conn.query(sql)
               conn.commit()
            except Exception as e:
                print(e)
        # 关闭连接
        conn.close()
        return item