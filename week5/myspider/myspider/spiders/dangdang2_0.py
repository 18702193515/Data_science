import scrapy
from scrapy.http import Request
from myspider.items import MyspiderItem


class Dangdang20Spider(scrapy.Spider):
    name = "dangdang2.0"
    allowed_domains = ["dangdang.com"]
    start_urls = ["http://category.dangdang.com/pg1-cp01.54.00.00.00.00.html"]

    def parse(self, response):
        item = MyspiderItem()
        item["title"] = response.xpath("//a[@name='与-title']/@title").extract()
        item["link"] = response.xpath("//a[@name='itemlist-title']/@href").extract()
        item["price"] = response.xpath("//span[@class='search_now_price']/text()").extract()
        item["author"] = response.xpath("//a[@name='itemlist-author']/text()").extract()
        yield item