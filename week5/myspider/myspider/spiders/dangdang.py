# -*- coding: UTF-8 -*-
import scrapy


class DangdangSpider(scrapy.Spider):
    name = "dangdang"
    allowed_domains = ["dangdang.com"]
    start_urls = ["https://search.dangdang.com/?key=%BC%C6%CB%E3%BB%FA&act=input&page_index=1"]

    def parse(self, response):
        filename = "csbook .html"
        open(filename, 'wb').write(response.body)
