# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html
from scrapy import Item, Field

class AdviceItem(Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    
    url = Field()
    title = Field()
    date = Field()
    authors = Field()
    #tags = Field()
    #teaser = Field()
    
    #article = Field()
    #question = Field()
    #rating = Field()
    #explanation = Field()
    text = Field()
    studies = Field()
    questions = Field()
    answers = Field()
    explanations = Field()
    
    sources = Field()

