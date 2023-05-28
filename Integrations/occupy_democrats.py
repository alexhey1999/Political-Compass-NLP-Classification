from .base import BaseAPI
import requests
import json
from bs4 import BeautifulSoup

class OccupyDemocratsAPI(BaseAPI):
    def __init__(self, url, service_name, auth_method, credentials, database_file_name):
        super().__init__(url, service_name, auth_method, credentials, database_file_name)
        
    def all_news_reports(self,page_limit=814,start_page_num = 0):
        print(self.url)
        final_titles = []
        for page_num in range(start_page_num, page_limit):
            page = requests.get(url= f"https://{self.url}/category/news/page/{page_num}/", headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(page.content, "html.parser")
            titles_on_page = soup.find_all('div', class_="post-title")       
            for i in titles_on_page:
                try:
                    final_titles.append(i.text.split(":",1)[1])
                    print(i.text.split(":",1)[1])
                except:
                    try:
                        final_titles.append(i.text.split("?",1)[1])
                        print(i.text.split("?",1)[1])
                    except:
                        try:
                            final_titles.append(i.text.split("!",1)[1])
                            print(i.text.split("!",1)[1])
                        except:
                            final_titles.append(i.text)
        return final_titles
        
        
        
        # data = res.read()
        # data = data.decode("utf-8")
        # data = json.loads(data)
        
        # print(data["data"].keys())
        # print(data["data"]["before"])

        
        # records = data["data"]["children"]
        # print(len(records))
        
        # records = [i["data"] for i in records]
        # titles = [i["title"] for i in records]
        
        # print(titles)
        # print(len(titles))
        
        # return titles

        # print(url)
        # response = requests.get(url)
        # print(response.__attrs__)
        # print(response.reason)


    
    
    