from .base import BaseAPI
import requests
import json
from bs4 import BeautifulSoup
import re

class GabAPI(BaseAPI):
    def __init__(self, url, service_name, auth_method, credentials, database_file_name):
        super().__init__(url, service_name, auth_method, credentials, database_file_name)
        
    def all_posts(self, account_id, start_page_num = 1, page_limit = 10000000):
        print(self.url)
        final_posts = []
        for page_num in range(start_page_num, page_limit):
            page = requests.get(url= f"https://{self.url}/api/v1/accounts/{account_id}/statuses?page={page_num}&sort_by=newest/", headers={'User-Agent': 'Mozilla/5.0'})
            message_json = json.loads(page.text)
            if message_json != []:
                for i in message_json:
                    tags_removed = re.sub('<[^<]+?>', '', i["content"])
                    links_removed = re.sub(r'http\S+', '', tags_removed)
                    final_posts.append(links_removed)
                    print(links_removed)
            else:
                print("####")
                print("Done")
                print("####")
                break
    
        return final_posts