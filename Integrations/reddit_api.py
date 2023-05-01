from .base import BaseAPI
import requests
import http.client
import json

class RedditAPI(BaseAPI):
    def __init__(self, url, service_name, auth_method, credentials, database_file_name):
        super().__init__(url, service_name, auth_method, credentials, database_file_name)
        
    def top_posts_subreddit(self, subreddit_name):
        print(self.url)
        conn = http.client.HTTPSConnection(self.url)
        payload = ''
        conn.request("GET", f"/r/{subreddit_name}/top.json?limit=30&t=all", payload)
        res = conn.getresponse()
        data = res.read()
        data = data.decode("utf-8")
        data = json.loads(data)
        
        print(data["data"].keys())
        # print(data["data"]["before"])

        
        records = data["data"]["children"]
        # print(len(records))
        
        records = [i["data"] for i in records]
        titles = [i["title"] for i in records]
        
        # print(titles)
        # print(len(titles))
        
        return titles

        # print(url)
        # response = requests.get(url)
        # print(response.__attrs__)
        # print(response.reason)


    
    
    