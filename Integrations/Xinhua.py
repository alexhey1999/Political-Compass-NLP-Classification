from .base import BaseAPI
import requests
import json
from bs4 import BeautifulSoup
import re

class XinhuaAPI(BaseAPI):
    def __init__(self, url, service_name, auth_method, credentials, database_file_name):
        super().__init__(url, service_name, auth_method, credentials, database_file_name)
        
    def all_posts(self):
        pass