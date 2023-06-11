from .base import BaseAPI
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait


class CatoIntegration(BaseAPI):
    def __init__(self, url, service_name, auth_method, credentials, database_file_name):
        super().__init__(url, service_name, auth_method, credentials, database_file_name)
        
    def all_posts(self, start_page_num = 0, page_limit = 3602):
        full_url = "https://" + self.url + "/blog"
        driver = webdriver.Chrome()
        final_list = []
        for i in range(start_page_num, page_limit):
            driver.get(f"{full_url}?page={i}")
            try:
                posts = driver.find_elements(By.CLASS_NAME, "blog-page__header.mb-5")
                if posts == []: break
            except:
                break
            for j in posts:
                final_list.append(j.find_element(By.CLASS_NAME, "h2").text)
            time.sleep(2)
            print(f"Page {i}")
        
        return final_list
            