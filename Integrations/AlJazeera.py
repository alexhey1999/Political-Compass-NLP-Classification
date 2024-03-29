from .base import BaseAPI
import re
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
import re


class AlJazeeraAPI(BaseAPI):
    def __init__(self, url, service_name, auth_method, credentials, database_file_name):
        super().__init__(url, service_name, auth_method, credentials, database_file_name)
        
    def all_posts_opinion(self, limit = 10):
        full_url = "https://" + self.url + "/opinion/"
        driver = webdriver.Chrome()
        driver.get(full_url)
        for i in range(limit):
            posts = driver.find_elements(By.CLASS_NAME, "gc.u-clickable-card.gc--type-opinion.gc--list.gc--with-image")
            try:
                more_post_button = driver.find_element(By.CLASS_NAME, "show-more-button")
            except:
                break
            driver.execute_script("arguments[0].scrollIntoView();", more_post_button)
            driver.execute_script("arguments[0].click();", more_post_button)
            print("Done: ", i)
            el = WebDriverWait(driver, 5)
        
        final_list = []
        for i in posts:
            final_list.append(i.find_element(By.CLASS_NAME, "gc__title").text.replace("\xad", ""))
        
        for i in final_list:
            print(i)
        
        return final_list


    def all_posts_news(self, limit = 1000):
        full_url = "https://" + self.url + "/news/"
        driver = webdriver.Chrome()
        driver.get(full_url)
        for i in range(limit):
            posts = driver.find_elements(By.CLASS_NAME, "gc.u-clickable-card.gc--type-post.gc--list.gc--with-image")
            try:
                more_post_button = driver.find_element(By.CLASS_NAME, "show-more-button")
            except:
                break
            driver.execute_script("arguments[0].scrollIntoView();", more_post_button)
            driver.execute_script("arguments[0].click();", more_post_button)
            print("Done: ", i)
            time.sleep(2)
            el = WebDriverWait(driver, 5)
        
        final_list = []
        for i in posts:
            final_list.append(i.find_element(By.CLASS_NAME, "gc__title").text.replace("\xad", ""))
        
        for i in final_list:
            print(i)
        
        
        
        return final_list
        
        