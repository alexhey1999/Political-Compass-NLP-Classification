import matplotlib.pyplot as pyplot
from selenium import webdriver
from selenium.webdriver.common.by import By
from PIL import Image
from io import BytesIO
import time
import uuid

# Code Referenced from https://github.com/alexchandel/political-compass/blob/master/compass.py
def prob_dicts_to_xy(x_prob_dict, y_prob_dict):
    final_x = 0
    final_y = 0
    max_attr_x = max(x_prob_dict, key=x_prob_dict.get)
    max_attr_y = max(y_prob_dict, key=y_prob_dict.get)
    
    max_val_x = x_prob_dict[max_attr_x]
    max_val_y = y_prob_dict[max_attr_y]
    
    if max_attr_x == "Left":
        bias = (max_val_x - 0.5) * 2
        x_coord = -1 * (bias * 10)        
    else:
        bias = (max_val_x - 0.5) * 2
        x_coord = bias * 10
        
    if max_attr_y == "Libertarian":
        bias = (max_val_y - 0.5) * 2
        y_coord = -1 * (bias * 10)        
    else:
        bias = (max_val_y - 0.5) * 2
        y_coord = bias * 10
        
    return x_coord, y_coord
    
def get_official_compass(x_prob_dict, y_prob_dict):
    x,y = prob_dicts_to_xy(x_prob_dict, y_prob_dict)
    full_url = f"https://www.politicalcompass.org/crowdchart2?spots={round(x,2)}%7C{round(y)}%7C"
    driver = webdriver.Chrome()
    driver.minimize_window()
    driver.get(full_url)
    time.sleep(1)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
    # chart = driver.find_element(By.ID, "analytics")
    # driver.execute_script("arguments[0].scrollIntoView();", chart)
    file_name = f"Grapher\Graphs\{uuid.uuid4()}.png"
    driver.maximize_window()
    driver.save_screenshot(file_name)
    driver.close()
    screenshot = Image.open(file_name)
    screenshot = screenshot.crop((1000, 138, 1975, 1100))
    screenshot.show()
    screenshot.save(file_name)