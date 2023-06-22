import matplotlib.pyplot as pyplot

# Code Referenced from https://github.com/alexchandel/political-compass/blob/master/compass.py
def prob_dicts_to_xy(x_prob_dict, y_prob_dict):
    final_x = 0
    final_y = 0
    max_attr_x = max(x_prob_dict, key=x_prob_dict.get)
    max_attr_y = max(y_prob_dict, key=y_prob_dict.get)
    
    print(max_attr_x)
    print(max_attr_y)
    
    if max_attr_x == "Left":
        pass

def plot_compass(x_prob_dict, y_prob_dict):
    pyplot.scatter(x, y)
    pyplot.xlim(-10, 10)
    pyplot.ylim(-10, 10)
    pyplot.title("Political coordinates")
    pyplot.xlabel("Economic Left/Right")
    pyplot.ylabel("Social Libertarian/Authoritarian")
    pyplot.grid()
    pyplot.show()