import random
from alive_progress import alive_bar

class NLPBase():
    def __init__(self,database, debug=False):
        self.database = database
        self.percentage = 0.8
        self.global_feature_dict = {}
        self.debug = debug
        self.random_seed = 10000
        self.x_split_cats = ["Left", "Right"]
        self.y_split_cats = ["Authoritarian", "Libertarian"]
        random.seed(self.random_seed)
        
   
    def load_data(self):
        data = self.database.get_all_data()
        self.raw_data = []
        self.classes = {}
        for row in data:
            if row[3] not in self.classes:
                self.classes[row[3]] = 1
            else:
                self.classes[row[3]] += 1
                
            self.raw_data.append((row[2],row[3]))
        random.shuffle(self.raw_data)
        balanced_data = self.balance_data(self.raw_data, self.classes)
        
        self.classes = {}
        for i in balanced_data:
            if i[1] not in self.classes:
                self.classes[i[1]] = 1
            else:
                self.classes[i[1]] += 1
        
        self.raw_data = balanced_data
        
        return self.raw_data     
   
    def balance_data(self, data, item_dict):
        running_record_counts = {}
        # Balance the dataset by taking the category with the minimum data and then reducing the data to the correct size.
        max_record_value = min(item_dict.values())
        reduced_data = []
        for i in data:
            if i[1] not in running_record_counts:
                running_record_counts[i[1]] = 1
            else:
                running_record_counts[i[1]] += 1
                
            if running_record_counts[i[1]] < max_record_value:
                reduced_data.append(i)
        
        libertarian = [a for a in reduced_data if a[1] == 'Libertarian']
        authoritarian = [a for a in reduced_data if a[1] == 'Authoritarian']
        left = [a for a in reduced_data if a[1] == 'Left']
        right = [a for a in reduced_data if a[1] == 'Right']
        
        final_reduced_data = []
        for li, au, le, ri in zip(libertarian, authoritarian, left, right):
            final_reduced_data.append(li)
            final_reduced_data.append(au)
            final_reduced_data.append(le)
            final_reduced_data.append(ri)
        return final_reduced_data
    
    
    def define_categories(self):
        x_data = [a for a in self.raw_data if a[1] in self.x_split_cats]
        y_data = [a for a in self.raw_data if a[1] in self.y_split_cats]
        return x_data, y_data
    

    # HELPER FUNCTIONS
    def class_split(self):
        self.libertarian = [a for a in self.raw_data if a[1] == 'Libertarian']
        self.authoritarian = [a for a in self.raw_data if a[1] == 'Authoritarian']
        self.left = [a for a in self.raw_data if a[1] == 'Left']
        self.right = [a for a in self.raw_data if a[1] == 'Right']
        
    def validate_compulsory_fields(self):
        print(f"Libertarian: {len(self.libertarian)}")
        print(f"Authoritarian: {len(self.authoritarian)}")
        print(f"Left: {len(self.left)}")
        print(f"Right: {len(self.right)}")