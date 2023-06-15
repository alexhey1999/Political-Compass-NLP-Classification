import csv                               # csv reader
from sklearn.svm import LinearSVC
from nltk.classify import SklearnClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support # to report on precision and recall
import numpy as np
import re

class NLPModel():
    def __init__(self,database):
        self.database = database
        
    def get_data(self):
        self.data = self.database.get_all_data()
        return self.data
    
    def format_data(self):
        # Step 1 - Load data in
        self.get_data()
        self.dataset = []
        for index, row in enumerate(self.data):
            self.dataset.append((row[3],row[2]))
            
        self.libertarian = [a for a in self.dataset if a[0] == 'Libertarian']
        self.authoritarian = [a for a in self.dataset if a[0] == 'Authoritarian']
        self.left = [a for a in self.dataset if a[0] == 'Left']
        self.right = [a for a in self.dataset if a[0] == 'Right']
        
        
    def pre_process(self,text):
        # Should return a list of tokens
        # First I lower case the entire string to increase consistency
        # Some words are cased differently depending on political standing e.g. (Black and black)
        # This standardises all instances However may prove a problem regarding names and proper nouns
        text = text.lower()
        
        # Following 2 commands are referenced from Lab 1 work with modifications to 
        # include more frequently used characters e.g. [ and ]
        
        # Add a space before all singular punctuation
        text = re.sub(r"(\w)([.,;:!?'\"”\)\[\]])", r"\1 \2", text)
        # Add a space after all singular punctuation
        text = re.sub(r"([.,;:!?'\"“\(\)\[\]])(\w)", r"\1 \2", text)

        # Split the string into words by spaces
        words = text.split(" ")
        return words
        
        
    def validate_compulsory_fields(self):
        print(f"Libertarian: {len(self.libertarian)}")
        print(f"Authoritarian: {len(self.authoritarian)}")
        print(f"Left: {len(self.left)}")
        print(f"Right: {len(self.right)}")

    def start(self):
        self.format_data()
        # self.pre_process()
        