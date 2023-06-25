from .BaseNLP import NLP

import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer
import random
from alive_progress import alive_bar

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

class BERTClassifier(NLP):
    def start(self):
        # load in data
        # raw_data defined here
        self.load_data()
        
        x_data, y_data = self.define_categories()
        
        # train_data and test_data defined here
        x_train, x_test = self.split_and_preprocess_data(x_data)
        y_train, y_test = self.split_and_preprocess_data(y_data)