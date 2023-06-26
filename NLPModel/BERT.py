from .BaseNLP import NLPBase

from sklearn.model_selection import train_test_split
import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer
import random
import pandas as pd
from alive_progress import alive_bar

import matplotlib.pyplot as plt

tf.get_logger().setLevel("ERROR")

# CODE Referenced from https://www.tensorflow.org/text/tutorials/classify_text_with_bert

class BERTClassifier(NLPBase):
    def convert_to_dataframe(self, data, cat_list):
        final = [[i,j] for (i,j) in data]
        df = pd.DataFrame(final, columns=["Text", "Category"])
        df["Category"] = df["Category"].apply(lambda cat: cat_list.index(cat))
        final_dict = {}
        for i,cat in enumerate(cat_list):
            final_dict[i] = cat
        return df, final_dict
                
    def start(self):
        # load in data
        # raw_data defined here
        self.load_data()
        x_data, y_data = self.define_categories()
        x_dataframe, x_dataframe_dict = self.convert_to_dataframe(x_data, self.x_split_cats)
        y_dataframe, y_dataframe_dict = self.convert_to_dataframe(y_data, self.y_split_cats)
                
        print(x_dataframe["Category"].value_counts())
        print(x_dataframe.sample())
        print(x_dataframe_dict)
        
        x_train, x_test, y_train, y_test = train_test_split(x_dataframe['Text'], x_dataframe["Category"], stratify=x_dataframe["Category"])
        
        self.get_bert_model()
        classifier_model = self.build_classifier_model() 
        
        # classifier_model.summary()
        tf.keras.utils.plot_model(classifier_model)
        


    def build_classifier_model(self):
        self.tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
        self.tfhub_handle_encoder = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
        
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(self.tfhub_handle_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(self.tfhub_handle_encoder, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
        return tf.keras.Model([text_input], [net])
        
        
        
                

        
    def get_bert_model(self, bert_model_name = "bert_en_uncased_L-12_H-768_A-12"):
        map_name_to_handle = {
            "bert_en_uncased_L-12_H-768_A-12":
                "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3",
            "bert_en_cased_L-12_H-768_A-12":
                "https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3",
            "bert_multi_cased_L-12_H-768_A-12":
                "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3",
            "small_bert/bert_en_uncased_L-2_H-128_A-2":
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1",
            "small_bert/bert_en_uncased_L-2_H-256_A-4":
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1",
            "small_bert/bert_en_uncased_L-2_H-512_A-8":
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1",
            "small_bert/bert_en_uncased_L-2_H-768_A-12":
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1",
            "small_bert/bert_en_uncased_L-4_H-128_A-2":
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1",
            "small_bert/bert_en_uncased_L-4_H-256_A-4":
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1",
            "small_bert/bert_en_uncased_L-4_H-512_A-8":
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1",
            "small_bert/bert_en_uncased_L-4_H-768_A-12":
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1",
            "small_bert/bert_en_uncased_L-6_H-128_A-2":
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1",
            "small_bert/bert_en_uncased_L-6_H-256_A-4":
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1",
            "small_bert/bert_en_uncased_L-6_H-512_A-8":
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1",
            "small_bert/bert_en_uncased_L-6_H-768_A-12":
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1",
            "small_bert/bert_en_uncased_L-8_H-128_A-2":
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1",
            "small_bert/bert_en_uncased_L-8_H-256_A-4":
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1",
            "small_bert/bert_en_uncased_L-8_H-512_A-8":
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1",
            "small_bert/bert_en_uncased_L-8_H-768_A-12":
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1",
            "small_bert/bert_en_uncased_L-10_H-128_A-2":
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1",
            "small_bert/bert_en_uncased_L-10_H-256_A-4":
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1",
            "small_bert/bert_en_uncased_L-10_H-512_A-8":
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1",
            "small_bert/bert_en_uncased_L-10_H-768_A-12":
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1",
            "small_bert/bert_en_uncased_L-12_H-128_A-2":
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
            "small_bert/bert_en_uncased_L-12_H-256_A-4":
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1",
            "small_bert/bert_en_uncased_L-12_H-512_A-8":
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1",
            "small_bert/bert_en_uncased_L-12_H-768_A-12":
                "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1",
            "albert_en_base":
                "https://tfhub.dev/tensorflow/albert_en_base/2",
            "electra_small":
                "https://tfhub.dev/google/electra_small/2",
            "electra_base":
                "https://tfhub.dev/google/electra_base/2",
            "experts_pubmed":
                "https://tfhub.dev/google/experts/bert/pubmed/2",
            "experts_wiki_books":
                "https://tfhub.dev/google/experts/bert/wiki_books/2",
            "talking-heads_base":
                "https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1",
        }

        map_model_to_preprocess = {
            "bert_en_uncased_L-12_H-768_A-12":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "bert_en_cased_L-12_H-768_A-12":
                "https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3",
            "small_bert/bert_en_uncased_L-2_H-128_A-2":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "small_bert/bert_en_uncased_L-2_H-256_A-4":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "small_bert/bert_en_uncased_L-2_H-512_A-8":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "small_bert/bert_en_uncased_L-2_H-768_A-12":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "small_bert/bert_en_uncased_L-4_H-128_A-2":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "small_bert/bert_en_uncased_L-4_H-256_A-4":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "small_bert/bert_en_uncased_L-4_H-512_A-8":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "small_bert/bert_en_uncased_L-4_H-768_A-12":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "small_bert/bert_en_uncased_L-6_H-128_A-2":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "small_bert/bert_en_uncased_L-6_H-256_A-4":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "small_bert/bert_en_uncased_L-6_H-512_A-8":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "small_bert/bert_en_uncased_L-6_H-768_A-12":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "small_bert/bert_en_uncased_L-8_H-128_A-2":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "small_bert/bert_en_uncased_L-8_H-256_A-4":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "small_bert/bert_en_uncased_L-8_H-512_A-8":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "small_bert/bert_en_uncased_L-8_H-768_A-12":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "small_bert/bert_en_uncased_L-10_H-128_A-2":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "small_bert/bert_en_uncased_L-10_H-256_A-4":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "small_bert/bert_en_uncased_L-10_H-512_A-8":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "small_bert/bert_en_uncased_L-10_H-768_A-12":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "small_bert/bert_en_uncased_L-12_H-128_A-2":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "small_bert/bert_en_uncased_L-12_H-256_A-4":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "small_bert/bert_en_uncased_L-12_H-512_A-8":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "small_bert/bert_en_uncased_L-12_H-768_A-12":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "bert_multi_cased_L-12_H-768_A-12":
                "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3",
            "albert_en_base":
                "https://tfhub.dev/tensorflow/albert_en_preprocess/3",
            "electra_small":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "electra_base":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "experts_pubmed":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "experts_wiki_books":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            "talking-heads_base":
                "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        }

        self.tfhub_handle_encoder = map_name_to_handle[bert_model_name]
        self.tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

        print(f"BERT model selected           : {self.tfhub_handle_encoder}")
        print(f"Preprocess model auto-selected: {self.tfhub_handle_preprocess}")