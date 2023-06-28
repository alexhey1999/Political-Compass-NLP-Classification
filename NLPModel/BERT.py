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
    def __init__(self, database, debug=False):
        super().__init__(database, debug)
        self.get_bert_model()
        self.metrics = [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.TruePositives(name='true-positives'),
            tf.keras.metrics.TrueNegatives(name='true-negatives'),
            tf.keras.metrics.FalsePositives(name='false-positives'),
            tf.keras.metrics.FalseNegatives(name='false-negatives')
        ]
        self.epochs = 5
        self.init_lr = 3e-5
        self.text_test = ['COULD SOMEBODY PLEASE EXPLAIN TO THE DERANGED, TRUMP HATING JACK SMITH, HIS FAMILY, AND HIS FRIENDS, THAT AS PRESIDENT OF THE UNITED STATES, I COME UNDER THE PRESIDENTIAL RECORDS ACT, AS AFFIRMED BY THE CLINTON SOCKS CASE, NOT BY THIS PSYCHOS’ FANTASY OF THE NEVER USED BEFORE ESPIONAGE ACT OF 1917. “SMITH” SHOULD BE LOOKING AT CROOKED JOE BIDDEN AND ALL OF THE CRIMES THAT HE HAS PERPETRATED ON THE AMERICAN PUBLIC, INCLUDING THE MILLIONS & MILLIONS OF DOLLARS HE EXTORTED FROM FOREIGN COUNTRIES!', 'WATCH: Brandon Herrera SLAMS the ATF for exceeding its authority over gun owners: “The ATF is not a legislative body. It is not Congress and it cannot make new laws. We shouldn’t have to worry about new gun laws that turn law-abiding citizens into felons, especially without an act of Congress.”','George Santos loses support of Kevin McCarthy', 'Planned Parenthood urges pregnant women to avoid Tennessee']
        
    def convert_to_dataframe(self, data, cat_list):
        final = [[i,j] for (i,j) in data]
        df = pd.DataFrame(final, columns=["Text", "Category"])
        df["Category"] = df["Category"].apply(lambda cat: cat_list.index(cat))
        final_dict = {}
        for i,cat in enumerate(cat_list):
            final_dict[i] = cat
        return df, final_dict
                
    def start(self):
        self.load_data()
        x_data, y_data = self.define_categories()
        x_dataframe, x_dataframe_dict = self.convert_to_dataframe(x_data, self.x_split_cats)
        y_dataframe, y_dataframe_dict = self.convert_to_dataframe(y_data, self.y_split_cats)
        self.full_build(x_dataframe, "left-right")
        self.full_build(y_dataframe, "libertarian-authoritarian")
    
    def full_build(self,full_dataftame, name):
        x_train, x_test, y_train, y_test = train_test_split(full_dataftame['Text'], full_dataftame["Category"], stratify=full_dataftame["Category"])
        classifier_model = self.build_classifier_model() 

        if self.debug:
            bert_raw_result = classifier_model(tf.constant(self.text_test))
            print(tf.sigmoid(bert_raw_result))
        
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        steps_per_epoch = len(list(x_train))
        num_train_steps = steps_per_epoch * self.epochs
        num_warmup_steps = int(0.1*num_train_steps)
        optimizer = optimization.create_optimizer(init_lr=self.init_lr, num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps, optimizer_type='adamw')
        classifier_model.compile(optimizer=optimizer, loss=loss, metrics=self.metrics)
        self.train_model(classifier_model, x_train, y_train, x_test, y_test)
        self.save_model(classifier_model, name)
        
    def save_model(self, classifier, name):
        saved_model_path = './NLPModel/BERT Models/{}_bert'.format(name.replace('/', '_'))
        classifier.save(saved_model_path, include_optimizer=True)
        
    def load_model(self, name):
        saved_model_path = './NLPModel/BERT Models/{}_bert'.format(name.replace('/', '_'))
        reloaded_model = tf.saved_model.load(saved_model_path)
        return reloaded_model
     
    def train_model(self, classifier, x_train, y_train, x_test, y_test):
        print(f'Training model with {self.tfhub_handle_encoder}')
        history = classifier.fit(x_train, y_train, epochs=self.epochs)
        classifier.evaluate(x_test, y_test)
        
    def predict_results(self, classifier, to_classifiy):
        y_pred = classifier.predict(to_classifiy)
        y_pred = y_pred.flatten()
        print(y_pred)
        
    def build_classifier_model(self):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(self.tfhub_handle_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(self.tfhub_handle_encoder, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
        return tf.keras.Model(text_input, net)
    
    def choose_option(self, classifier):
        print("1. Train Model")
        print("2. Predict")
        option = int(input("\n"))
        if option == 1:
            pass
        elif option == 2:
          pass  
        else:
            pass
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        
    def get_bert_model(self, bert_model_name = "small_bert/bert_en_uncased_L-4_H-512_A-8"):
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