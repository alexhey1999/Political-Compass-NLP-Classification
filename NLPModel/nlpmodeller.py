import re
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.util import ngrams
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import LinearSVC, SVC
from nltk.classify import SklearnClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from alive_progress import alive_bar
import random



class NLPModel():
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
        
    def split_and_preprocess_data(self, data):
        """Split the data between train_data and test_data according to the percentage
        and performs the preprocessing."""
        train_data = []
        test_data = []

        num_samples = len(data)
        num_training_samples = int((self.percentage * num_samples))
        print("Generating training data")
        with alive_bar(len(data[:num_training_samples])) as bar:
            for (text, label) in data[:num_training_samples]:
                train_data.append((self.to_feature_vector(self.pre_process(text)),label))
                bar()
        
        print("Generating test data")
        with alive_bar(len(data[num_training_samples:])) as bar:
            for (text, label) in data[num_training_samples:]:
                test_data.append((self.to_feature_vector(self.pre_process(text)),label))
                bar()     
        
        return train_data, test_data

    def pre_process(self,text):
        text = text.lower()
        lemma = WordNetLemmatizer()
        words_filter = [""]
        text = re.sub(r"(\w)([.,;:!?'\"”\)\[\]])", r"\1 \2", text)
        text = re.sub(r"([.,;:!?'\"“\(\)\[\]])(\w)", r"\1 \2", text)
        words = text.split(" ")
        final_tokenized_string = []
        for word in words:
            if word not in stopwords.words('english'):
                pass
                lemma_word = lemma.lemmatize(word)
                final_tokenized_string.append(lemma_word)
        
        # remove blank words from final_tokenized_string
        final_tokenized_string = list(filter(lambda val: val not in words_filter, final_tokenized_string))
        
        # print(final_tokenized_string)
        return final_tokenized_string
    
        
    def to_feature_vector(self, tokens):
        # Should return a dictionary containing features as keys, and weights as values
        
        local_feature_dict = {}
        word_index = 0
        # Loop through every word in token
        for word in tokens:
            
            # Check to see if the word is not in the global_feature_dict
            if word in self.global_feature_dict:
                # get the index of the word
                word_index = self.global_feature_dict[word]
            else:
                # Add new word with corresponding index to global_feature_dict
                word_index = len(self.global_feature_dict) + 1
                self.global_feature_dict[word] = word_index
                
            # Check to see if word is in the local_feature_dict
            # Only applies to this set of tokens
            if word_index in local_feature_dict:
                # Increment value in local_feature_dict
                local_feature_dict[word_index] += 1/len(tokens)
            else:
                # Set the key of word to 1/length of tokens in local_feature_dict
                local_feature_dict[word_index] = 1/len(tokens)
                
        # Return local_feature_dict as global_feature_dict is set globally (as stated in the name)
        return local_feature_dict
    
    
    def cross_validate(self, dataset, folds, classifications):
        results = []
        fold_size = int(len(dataset)/folds) + 1    
        for i in range(0,len(dataset),int(fold_size)):
            # insert code here that trains and tests on the 10 folds of data in the dataset
            print("Fold start on items %d - %d" % (i, i+fold_size))
            
            # FILL IN THE METHOD HERE
            statements = [x[0] for x in dataset[i:i+fold_size]]
            correct_outputs = [x[1] for x in dataset[i:i+fold_size]]
            classifier = self.train_classifier(dataset[i:i+fold_size])
            predicted_outputs = self.predict_labels(statements,classifier)
            if len(set(predicted_outputs)) != len(classifications):
                # raise Exception("Classifier did not learn all labels")
                print("Case of classifier not learning all classes in fold")
                continue
            
            if self.debug:
                print(classification_report(correct_outputs,predicted_outputs,target_names=classifications))
            
            (precision, recall, f1, _) = precision_recall_fscore_support(correct_outputs,predicted_outputs, average = "macro")
            accuracy = accuracy_score(correct_outputs,predicted_outputs)
            
            results.append(( precision, recall, f1, accuracy))
        
        cv_results = []
        for result_index in range(4):
            sum_of_values = 0
            for result in results:
                sum_of_values += result[result_index]
            cv_results.append(sum_of_values/len(results))
        
        print(cv_results)
        return cv_results, classifier


    def predict_labels(self, samples, classifier):
        """Assuming preprocessed samples, return their predicted labels from the classifier model."""
        return classifier.classify_many(samples)


    def predict_label_from_raw(self, sample, classifier):
        """Assuming raw text, return its predicted label from the classifier model."""
        return classifier.classify(self.to_feature_vector(self.pre_process(sample)))
    
    def predict_label_list_from_raw(self, sample, classifier):
        prob = classifier.prob_classify(self.to_feature_vector(self.pre_process(sample)))
        non_zero = prob.samples()
        prob_dict = {}
        for i in non_zero:
            prob_dict[i] = prob.prob(i)
        return prob_dict

    def train_classifier(self, data):
        # pipeline =  Pipeline([('svc', LinearSVC())])
        # return SklearnClassifier(pipeline).train(data)
        return SklearnClassifier(SVC(kernel='linear',probability=True)).train(data)
    

    def start(self):
        # load in data
        # raw_data defined here
        self.load_data()
        
        x_data, y_data = self.define_categories()
        
        # train_data and test_data defined here
        x_test, x_train = self.split_and_preprocess_data(x_data)
        y_test, y_train = self.split_and_preprocess_data(y_data)

        print("Training X Classifier...")
        x_results, x_classifier = self.cross_validate(x_test, 10, self.x_split_cats)
        print("Training Y Classifier...")
        y_results, y_classifier = self.cross_validate(y_test, 10, self.y_split_cats)
        
        self.get_manual_prediction(x_classifier, y_classifier)
       

    def get_manual_prediction(self, x_classifier, y_classifier):
        print("Manual Classifier...")
        test_input = input("Input to Classify:\n")
        print("X Classifier")
        x_confidence = self.predict_label_list_from_raw(test_input, x_classifier)
        print(x_confidence)
        print("Y Classifier")
        y_confidence = self.predict_label_list_from_raw(test_input, y_classifier)
        print(y_confidence)
    

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