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
from sklearn.svm import LinearSVC
from nltk.classify import SklearnClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support


class NLPModel():
    def __init__(self,database):
        self.database = database
        self.percentage = 0.8
        
   
    def load_data(self):
        self.data = self.database.get_all_data()
        return self.data
    
   
    def format_data(self):
        # Initialize raw data
        self.raw_data = []
        for index, row in enumerate(self.data):
            self.raw_data.append((row[2],row[3]))
   
   
    def split_and_preprocess_data(self):
        """Split the data between train_data and test_data according to the percentage
        and performs the preprocessing."""
        self.train_data = []
        self.test_data = []
        
        num_samples = len(self.raw_data)
        num_training_samples = int((self.percentage * num_samples))
        for (text, label) in self.raw_data[:num_training_samples]:
            self.train_data.append((self.to_feature_vector(self.pre_process(text)),label))
        for (text, label) in self.raw_data[num_training_samples:]:
            self.test_data.append((self.to_feature_vector(self.pre_process(text)),label))
                     

    def pre_process(self,text):
        text = text.lower()
        lemma = WordNetLemmatizer()
        text = re.sub(r"(\w)([.,;:!?'\"”\)\[\]])", r"\1 \2", text)
        text = re.sub(r"([.,;:!?'\"“\(\)\[\]])(\w)", r"\1 \2", text)
        words = text.split(" ")
        final_tokenized_string = []
        for word in words:
            if word not in stopwords.words('english'):
                pass
                lemma_word = lemma.lemmatize(word)
                final_tokenized_string.append(lemma_word)
        return final_tokenized_string
    
        
    def to_feature_vector(self, tokens):
        # Should return a dictionary containing features as keys, and weights as values
        self.global_feature_dict = {}
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
    
    
    def cross_validate(self, dataset, folds, classifications = None):
        results = []
        fold_size = int(len(dataset)/folds) + 1    
        for i in range(0,len(dataset),int(fold_size)):
            # insert code here that trains and tests on the 10 folds of data in the dataset
            print("Fold start on items %d - %d" % (i, i+fold_size))
            
            # FILL IN THE METHOD HERE
            classifications = ["Left","Right", "Authoritarian", "Libertarian"]
            statements = [x[0] for x in dataset[i:i+fold_size]]
            correct_outputs = [x[1] for x in dataset[i:i+fold_size]]
            
            classifier = self.train_classifier(dataset[i:i+fold_size])
            predicted_outputs = self.predict_labels(statements,classifier)
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
        return cv_results


    def predict_labels(self, samples, classifier):
        """Assuming preprocessed samples, return their predicted labels from the classifier model."""
        return classifier.classify_many(samples)


    def predict_label_from_raw(self, sample, classifier):
        """Assuming raw text, return its predicted label from the classifier model."""
        return classifier.classify(self.to_feature_vector(self.pre_process(sample)))


    def train_classifier(self, data):
        print("Training Classifier...")
        pipeline =  Pipeline([('svc', LinearSVC())])
        return SklearnClassifier(pipeline).train(data)
    

    def start(self):
        # load in data
        self.load_data()
        
        # raw_data defined here
        self.format_data()
        
        # train_data and test_data defined here
        self.split_and_preprocess_data()
        
        print("Now %d rawData, %d trainData, %d testData" % (len(self.raw_data), len(self.train_data), len(self.test_data)), "Preparing training and test data...",sep='\n')
        
        x_axis_classifier = self.cross_validate(self.train_data, 10)
        
        
    

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