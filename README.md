# QMUL-MSc-Project

## Introduction

This is my MSc Project at Queen Mary University of London. The purpose of this implementation is to utilise multiple APIs in order to collate a dataset that can be used to classify sentances or phrases on a political compass.

## Pre-Requisites

All modules used in this project are detailed in the requirements.txt file. This will need to be installed using PIP.

In addition, to utilize selenium, you will require Chrome web driver in the path of your system.

This can be obtained from https://chromedriver.chromium.org/downloads

## Steps to run

**NOTE: If you would like to test the models used in the report, you must first download the entire contents of https://drive.google.com/drive/folders/1MGyfUPWsNIlOvyyyw8yyfdb1OEFFcoKs?usp=sharing. Once this is done, you should place all 3 folders under the NLPModel Folder. LinearSVC savefile exists in the GitHub repo however you should replace ALL files**

```
pip install -r requirements.txt
```

### General Running Command

All functions can be run from the central main file with the following command.

```
python main.py [option] [--debug]
```

#### Possible Options

[option] | Use | Notes
--- | --- | ---
```collect``` | Initialises the integration process and runs through each site in order of collection | As the database has already been created, there is no need to run this unless to refresh the database with new data
```clear``` | Drops the entire training data from the .db files | Use with caution as rebuilding can take up to days to re-obtain the data
```nlp-l``` | Starts the training for LinearSVM Method | Models will be overwritten so it is recommended to avoid this step when testing
```nlpload-l``` | Loads the LinearSVM Trained model and runs a few test prompts | 
```nlp-b``` | Starts the training for LinearSVM Method | Models will be overwritten so it is recommended to avoid this step when testing
```nlpload-b``` | Loads the LinearSVM Trained model and runs a few test prompts|
```openai``` | Starts the OpenAI Conversion from original headline data to tweet format | Will overwrite the original database. Credentials used for this were imported in a .env file however this has been removed to save accidental spreading of API Keys
```testing``` | Testing endpoint for miscellaneous tasks | Depricated 
```demo``` | Runs BERT OpenAI model with allowance for a dynamic prompt | Used in presentation and will be the best one to run for dynamic testing

```--debug ``` can be used for showing logs for each step. This is not recommended as it can become cluttered.


## Implementation

### Integrations Layer

This section integrates with multiple sources and collates them into an SQLite database. All integrations and their unique statements can be seen in the Integrations folder.

base.py provides a superclass of methods to write data to related databases as well as to hold data associated with the integration

- AlJazeera.py
- cato_institute.py
- gab.py
- occupy_democrats.py
- reddit_api.py (deprecated)
- xinhua.py (deprecated)

In addition, there is also a database.py file containing the database class. This allows easy reading and addition of records to each of the integrations detailed above.

This same database class is also utilised in openai.py which mirrors the database.py implementation and adds in new methods to translate data using the stated prompt.

### NLP Model

BaseNLP.py acts as a superclass for both NLP Implementations consisting of similar properties and methods.

LinearSVC.py details the training, loading, testing and demoing of the LinearSVC NLP Method

BERT.py details the training, loading, testing and demoing of the BERT NLP Method

### Databases

All of the following at SQLite Databases and can be read using a tool such as https://sqlitebrowser.org/

database.db details the raw headline dataset

databaseopenai.db details the headline data and its converted value using OpenAI


### Grapher

This acts as the conversion between model values to graph values. This utilized the calculation detailed in the final report and accessed the www.politicalcompass.org website in order to get an accurate graph.
