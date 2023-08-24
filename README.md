# QMUL-MSc-Project

## Introduction

This is my MSc Project at Queen Mary University of London. The purpose of this implementation is to utilise multiple APIs in order to collate a dataset that can be used to classify sentances or phrases on a political compass.

## Pre-Requisites

All modules that are used in this project are detailed in the requirements.txt file. This will need to be installed using PIP.

In addition, to utilize selenium, you will require chrome web driver in the path of your system.

This can be obtained from https://chromedriver.chromium.org/downloads

## Implementation

### Integrations Layer

This is the section that integrates with multiple sources and collates them into an sqllite database. All integrations and their unique statements can be seen in the Integrations folder.

base.py provides a super class of methods to write data to related databses as well as to hold data associated with the integration

- AlJazeera.py
- cato_institute.py
- gab.py
- occupy_democrats.py
- reddit_api.py (depricated)
- xinhua.py (depricated)

In addition, there is also a database.py file containing the database class. This allows easy reading and addition of records to each of the integrations detailed above.

This same database class is also utilised in openai.py which mirrrors the database.py implementation and adds in new methods to translate data using the stated prompt.

### NLP Model

BaseNLP.py acts as a superclass for both NLP Implementation consisting of similar properties and methods.

LinearSVC.py details the training, loading, testing and demoing of the LinearSVC NLP Method

BERT.py details the training, loading, testing and demoing of the BERT NLP Method

### Databases

All of the folloiwng at SQLite Databases and can be read using a tool such as https://sqlitebrowser.org/

database.db details the raw headline dataset

databaseopenai.db details the headline data and its converted value using OpenAI


### Grapher

This acts as the conversion between model values to graph values. This utilized the calculation detailed in the final report and accesses the www.politicalcompass.org website in order to get an accurate graph.
