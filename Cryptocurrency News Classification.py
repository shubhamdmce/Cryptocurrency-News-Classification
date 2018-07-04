# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 14:19:22 2018

@author: kshir
"""
#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
data=pd.read_csv('book2.txt',encoding = "ISO-8859-1", delimiter='\t', quoting=3)

#cleaning the dataset
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,199):
    porter = PorterStemmer()
    news=re.sub('[^a-zA-Z]',' ',data['Title'][i]);
    news=news.lower()
    news=news.split()
    news=[porter.stem(word) for word in news if not word in set(stopwords.words('english'))]
    news=' '.join(news)
    corpus.append(news)
    
#creating the bag of word model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=840)
x=cv.fit_transform(corpus).toarray()
y=data.iloc[:,1].values

#splitting the data into the training set and test set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

#fitting naive bayes to the training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

#predicting the test set results 
y_pred=classifier.predict(x_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
(37+3)/40

                                        
