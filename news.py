
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df=pd.read_csv('/Users/prakashchandraprasad/Desktop/datasets/NewsAggregatorDataset/newsCorpora.csv',delimiter='\t',header=None)

X_train_raw,X_test_raw,y_train,y_test=train_test_split(df[1],df[4])

vec=TfidfVectorizer()

X_train=vec.fit_transform(X_train_raw)

X_test=vec.transform(X_test_raw)
   
classifier=LogisticRegression()

classifier.fit(X_train,y_train)

predictions=classifier.predict(X_test)

print 'Accuracy:', accuracy_score(y_test,predictions)

confusion_matrix=confusion_matrix(y_test,predictions)

print(confusion_matrix)



