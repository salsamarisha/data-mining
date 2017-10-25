# -*- coding: utf-8 -*-
"""
Created on Thu May 18 19:06:50 2017

@author: Marisha Salsabila
"""

#%%
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
X=pd.read_csv('data_train.csv')
test = pd.read_csv('data_test.csv')

#%%
X.n_non_stop_words.value_counts(dropna=False)
X.n_non_stop_unique_tokens.value_counts(dropna=False)
X.num_self_hrefs.value_counts(dropna=False)
X.num_imgs.value_counts(dropna=False)
X.data_channel.value_counts(dropna=False)
X.day.value_counts(dropna=False)
#%%
X.n_non_stop_unique_tokens= X.n_non_stop_unique_tokens.fillna(round(X.year.mean()))
X.n_non_stop_words= X.n_non_stop_words.fillna(round(X.n_non_stop_words.mean()))
X.n_tokens_content= X.n_tokens_content.fillna(round(X.n_tokens_content.mean()))
X.n_tokens_title= X.n_tokens_title.fillna(round(X.n_tokens_title.mean()))
X.n_unique_tokens= X.n_unique_tokens.fillna(round(X.n_unique_tokens.mean()))
X.num_hrefs=X.num_hrefs.fillna(round(X.num_hrefs.mean()))
X.num_self_hrefs=X.num_self_hrefs.fillna(round(X.num_self_hrefs.mean()))
X.num_imgs=X.num_imgs.fillna(round(X.num_imgs.mean()))
X.num_videos=X.num_videos.fillna(round(X.num_videos.mean()))

test.n_non_stop_unique_tokens= test.n_non_stop_unique_tokens.fillna(round(test.year.mean()))
test.n_non_stop_words= test.n_non_stop_words.fillna(round(test.n_non_stop_words.mean()))
test.n_tokens_content= test.n_tokens_content.fillna(round(test.n_tokens_content.mean()))
test.n_tokens_title= test.n_tokens_title.fillna(round(test.n_tokens_title.mean()))
test.n_unique_tokens= test.n_unique_tokens.fillna(round(test.n_unique_tokens.mean()))
test.num_hrefs=test.num_hrefs.fillna(round(test.num_hrefs.mean()))
test.num_self_hrefs=test.num_self_hrefs.fillna(round(test.num_self_hrefs.mean()))
test.num_imgs=test.num_imgs.fillna(round(test.num_imgs.mean()))
test.num_videos=test.num_videos.fillna(round(test.num_videos.mean()))
#%%
def dataToNumber(data_channel):
    if data_channel == 'bussiness':
        return 0
    if data_channel == 'technology':
        return 1
    if data_channel == 'world':
        return 2
    if data_channel == 'lifestyle':
        return 3
    if data_channel == 'social media':
        return 4
    if data_channel == 'entertainment':
        return 5
    if data_channel == 'other':
        return 6
X.data_channel=X.data_channel.apply(dataToNumber)
X.data_channel=X.data_channel.fillna(round(X.data_channel.mean()))
test.data_channel=test.data_channel.apply(dataToNumber)
test.data_channel=test.data_channel.fillna(round(test.data_channel.mean()))
def dayToNumber(day):
    if day == 'monday': 
        return 0
    if day == 'tuesday': 
        return 1
    if day == 'wednesday': 
        return 2
    if day == 'thursday': 
        return 3
    if day == 'friday': 
        return 4
    if day == 'saturday': 
        return 5
    if day == 'sunday': 
        return 6
test.day = test.day.apply(dayToNumber)
test.day=test.day.fillna(round(test.day.mean()))
X.day=X.day.apply(dayToNumber)
X.day=X.day.fillna(round(X.day.mean()))
#%%
data_y = X[['popularity']]
data_x = X.drop(['popularity','id'],axis=1)

test_x = test.drop(['id'],axis=1)
test_x.info()

#%%
data_x = pd.get_dummies(data_x)
test_x = pd.get_dummies(test_x)
test_x.head()
test_x.info()
#%%
knn = KNeighborsClassifier(5)
rf = RandomForestClassifier()
nb = GaussianNB()
dectree = DecisionTreeClassifier()
logreg = LogisticRegression()
svm = SVC()
mlp = MLPClassifier()
#%%
clf=svm
clf.fit(data_x, np.ravel(data_y))
score = cross_val_score(clf, data_x, np.ravel(data_y), cv=10, scoring='accuracy')

print(np.mean(score))
#%%
submission = pd.read_csv('sample_submission (1).csv')
#%%
clf.fit(data_x, np.ravel(data_y))
predict = clf.predict(test_x)

#%%
submit = pd.DataFrame()
submit['id']=test.id
submit['popularity']=predict
submit.to_csv('hasil.csv',index=False)