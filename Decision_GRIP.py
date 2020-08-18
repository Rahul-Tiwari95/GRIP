# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 12:35:00 2020

@author: rahul
"""
#Decision Tree TASK 4
import pandas as pd
data = pd.read_csv("E:\\Data Science\\Data Sheet\\iris.csv")
data.head()
data['Species'].unique()
data.Species.value_counts()
colnames = list(data.columns)
predictors = colnames[1:5]
target = colnames[5]

# Splitting data into training and testing data set

import numpy as np
from sklearn.model_selection import train_test_split
train,test = train_test_split(data,test_size = 0.2)

from sklearn.tree import  DecisionTreeClassifier

model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])

preds = model.predict(test[predictors])
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)

# Accuracy = train
np.mean(train.Species == model.predict(train[predictors]))

# Accuracy = Test
np.mean(preds==test.Species) 


from sklearn import tree 
import matpoltlib.pyplot as plt
plt.figure(figsize=(24,15))
tree.plot_tree(model.fit(train[predictors],train[target]),features_names=predictors,filled=True,precision=3,
               proportion=True, rounded=True)
plt.show()