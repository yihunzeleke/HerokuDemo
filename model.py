# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 14:20:09 2020

@author: yihun
"""

import  numpy as np
import pandas as pd
import matplotlib.pylab as plt
import pickle

dataset = pd.read_csv("hiring.csv")

dataset['experience'].fillna(0, inplace = True)  

dataset['test_score'].fillna(dataset['test_score'].mean, inplace = True)

X = dataset.iloc[:,:3]

# converting words to integer values

def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6,
                 'seven':7, 'eight':8, 'nine':9, 'ten':10, 'eleven':11, '0':0
                 }
    return word_dict[word]
    

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:,-1]

# splitting data train and test

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

# Fitting model with train data

regressor.fit(X,y)

# Saving model to disk

pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the result

model = pickle.load(open('model.pkl','rb'))
print("The predicted salary will be {}".format(model.predict([[2,9,6]])))