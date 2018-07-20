# -*- coding: utf-8 -*-
"""
Created on Mon May 28 22:29:28 2018

@author: alyssa
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
# dealing w/ missing data
from sklearn.preprocessing import Imputer #class
#ctrl i to get info on how to use 'Imputer'
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#take mean where there is missing data, indexed
#at 0, doesn't include last column
#':,' gets columns
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

#encoding catagorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x = LabelEncoder()
#fixes first column-'country',replaces
x[:,0] = labelencoder_x.fit_transform(x[:,0])
"""
Doing it this way thinks that some countries will
be greater than others based on #'s, so use 
dummy variables

use three columns; if row is for Germany: Germany ==1,
Spain and France == 0
"""
#2nd input of OneHotEncoder, tells which column categorical values apply to
onehotencoder = OneHotEncoder(catagorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()

#taking care of purchase
labelencoder_y = LabelEncoder()
y[:,0] = labelencoder_y.fit_transform(y)

#splitting data set into training and test
from sklearn.cross_validation import train_test_split
"""
takes in arrays: x= independent, y=dependent
test size is 20% of the set, so train set will be 80%
-possible to 'overfit the data'"""
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

#scaling the data
from sklear.processing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
#don't need to fit the test
x_test = sc_x.transform(x_test)
"""
fit/transform dummy?
depends on the context, scaling makes data easier to interp on the
categorical data, but then you will lose the meaning behind the values:
    i.e Germany == 0, Yes == 1 etc.
 
do not need to apply feature scaling to the dependent variable y
apply the scaling if range on y is huge """


    