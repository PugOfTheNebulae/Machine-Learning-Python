# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values #independent, years exp
y = dataset.iloc[:, 1].values #dependent, salary
#indexed at column 2 , doesnt include last one, really column1
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #makes "regressor" object of the class
regressor.fit(X_train, y_train) 
#machine is regressor, learns the linear regression fro, x,y train

#letting machine predict the results of x_test, compare 
#to actual salary
y_pred = regressor.predict(X_test)
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (training set)')
plt.xlabel('Years Exp.')
plt.ylabel('Salary')
plt.show()

y_pred = regressor.predict(X_test)
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#don't need to change because the line is the same 
#for test and train
plt.title('Salary vs Experience (test set)')
plt.xlabel('Years Exp.')
plt.ylabel('Salary')
plt.show()