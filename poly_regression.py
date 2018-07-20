# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

"""independent is exp, dep is salary"""
dataset = pd.read_csv('Position_Salaries.csv')

"""want x to be a matrix and y an array"""
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set

"""
-not enough info for test and train
-scaling isn't necessary
"""

"""
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
polyreg = PolynomialFeatures(degree = 5)
X_poly = polyreg.fit_transform(X)

"""
refitting the linear using the 
powers of the "years exp"""
linreg2 = LinearRegression()
linreg2.fit(X_poly, y)

plt.scatter(X, y)
plt.plot(X, linreg.predict(X))
plt.title('Truth vs Lie')
plt.xlabel('position lvl')
plt.ylabel('salary')
plt.show()

"""
-can't just replace with linreg2
-must use matrix of data : polyreg.fit.transform(X)
"""
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y)
plt.plot(X_grid, linreg2.predict(polyreg.fit_transform(X_grid)))
plt.title('Truth vs Lie')
plt.xlabel('position lvl')
plt.ylabel('salary')
plt.show()

"""finding the actual prediction using 6.5 instead 
of all x: linear and poly"""
linreg.predict(6.5)
linreg2.predict(polyreg.fit_transform(6.5))




















