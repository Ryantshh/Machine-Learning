#importing the libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:, 1:-1].values #just the middle col
Y=dataset.iloc[:, -1].values

#Training the linear regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)

#Training the Polynomial Regression model on the whole dataset
#create the polynomial features which are X^n features  
from sklearn.preprocessing import PolynomialFeatures
poly_reg_features=PolynomialFeatures(degree=4)
X_poly=poly_reg_features.fit_transform(X) #new matrix of features 
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

#visualising the linear regression results
plt.scatter(X ,Y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualising the Polynomial Regression results
plt.scatter(X ,Y, color='red')
plt.plot(X,lin_reg_2.predict(X_poly),color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])

# Predicting a new result with Polynomial Regression
print(lin_reg_2.predict(poly_reg_features.fit_transform([[6.5]])))