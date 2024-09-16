#import the libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the data set
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Splitting the dataset into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Training the Simple Linear Regression model on the Training Set
from sklearn.linear_model import LinearRegression #linearRegression is the class
regressor= LinearRegression() #get an obj of the class
regressor.fit(X_train,y_train) 
#fit the training data and finds the best 
# fitting line for the data this essentially is the best linear equation


#Predict the test set results
y_pred = regressor.predict(X_test) 
#Y value predicted based on the model and test set X values

#visualising against the training set results (TRAINING SET)
plt.scatter(X_train,y_train, color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#visualising against the test set results (TEST SET) shows how well model fits into new data
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#find the equation of the model 
coefficient = regressor.coef_[0]
intercept = regressor.intercept_

print(f'Linear Equation: Y = {intercept:.2f} + {coefficient:.2f}X')


# Loop to accept user input and predict Y value
while True:
    user_input = input("Enter an X value (or type 'exit' to quit): ")
    
    if user_input.lower() == 'exit':
        break
    
    try:
        X_value = float(user_input)
        predicted_Y = regressor.predict([[X_value]])
        print(f'The predicted Y value for X = {X_value} is {predicted_Y}')
    except ValueError:
        print("Invalid input. Please enter a numeric value for X or type 'exit' to quit.")