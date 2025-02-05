#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset= pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values #gives only a vector/list
y= y.reshape(len(y),1) #needs to be a matrix 
x_original=X
y_original=y


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
print(X)
print(y)

#training the svr model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') #radial base function can handle non linear relationships and is versatile 
regressor.fit(X, y)

#predicting a new result 
print(sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1)))
#this reverses the scaling of the predicted result

#visualising the svr results
plt.scatter(x_original,y_original, color='red')
plt.plot(x_original, sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)), color ='blue'  )
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Postition level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()