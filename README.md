# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Initialize weights and bias** randomly and set the learning rate 
2. **For each training example**, calculate the prediction and compute the gradient of the cost function with respect to weights and bias.
3. **Update weights and bias** using the gradient:
   ![image](https://github.com/user-attachments/assets/2fdc077e-cc2b-4037-8a87-76cd12c35a9e)

5. **Repeat** until convergence or for a fixed number of iterations.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: NATARAJ KUMARAN S
RegisterNumber:  212223230137
*/
```
```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor #Stochastic Gradient Descent Regressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
Y=np.column_stack((data.target,data.data[:,6]))
X=df.drop(columns=["AveOccup","target"],inplace=False)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

sgd = SGDRegressor(max_iter=1000, tol=1e-3)

multi_output_sgd = MultiOutputRegressor(sgd)


multi_output_sgd.fit(X_train, Y_train)


Y_pred = multi_output_sgd.predict(X_test)

Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)


mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)


print("\nPredictions:\n", Y_pred[:5])  
```

## Output:


![image](https://github.com/user-attachments/assets/7c49fcac-f9de-4f4e-9cfd-4d1c78d7eed6)





## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
