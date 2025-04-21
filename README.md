# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start the program

Step 2: Data Preparation 

Step 3: Hypothesis Definition 

Step 4: Cost Function 

Step 5: Parameter Update Rule 

Step 6: Iterative Training 

Step 7: Model Evaluation 

Step 8: End the program


## Program And Output:
```
Program to implement the multivariate linear regression model for predicting the price of the house and
number of occupants in the house with SGD regressor.

Developed by: SURIYA PRAKASH.S
RegisterNumber: 212223100055 
```
```

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
dataset = fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())
```


![image](https://github.com/user-attachments/assets/1ec4e143-a107-4c74-adfd-cdb67666db12)

```
X = df.drop(columns=['AveOccup','HousingPrice'])
X.info()
```
![image](https://github.com/user-attachments/assets/36cc262a-78e3-4523-b8f7-9525e5aaba3a)

```
Y = df[['AveOccup','HousingPrice']]
Y.info()
```

![image](https://github.com/user-attachments/assets/9ba395ab-ad56-4f91-8b23-7e1ce1ddfa77)

```
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
```
![image](https://github.com/user-attachments/assets/baef5f92-42bc-40e4-9bab-e5f66fe06084)

```
Y_pred = multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
```
![image](https://github.com/user-attachments/assets/9cf8d816-cd01-4509-98ba-d8611b74097a)


```
print("\nPredictions:\n", Y_pred[:5])
```
![image](https://github.com/user-attachments/assets/ca984f1a-6a78-4b9f-a926-4953da434357)



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
