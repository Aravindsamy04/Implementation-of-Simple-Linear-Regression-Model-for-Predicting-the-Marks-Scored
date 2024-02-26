# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5.Predict the regression for the marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ARAVIND SAMY.P
RegisterNumber:  212222230011

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs scores(Test Data Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE : ",rmse)
*/
```

## Output:
### df.head() & df.tail():

![307657843-f7c2030e-b396-4100-8185-9645f06cefa0](https://github.com/Aravindsamy04/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497037/7f3237b5-6805-4869-bfda-63836bb730d2)

### Values of X:
![307658048-fed6b8b4-4ffd-4a1a-b04c-069e8076b1b2](https://github.com/Aravindsamy04/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497037/976d8ea2-0417-4c61-8b6e-08947ffdaad1)

### Values of Y:

![307658187-dbd21d59-3d91-4d7e-bc0f-42ed33018b3d](https://github.com/Aravindsamy04/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497037/94aa5133-1c0f-43d7-ab8c-656b53218092)

### Values of Y prediction:

![307658367-9b36d802-178f-4e26-8364-90a61e4d1b2d](https://github.com/Aravindsamy04/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497037/014765b5-4b66-4f41-a0c0-a82f5607139d)

### Values of Y test:
![307658568-c9348485-a9bd-4b26-814d-83d806d997be](https://github.com/Aravindsamy04/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497037/563f2648-8722-4313-98b4-95f3c98993d5)


### Training set graph:
![307658711-f7e51f8d-8aac-4b37-b2c6-0c08211e6dd0](https://github.com/Aravindsamy04/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497037/89a13c52-7838-4d75-9a70-58784b3f1963)

### Test set graph:

![307658871-e83fb8be-cd5b-4c56-a306-0fe6b68e0b07](https://github.com/Aravindsamy04/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497037/7a160d32-254d-495b-ba11-ead2bbf12d1e)

### Value of MSE,MAE & RMSE:

![307658971-25363217-d0b3-42b4-b8ba-1c401a0120a8](https://github.com/Aravindsamy04/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/113497037/8f55d5fe-d063-4b46-af99-e8b6c56837b1)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
















## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
