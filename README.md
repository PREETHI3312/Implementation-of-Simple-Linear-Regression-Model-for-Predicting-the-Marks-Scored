# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PREETHI A K
RegisterNumber: 212223230156

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')

# display the content in datafield
df.head()
df.tail()

#Segregating data to variables

x = df.iloc[:,:-1].values
print(x)


y = df.iloc[:,1].values
print(y)

#Splitting train and test data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

#displaying predicted values
print(y_pred)

#displaying actual values
print(y_test)

#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)



```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
# Displaying the content in datafield

# head:
![image](https://github.com/user-attachments/assets/26da9207-078b-47e1-a65a-5567b58961ec)

# tail:
![image](https://github.com/user-attachments/assets/e9c2e062-83ea-4b85-bce9-59169a8f92fb)

# Segregating data to variables

![image](https://github.com/user-attachments/assets/42c9efdb-a110-4f02-afa2-c166d35c5329)

![image](https://github.com/user-attachments/assets/5b8ca5a5-f11b-4b94-98cb-2584234dcf94)

# Displaying predicted values

![image](https://github.com/user-attachments/assets/95038c76-689d-4e86-94d4-06c7b706f0ea)

# Displaying actual values

![image](https://github.com/user-attachments/assets/1a61dd1a-2f23-429b-a865-69b6a9be18a0)

# Graph plot for training data

![image](https://github.com/user-attachments/assets/e3132461-e5d0-499d-a866-0d53b5298abb)

# Graph plot for test data

![image](https://github.com/user-attachments/assets/0fef34ce-2e74-4132-8203-1fd79322161f)

# MSE MAE RMSE

![image](https://github.com/user-attachments/assets/5dbe3c3d-02f3-4329-bcaf-647c5c95403e)






















## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
