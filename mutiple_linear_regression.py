import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_csv('BostonHousing.csv')
x=df[['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','b','lstat']]
y=df['medv']

#splitting the dataset
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)

#multpile linear regression
mult_lr=LinearRegression()
mult_lr.fit(x_train,y_train)

printing coefficients and intercept
print("Intercept: ", mult_lr.intercept_)
print(list(zip(x, mult_lr.coef_)))


#predicting particular value
data_to_predict = pd.DataFrame({'crim':[0.04741],'zn':[0],'indus':[11.93],'chas':[0],'nox':[0.573],'rm':[6.03],'age':[80.8],'dis':[2.505],'rad':[1],'tax':[273],'ptratio':[21],'b':[396.9],'lstat':[7.88]})

pred=mult_lr.predict(data_to_predict)
print(pred)
