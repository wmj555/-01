import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression

data = pd.read_csv("new_data.csv",sep=",")
print(data.shape)
print(data.head())

#取鲍鱼的其他特征比如性别、长度、直径、高度、整体重量、去壳后重量、脏器重量、壳的重量
X = data[["Sex","Length","Diameter continuous","Height","Whole weight","Shucked weight","Viscera weight","Shell weight"]]
print(X.head())
#环数为预测值
Y = data[["Rings"]]
print(Y.head())
#划分训练集与测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)
# print (X_train.shape)
# print (Y_train.shape)
# print (X_test.shape)
# print (Y_test.shape)
linreg = LinearRegression()
linreg.fit(X_train, Y_train)
print(linreg.intercept_)
print(linreg.coef_)
print(linreg.score(X_test,Y_test))
#结果为[3.18278593]
# [[ 7.10390414e-02  1.05670230e+01 -1.59918112e-05 -5.02177938e-06
# 1.05701874e+01 -2.22645264e+01 -1.03564630e+01  9.62477466e+00]]
#即 Rrings = 3.18278593-10.05701874 Sex-22.2645264 Length - 10.3564630
#模型拟合测试集
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(linreg, X, Y, cv=2)
fig, ax = plt.subplots()
ax.scatter(Y, predicted)
ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=1)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
