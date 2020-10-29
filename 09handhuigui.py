import pandas as pd
from sklearn import datasets, linear_model
from sklearn.linear_model import Ridge
#普通线性模型、岭回归模型、lasso模型
from sklearn.model_selection import train_test_split, cross_val_predict
import numpy as np
data = pd.read_csv("new_data.csv",sep=",")
print(data.shape)
print(data.head())
#取鲍鱼的其他特征比如性别、长度、直径、高度、整体重量、去壳后重量、脏器重量、壳的重量
X = data[["Sex","Length","Diameter continuous","Height","Whole weight","Shucked weight","Viscera weight","Shell weight"]]
print(X.head())

#环数为预测值
Y = data["Rings"]
print(Y.head())

#划分训练集与测试机
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)

#b = (X'X)-1 (X'Y)
def linear(X,Y):
    q = np.zeros_like(X.shape[1])#构造新矩阵用来存放
    if np.linalg.det(X.T.dot(X)) != 0:#若矩阵不为0
        q = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)#矩阵求逆与乘法运算
    return q
q1 = linear(X_train,Y_train)
print(q1)


def ridgr(X,Y,l):
    m = np.eye(X.shape[1])#生成对角线为1，其余位置为0的数组。
    #m[X.shape[1]-1][X.shape[1]-1] = 0
    q = np.linalg.inv(X.T.dot(X)+l *m).dot(X.T).dot(Y)#套公式
    return q
q2 = ridgr(X_train,Y_train,1)
print(q2)