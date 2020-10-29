#导入相关库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series,DataFrame
from sklearn.model_selection import train_test_split
#普通线性模型、岭回归模型、lasso模型
from sklearn import datasets, linear_model
from sklearn.linear_model import Ridge,Lasso
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LinearRegression

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

def linreg(*data):
    #普通线性回归
    linreg = LinearRegression()
    linreg.fit(X_train, Y_train)
    print(linreg.intercept_)
    print(linreg.coef_)
    print(linreg.score(X_test,Y_test))
    x_pre_test = linreg.predict(X_test)
    plt.plot(np.arange(len(x_pre_test[1:100])), x_pre_test[1:100])
    plt.plot(np.arange(len(Y_test[1:100])), Y_test[1:100])
    plt.title("linreg")
    plt.savefig('01linreg.png')
    plt.close()

def ridge(*data):
    #创建岭回归实例
    clf =Ridge(alpha=1.0,fit_intercept=True)
    clf.fit(X_train,Y_train)
    print(clf.intercept_)
    print(clf.coef_)
    print(clf.score(X_test,Y_test))
    y_pre_ridge = clf.predict(X_test)
    plt.plot(np.arange(len(y_pre_ridge[0:150])), y_pre_ridge[0:150])
    plt.plot(np.arange(len(Y_test[0:150])), Y_test[0:150])
    plt.title("Ridge")
    plt.savefig('02ridge.png')
    plt.close()

def lasso(*data):
    #lasso回归
    lasso = Lasso(alpha=0.001, normalize=True)
    lasso.fit(X_train, Y_train)
    print(lasso.intercept_)
    print(lasso.coef_)
    print(lasso.score(X_test, Y_test))
    y_pre_lasso = lasso.predict(X_test)
    plt.plot(np.arange(len(y_pre_lasso[0:150])), y_pre_lasso[0:150])
    plt.plot(np.arange(len(Y_test[0:150])), Y_test[0:150])
    plt.title("lasso")
    plt.savefig('03lasso.png')
    plt.close()
linreg(X_train, X_test, Y_train, Y_test)
ridge(X_train, X_test, Y_train, Y_test)
lasso(X_train, X_test, Y_train, Y_test)