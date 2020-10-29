import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.linear_model import Ridge
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
#创建回归器，并进行训练
#创建岭回归实例
clf =Ridge(alpha=1.0,fit_intercept=True)
#调用fit函数使用训练集训练回归器
clf.fit(X_train,Y_train)

#当对所有输入都输 出同一个值时，拟合优度为0。
print(clf.score(X_test,Y_test))

alphas = np.logspace(-10,10,4000)

coefs = []
#实际计算中可选非常多的  值，做出一个岭迹图，看看这个图在取哪个值的时候变稳定了，那就确定  值了。
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)    #每个循环都要重新实例化一个estimator对象
    ridge.fit(X,Y)
    coefs.append(ridge.coef_)
# 展示结果
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alphas')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.savefig('p10.png')


def test_ling_alpha(*data):
    X_train, X_test, y_train, y_test = data
    alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    scores = []
    for i, alpha in enumerate(alphas):
        regr = linear_model.Ridge(alpha=alpha)
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test, y_test))
    ## 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alphas, scores)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title("Ling")
    plt.savefig("p10ling.png")
    plt.close()
test_ling_alpha(X_train, X_test, Y_train, Y_test)
