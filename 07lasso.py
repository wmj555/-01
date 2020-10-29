import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

data = pd.read_csv("new_data.csv", sep=",")
print(data.shape)
print(data.head())

# 取鲍鱼的其他特征比如性别、长度、直径、高度、整体重量、去壳后重量、脏器重量、壳的重量
X = data[["Sex", "Length", "Diameter continuous", "Height", "Whole weight", "Shucked weight", "Viscera weight",
          "Shell weight"]]
print(X.head())

# 环数为预测值
Y = data["Rings"]
print(Y.head())

# 划分训练集与测试机
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)

from sklearn.linear_model import Lasso, LassoCV

# 通过交叉验证找到最优λ
Lambdas = np.logspace(-10, 10, 4000)
lasso_cv = LassoCV(alphas=Lambdas, normalize=True, cv=10, max_iter=10000)
lasso_cv.fit(X_train, Y_train)
lasso_cv.alpha_
# 1.86911411820748e-07

lasso = Lasso(alpha=lasso_cv.alpha_, normalize=True, max_iter=10000)
lasso.fit(X_train, Y_train)

# mean_squared_error(Y_test,lasso_pre)
# 0.616976900661324
print(lasso.score(X_test, Y_test))


def test_Lasso_alpha(*data):
    X_train, X_test, y_train, y_test = data
    alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    scores = []
    for i, alpha in enumerate(alphas):
        regr = linear_model.Lasso(alpha=alpha)
        regr.fit(X_train, y_train)
        scores.append(regr.score(X_test, y_test))
    ## 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alphas, scores)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"score")
    ax.set_xscale('log')
    ax.set_title("Lasso")
    plt.savefig("p10lasso.png")
    plt.close()





# 调用 test_Lasso_alpha
test_Lasso_alpha(X_train, X_test, Y_train, Y_test)

coef=[]
alphas = np.linspace(0.01,0.2,20)
for alpha in alphas:
    lasso = linear_model.Lasso(alpha=alpha)
    lasso.fit(X_train,Y_train)
    coef.append(lasso.coef_)

ax = plt.gca()
ax.plot(alphas, coef)

plt.legend(['Sex','Length','Diameter','Height',
            'Whole weight','Shucked weight',
            'Viscera weight','Shell weight'])
plt.xlabel('alpha',fontsize=15)
plt.ylabel('weights',fontsize=15)
plt.show()
plt.savefig("p11lasso.png")
plt.close()
