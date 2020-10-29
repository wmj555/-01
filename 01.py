#-*- coding: utf-8 -*-
import  pandas as  pd
import numpy as np
import stats as stats
import sklearn
import scipy
import csv

# 读取数据
data = pd.read_csv("abalone.csv",sep=",")
data.columns = ["Sex","Length","Diameter continuous","Height","Whole weight","Shucked weight","Viscera weight","Shell weight","Rings"]
data.head()
print(data)
#标称数据连续化
def num(x):
    if x == "M":
        return 1
    elif x == "F":
        return -1
    else:
        return 0
data.Sex = data.Sex.apply(num)


#处理缺失值
null_all = data.isnull().sum()
print(null_all)
#发现缺失值为一，且在Shucked weight 列，定位到缺失值
Shuck_null = data[pd.isnull(data["Shucked weight"])]
print(Shuck_null)
data.fillna(data.median(),inplace=True)
print(data)
null_all = data.isnull().sum()
print(null_all)

#处理极端值
df = data
print(df.describe())
a=df.median()
b = df.mean()*4
df=df[df.abs()<1000]

df=df.fillna(a)
print(df.describe())

df.to_csv('new_data.csv')


