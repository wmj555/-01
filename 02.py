#非常大或者非常小的，明显错误的数据处理。
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
# 导入试验数据
df = pd.read_csv('abalone.csv')
dd = open('abalone.csv')
#画散点图，观察异常值
reads = csv.reader(dd)
list = []
for line in reads:
   list.append(line)
list1 = list[0]
data1 = df.T
for i in range(1,9):
    print(i)
    print(list1[i])
    list1[i] = data1.values[i]
    plt.plot(list1[i], 'o')
    plt.savefig('p'+str(i)+'.png')
    plt.close()


'''
#画箱型图观察异常值
fig,axes = plt.subplots(1,8)

# boxes表示箱体，whisker表示触须线
# medians表示中位数，caps表示最大与最小值界限

df.plot(kind='box', ax=axes, subplots=True,
          title='Different boxplots', sym='r+')
# sym参数表示异常值标记的方式
fig.subplots_adjust(wspace=6,hspace=6)  # 调整子图之间的间距
'''
#在图片中观察到异常值，并使用平均值替换
