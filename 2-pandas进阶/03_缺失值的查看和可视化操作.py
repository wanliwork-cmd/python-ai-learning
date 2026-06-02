# 1.导包
import pandas as pd
import matplotlib.pyplot as plt
#windows中文乱码
# import matplotlib
# matplotlib.use('TkAgg')
#mac中文乱码
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
# 2.读取数据
train = pd.read_csv('data/titanic_train.csv')
test = pd.read_csv('data/titanic_test.csv')
# 3.了解数据
print(train.shape)  # (892, 12)
train.info()  # 每列不为空的个数
print(train.count())
print('==================================================')
# 4.缺失值查看
# 4.1 查看是否是缺失值,结果是True或者False
print(train.isna())
print('----------------------------------')
# 4.2 获取每列有缺失值的个数(True:1,False:0)
print(train.isna().sum())
print('----------------------------------')
# 4.3 获取每列缺失值的比例
print(train.isna().sum() / len(train) * 100)
print('==================================================')
# 5.了解生存和死亡的人数,以及可视化展示
# 查看Survived各个值的个数
result = train['Survived'].value_counts()
print(result, type(result))
# 可视化展示
result.plot(kind='bar')  # 柱状图
plt.grid()
# 展示
plt.show()
