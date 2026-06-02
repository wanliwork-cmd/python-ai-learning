# TODO 1.导包  导入numpy pandas matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from streamlit import columns
#解决中文乱码
plot.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
plot.rcParams['axes.unicode_minus'] = False


# TODO 2.读取数据  sep 默认按,切割
# 注意： 以后用的多的基本上都是通过读取文件，生成df对象
df = pd.read_csv('data/1960-2019全球GDP数据.csv',encoding='gbk',sep=',')

# TODO 3.了解数据
print(df.ndim)   #维度 2
print(df.shape)  #形状 (9931, 3)
# print(df.head(3))  #默认是前五行
# print(df.tail(3))  #默认是后五行
# print(df.columns)  #输出所有的列
# print(df.count())  #统计有多少行
# df.info()          #输出数据基本信息

# TODO 4.数据处理
df.dropna(inplace=True)
print(df.shape) #形状 (9930, 3)

# TODO 需求1：先查询中美日的数据
my_cn = df[df['country'] == '中国']
my_us = df[df['country'] == '美国']
my_jp = df[df['country'] == '日本']
print(my_cn.head())
print(my_us.head())
print(my_jp.head())
print("============================")

# TODO 需求2：把年份作为索引列
my_cn.set_index('year',inplace=True)
my_us.set_index('year',inplace=True)
my_jp.set_index('year',inplace=True)
# print(my_cn.head())
# print(my_us.head())
# print(my_jp.head())

# TODO 需求3.把GDP列名修改为中国，美国，日本
my_cn = my_cn.rename(columns={'GDP': "中国"})
my_us = my_us.rename(columns={'GDP': "美国"})
my_jp = my_jp.rename(columns={'GDP': "日本"})
print(my_cn.head())
print(my_us.head())
print(my_jp.head())

# TODO 需求4，可视化展示中美日的每年变化折线图
plot.title('1960-2019年中美日GDP数据折线图')
plot.plot(my_cn.index,my_cn['中国'],label='中国',color='red')
plot.plot(my_us.index,my_us['美国'],label='美国',color='blue')
plot.plot(my_jp.index,my_jp['日本'],label='日本',color='green')

#添加图例
plot.legend()
#添加网格
plot.grid()
#生成图
plot.show()