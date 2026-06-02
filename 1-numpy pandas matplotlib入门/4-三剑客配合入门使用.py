# -*- coding: utf-8 -*-
# @Time    : 2026/2/11 12:11
# @Author  : WanLi
# @File    : 4-三剑客配合入门使用.py
import matplotlib
# numpy pandas matplotlib

#需求: numpy负责生成数据 pandas负责转换数据为表格  matplotlib负责将数据转换为图表

# 导包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
#解决中文乱码
plot.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
plot.rcParams['axes.unicode_minus'] = False



# 1. numpy生成5行3列的数据 数据范围1-10
my_arr =  np.random.randint(low=1,high=100,size=(5,3))
print(my_arr,type(my_arr))

# 2.使用pandas给行和列加上表头
my_pd = pd.DataFrame(my_arr,index=['张三','李四','王五','赵六','田七'],columns=['语文','数学','生物'])
print(my_pd)

# 3.获取语文 数学 生物的总成绩
print("语文",my_pd['语文'].sum())
print("数学",my_pd['数学'].sum())
print("生物",my_pd['生物'].sum())
print("==="* 10)
print(my_pd.sum())

# 4.获取每个学生的成绩  axis参数 默认值为0   0表示统计列  1表示统计行
print(my_pd.sum(axis = 0))
print(my_pd.sum(axis = 1))

# 5.使用matplotlib可视化数据
fig,ax = plot.subplots()
ax.scatter(x=my_pd['语文'],y=my_pd['数学'])
ax.set_title('散列图测试')
plot.show()

# 6.画出每个学生的折线图
_, ax = plot.subplots()

for name in my_pd.index:
    ax.plot(
        my_pd.columns,          # X轴：科目
        my_pd.loc[name],        # Y轴：该学生成绩
        marker='o',
        label=name
    )

ax.set_title('学生成绩折线图')
ax.set_xlabel('科目')
ax.set_ylabel('分数')

ax.legend()  # 显示图例
plot.show()