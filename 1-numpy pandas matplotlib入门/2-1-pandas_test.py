# -*- coding: utf-8 -*-
# @Time    : 2026/2/ 12:10
# @Author  : WanLi
# @File    : 2-pandas_test.py

# 导包
import pandas as pd

# 将列表转换为pandas对象
my_list = [10,20,30,40]
# 创建Series对象(类似一列)
my_pd = pd.Series(my_list,name="分数")
print(my_pd,type(my_pd))    #dtype: int64 <class 'pandas.core.series.Series'>

# 创建DataFrame对象(类似表格)  当数组是一维的时候显示的是一列
my_list2 = [10,20,30,40,50,60,70]
my_df = pd.DataFrame(my_list2,columns=['nums'])
print(my_df,type(my_df))

print("="* 20)

my_list3 = [[11,12],[21,22],[31,32],[41,42]]  # 多维的时候一个元素代表一行
my_df2 = pd.DataFrame(my_list3,columns=['nums1','nums2'])
print(my_df2,type(my_df2))

# pandas相比numpy多了所有 列名
# pandas相比numpy多了所有 列名
print(my_df2['nums2'])   #根据列名获取某一列
print(my_df2['nums2'].sum())
print(my_df2['nums2'].max())
print(my_df2['nums2'].min())
print(my_df2['nums2'].mean())