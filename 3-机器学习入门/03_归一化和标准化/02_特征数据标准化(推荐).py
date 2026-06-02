# -*- coding: utf-8 -*-
# @Time    : 2026/5/26 10:12
# @Author  : WanLi
# @File    : 01_特征数据标准化.py

#  标准化处理
"""
公式：z = (x - u) / s    x=原始数据  u=均值  s=标准差  标准差解释：标准差是数据分布的离散程度，越小表示数据越集中。
标准化处理：也称为Z-score标准化，是将数据减去均值，然后除以标准差。
标准化处理后的数据均值为0，标准差为1。
标准化处理适用于正态分布数据，不适用于 skewed data（偏态数据）。
"""

# 1.导包
from sklearn.preprocessing import StandardScaler

# 2.准备数据（模拟数据）
x_train = [
    [90, 20, 10, 40],
    [60, 4, 15, 45],
    [75, 3, 13, 46]
]
# 3.在模型训练化之前进行规范化数据
# 3.1创建归一化模型
model = StandardScaler()

# 3.2.规范化数据
new_x_train = model.fit_transform(x_train)

# 4.打印结果
print(new_x_train)




