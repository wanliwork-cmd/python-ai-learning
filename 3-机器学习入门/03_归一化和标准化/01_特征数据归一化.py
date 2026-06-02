# -*- coding: utf-8 -*-
# @Time    : 2026/5/26 10:12
# @Author  : WanLi
# @File    : 01_特征数据归一化.py

# 归一化处理
"""
公式： x' = (x - x_min) / (x_max - x_min)  x=原始数据  x_min=最小值  x_max=最大值
归一化处理：也称为Min-Max标准化，是将数据减去最小值，然后除以最大值与最小值的差。
归一化处理后的数据在0-1之间。
归一化处理适用于所有数据，但会改变数据的分布情况。
"""


# 1.导包
from sklearn.preprocessing import MinMaxScaler

# 2.准备数据（模拟数据）
x_train = [
    [90, 2, 10, 40],
    [60, 4, 15, 45],
    [75, 3, 13, 46]
]
# 3.在模型训练化之前进行规范化数据
# 3.1创建归一化模型
model = MinMaxScaler()

# 3.2.规范化数据
new_x_train = model.fit_transform(x_train)

# 4.打印结果
print(new_x_train)




