# -*- coding: utf-8 -*-
# @Time    : 2026/5/30 06:51
# @Author  : WanLi
# @File    : 01_张量入门以及创建.py

"""
    张量：说的直白一点就是多维数组，是深度学习中最基本的数据结构
    创建张量：
        1.使用torch.tensor()函数，将列表或元组转换为张量
        2.使用numpy数组创建张量
        3.使用容器创建张量

"""
import torch
import numpy as np

# 使用tensor()创建张量
t1 = torch.tensor(999)  #使用标量创建张量
print(t1.dtype)
print(f't1的值为：{t1},类型为：{type(t1)}')  #t1的值为：999,类型为：<class 'torch.Tensor'>

# 使用numpy数组创建张量
narr = np.random.randn(2,3)