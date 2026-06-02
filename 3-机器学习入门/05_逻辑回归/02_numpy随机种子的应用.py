# -*- coding: utf-8 -*-
# @Time    : 2026/5/28 11:29
# @Author  : WanLi
# @File    : 02_numpy随机种子的应用.py

import numpy as np

# 设置随机种子以确保结果可重复
# 原理：随机种子是一个初始值，通过这个初始值可以生成一个随机数序列。设置相同的随机种子，可以生成相同的随机数序列。
np.random.seed(100)

# 生成随机数
num1 = np.random.randint(0,10,5)
print(f'第一次生成的{num1}')

# 再次生成随机数
num2 = np.random.randint(0,10,5)
print(f'第二次生成的{num2}')