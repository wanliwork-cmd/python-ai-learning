# -*- coding: utf-8 -*-
# @Time    : 2026/6/11 11:42
# @Author  : WanLi
# @File    : 08_stack（堆叠）操作.py

"""
    把多个张量（tensor）沿着一个新的维度拼接起来。
    例如：有两个张量，形状分别为 (3, 4) 和 (3, 4)，沿着新的维度拼接起来，形状为 (2, 3, 4)。
    stack：
        1- 两个拼接的张量形状必须完全一样
        2- 会产生新维度，在新维度上进行拼接操作
"""

import torch

t1 = torch.randint(1,10,size=(5,6))
t2 = torch.randint(1,10,size=(5,6))
print(t1,t1.shape)
print(t2,t2.shape)

stack_1 = torch.stack([t1,t2],dim=0)
print(stack_1,stack_1.shape) # [2,5,6]

stack_2 = torch.stack([t1,t2],dim=1)
print(stack_2.shape) # [5,2,6]

stack_3 = torch.stack([t1,t2],dim=2)
print(stack_3.shape) # [5,6,2]