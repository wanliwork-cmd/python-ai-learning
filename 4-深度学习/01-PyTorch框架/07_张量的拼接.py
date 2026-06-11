# -*- coding: utf-8 -*-
# @Time    : 2026/6/11 11:39
# @Author  : WanLi
# @File    : 07_张量的拼接.py

"""
    总结:
        掌握：
            torch.cat(tensors, dim)：沿着指定维度拼接张量
            torch.stack(tensors, dim)：沿着新维度拼接张量
            torch.concat(tensors, dim)：沿着指定维度拼接张量
            torch.concatenate(tensors, dim)：沿着指定维度拼接张量
        张量拼接操作用于组合来自不同来源或经过不同处理的数据。
"""
import torch

"""
    cat：
        1- 不能修改张量的维度个数。例如：不能将2维变3维
        2- 除了拼接的维度以外，其他维度必须相同
"""
t1 = torch.randint(1,10,size=(2,3))
t2 = torch.randint(1,10,size=(2,3))
print(t1,t1.shape)
print(t2,t2.shape)


cat_1 = torch.cat([t1,t2],dim=0)
print(cat_1,cat_1.shape)

cat_2 = torch.cat([t1,t2],dim=1)
print(cat_2,cat_2.shape)


# 不能将2维变3维
# torch.cat([t1,t2],dim=2)


t1 = torch.randint(1,10,size=(2,3))
# 除了拼接的维度以外，其他维度必须相同
t2 = torch.randint(1,10,size=(5,3))
# t2 = torch.randint(1,10,size=(2,4))
print(t1,t1.shape)
print(t2,t2.shape)

cat_1 = torch.cat([t1,t2],dim=0)
print(cat_1,cat_1.shape)

"""
    stack：
        1- 可以修改张量的维度个数。例如：可以将2维变3维
        2- 拼接的维度必须是新维度
        3- 拼接的维度以外，其他维度必须相同
"""

