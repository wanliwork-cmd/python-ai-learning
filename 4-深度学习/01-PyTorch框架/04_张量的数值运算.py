# -*- coding: utf-8 -*-
# @Time    : 2026/6/3 09:30
# @Author  : WanLi
# @File    : 04_张量的数值运算.py

"""
    掌握：+ - * / @
    ### 基本运算
    加减乘除取负号： 这里取负号表示: 对张量中每个元素进行取负号运算 例如: 1 --> -1  -1 --> 1
     +、-、*、/、-
    add(other=)、sub、mul、div、neg
    `add_(other=)`、`sub_`、`mul_`、`div_`、`neg_`（其中带下划线的版本会修改原数据）
"""

import torch

# 定义张量, 浮点型.
t1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)

print(t1, t1.shape)

# 1- sum求和
# dim=0，按列求和
r1 = t1.sum(dim=0)
print(r1)

# dim=1，按行求和
r2 = t1.sum(dim=1)
print(r2)

# dim不设置值，对所有元素求和
r3 = t1.sum()
print(r3)


# 2- 均值，元素数据类型必须是float，不能是整数
t1 = torch.tensor([[1, 2, 3], [4, 5, 6]],dtype=torch.float32)
# t1 = torch.tensor([[1, 2, 3], [4, 5, 6]],dtype=torch.int32)

# r1 = t1.mean(dim=0)
r1 = t1.mean(dim=1)
print(r1)


# 3- 平方/立方/平方根/e的n次幂/对数
t1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)

print(t1.pow(2)) # 平方
print(t1.pow(3)) # 立方
print(t1.sqrt()) # 开根号
print(t1.exp()) # e的n次幂，元素作为幂使用
print(t1.log()) # 以e为底求对数
print(t1.log2()) # 以2为底的对数
print(t1.log10()) # 以10为底的对数
print(torch.log(t1) / torch.log(torch.tensor(3))) # 以3为底的对数（了解）

