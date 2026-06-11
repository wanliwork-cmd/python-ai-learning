# -*- coding: utf-8 -*-
# @Time    : 2026/6/11 11:35
# @Author  : WanLi
# @File    : 06_张量的形状.py

"""
    总结:
        掌握：
            reshape: 改变张量的形状
            squeeze: 去除张量中所有大小为1的维度
            unsqueeze: 在张量的指定位置插入一个维度
            transpose: 转置张量
            permute: 改变张量的维度顺序
        张量形状操作是指对张量的维度进行变换的一系列操作。
        张量的形状则描述了每个维度上的元素数量。

"""
print("-=======================reshape=========================")
import torch

data = torch.tensor([[10, 20, 30], [40, 50, 60]])
# 1. 使用 shape 属性或者 size 方法都可以获得张量的形状
print(data.shape, data.shape[0], data.shape[1])
print(data.size(), data.size(0), data.size(1)) # 效果同上

# 2. 使用 reshape 函数修改张量形状
new_data = data.reshape(1, 6)
print(new_data, new_data.shape)

new_data = data.reshape(1, -1)
print(new_data, new_data.shape)


print("-=======================squeeze和unsqueeze=========================")

import torch

# squeeze和unsqueeze
# 定义张量, 5个元素
t1 = torch.tensor([1, 2, 3, 4, 5])
print(t1,t1.shape)

# unsqueeze增加形状为1的维度
t2 = t1.unsqueeze(dim=0) # 1行5列
print(t2,t2.shape)
print("-"*30)

t3 = t1.unsqueeze(dim=1) # 5行1列
print(t3,t3.shape)

# squeeze删除所有形状为1的维度
t4 = t3.squeeze()
print(t4,t4.shape)

# 重新定义多维, 且包含1的维度.
t5 = torch.randint(1, 10, (2, 1, 3, 1, 5))
print(t5,t5.shape)

# 不设置参数，表示删除所有为1的维度
t6 = t5.squeeze() # 形状[2, 3, 5]
print(t6, t6.shape)
print("-"*30)

# 可以通过dim精准删除哪个轴上的1的维度
t7 = t5.squeeze(dim=1) # 形状[2, 3, 1, 5]
print(t7, t7.shape)

print("========================transpose和permute===========================")


t1 = torch.randint(1,10,(2,3,5))
print(t1, t1.shape)

# 需求1: 交换0轴 和 1轴.  (2, 3, 5) -> (3, 2, 5)
"""
    transpose（参数1，参数2）：注意每次只能交换两个维度的位置
        参数1、参数2 表示的是要交换哪几个轴的位置。参数传递顺序无所谓
"""
# 下面两个写法效果一样
# t2 = t1.transpose(dim0=1,dim1=0)
t2 = t1.transpose(dim0=0,dim1=1)
print(t2, t2.shape)
print("-"*30)


# 需求2: 从 (2, 3, 5) -> (5, 2, 3)
"""
    permute(dims)：同一时刻可以交换多个维度的位置。参数中传递的是维度顺序
"""
t3 = t1.permute(dims=[2,0,1])
print(t3, t3.shape)

