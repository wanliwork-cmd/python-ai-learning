# -*- coding: utf-8 -*-
# @Time    : 2026/6/3 08:23
# @Author  : WanLi
# @File    : 03_张量与NumPy以及标量之间的转换.py

"""
     张量与NumPy以及标量之间的转换
     1. 张量与NumPy数组之间的转换
        1.1 张量转换为NumPy数组
            tensor.numpy()
        1.2 NumPy数组转换为张量
            np.array(tensor)

     2. 张量与标量之间的转换
        2.1 张量转换为标量
            tensor.item()
        2.2 标量转换为张量
            torch.tensor(scalar)
"""

import torch
import numpy as np

print("\n================ 1. 张量转numpy ================")
# 1.1 张量转换为NumPy数组
t1 = torch.tensor([1, 2, 3])
print(t1,type(t1))  # tensor([1, 2, 3]) <class 'torch.Tensor'>

# 1.2 NumPy数组转换为张量 共享内存（即修改一个，另一个也会修改）
t2 = t1.numpy()
print(t2,type(t2))  # [1 2 3] <class 'numpy.ndarray'>

t1[1] = 999
print(f'numpy--->{t2}')  # [  1 999   3]
print(f'张量--->{t1}')  # tensor([  1, 999,   3])

# 可以在后面使用copy()，不共享内存
t3 = t1.numpy().copy()
print(t3,type(t3))  # [1 2 3] <class 'numpy.ndarray'>
t1[1] = 888
print(f'numpy--->{t3}')  # [1 2 3]
print(f'张量--->{t1}')  # tensor([1, 888, 3])

#2.numpy数组转换为张量 np.array(tensor)
print("\n================ 2. numpy转张量 ================")
arr = np.array([11,22,33])

#不共享内存
t4 = torch.tensor(arr)
print(t4,type(t4))  #tensor([11, 22, 33]) <class 'torch.Tensor'>

#共享内存(值会被修改)
t5 = torch.from_numpy(arr)
print(t5,type(t5))  #tensor([11, 22, 33]) <class 'torch.Tensor'>

arr[0] = 9999
print(f'numpy--->{arr},t4--->{t4},t5--->{t5}')  # [9999  22  33]



print("\n================ 3. 张量与标量之间转换 ================")

# 3. 张量与标量之间进行转换
#3.1 标量转张量
t3 = torch.tensor(666)
print(t3,type(t3))  #tensor(666) <class 'torch.Tensor'>

# 3.2 张量转标量
t4 = t3.item()
print(t4,type(t4))  #666 <class 'int'>

# 3.3扩展 当
t5 = torch.tensor([1, 2, 3])
print(t5,type(t5))  # tensor([1, 2, 3]) <class 'torch.Tensor'>
t6 = t5.item()
print(t6,type(t6))  #RuntimeError: a Tensor with 3 elements cannot be converted to Scalar







