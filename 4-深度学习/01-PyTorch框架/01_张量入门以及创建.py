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
from torch import dtype

# 1.使用tensor()创建张量
t1 = torch.tensor(data=999)  # 使用标量创建张量
print(f't1的值为：{t1},类型为：{type(t1)}')  # t1的值为：999,类型为：<class 'torch.Tensor'>

# 使用numpy数组创建张量
narr = np.random.randn(2, 3)
t2 = torch.tensor(data=narr)  # dtype=torch.float64
print(f't2的值为：{t2},类型为：{type(t2)}')  # t2的值为：tensor([[ 0.1234, -0.5678,  0.9101],
#                [-0.2345,  0.6789, -0.1012]])
#        types: torch.FloatTensor

# 使用容器来创建张量
t3 = torch.tensor(data=[1, 2, 3, 4, 5, 6, 7, 8])
print(t3.dtype)  # torch.int64
print(f't3的值为：{t3},类型为：{type(t3)}')  # t3的值为：tensor([1, 2, 3, 4, 5, 6, 7, 8]),类型为：<class 'torch.Tensor'>

t4 = torch.tensor(data=[1.1, 0.28, 1.89, 3.98, 6.87])
print(t4.dtype)  # torch.float64
print(
    f't4的值为：{t4},类型为：{type(t4)}')  # t4的值为：tensor([1.1000, 0.2800, 1.8900, 3.9800, 6.8700]),类型为：<class 'torch.Tensor'>

# 使用Tensor来创建张量
t5 = torch.Tensor(5)  # 这里创建的是包含5个元素的张量，而不是将5变成张量
print(t5.dtype)  # torch.float32
print(f't5的值为：{t5},类型为：{type(t5)}')  # t5的值为：tensor([1., 2., 3., 4., 5., 6., 7., 8.]),类型为：<class 'torch.Tensor'>
# 如果想将5变成张量
t6 = torch.Tensor([5])
print(t6.dtype)  # torch.float32
print(f't6的值为：{t6},类型为：{type(t6)}')  # t6的值为：tensor([5.]),类型为：<class 'torch.Tensor'>


# 3.创建指定类型的张量
#3.1使用IntTensor创建张量
# t7 = torch.IntTensor(3)  # 创建一个包含3个元素的张量
t7 = torch.IntTensor([3])  # 创建一个包含3个元素的张量
t8 = torch.IntTensor([3.9999])  # 只保留整数部分
print(f' t7的值为：{t7},类型为：{type(t7)}')  # t7的值为：tensor([1, 2, 3, 4, 5]),类型为：<class 'torch.Tensor'>
print(f' t8的值为：{t8},类型为：{type(t8)}')


#3.2使用FloatTensor创建张量
t9 = torch.FloatTensor([100])  # 创建一个包含100的张量
t10 = torch.FloatTensor([100.9999])  # 创建一个包含100.9999的张量，实际只保留小数部分
print(f' t9的值为：{t9},类型为：{type(t9)}')
print(f' t10的值为：{t10},类型为：{type(t10)}')

#3.3使用DoubleTensor来创建张量
t11 = torch.DoubleTensor([9]) # 实际值为9.0
t12 = torch.DoubleTensor([9.9999]) # 实际值为9.9999
print(f' t11的值为：{t11},类型为：{type(t11)}') # t11的值为：tensor([9.0]),类型为：<class 'torch.Tensor'>
print(f' t12的值为：{t12},类型为：{type(t12)}') # t12的值为：tensor([9.9999]),类型为：<class 'torch.Tensor'>


# 4.创建线性张量
# torch.arange() a是数组range是范围，类似于numpy中的arange() 参数： 开始，结束，步长 （左闭右开）

#4.1 使用arange()创建线性张量
t13 = torch.arange(start=1,end=10,step=1)
print(f't13的值为：{t13}') #tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
#4.2 使用linspace()创建线性张量 torch.linspace() 参数：开始，结束，元素个数 （左右都是闭区间）
t14 = torch.linspace(start=1,end=10,steps=5)
print(f't14的值为：{t14}') #tensor([ 1.0000,  3.2500,  5.5000,  7.7500, 10.0000])

#5.随机张量
# 5.1使用randint来创建随机张量
# 参数：low, high, size (tuple) 左闭右开区间 随机整数张量
t15 = torch.randint(low=1,high=10,size=(5,))  # 创建一个包含5个元素的随机整数张量，元素范围在1到10之间（左闭右开）
print(f't15的值为：{t15}') #tensor([3, 7, 1, 2, 8])  #向量张量
#生成3维随机张量
"""
[
    [
        [3, 1, 4],
        [1, 5, 9],
        [2, 6, 3]
    ],
    [
        [5, 8, 9],
        [4, 2, 6],
        [7, 1, 3]
    ]
]
"""

# 获取随机数种子
print(torch.initial_seed())
#手动设置随机数种子
torch.manual_seed(100) # 手动设置随机数种子后，每次运行结果都相同 相当于固定了随机数生成器的初始状态 底层使用Mersenne Twister算法
t16 = torch.randint(low=1,high=10,size=(2,3,3))
print(f't16的值为：{t16}') #tensor([[[3, 1, 4],


# 5.2使用rand()来创建随机张量
# 参数：size (tuple) 随机张量的形状
t17 = torch.rand(4)#生成包含4个元素的张量，范围在0到1之间（左闭右开）
print(f't17的值为：{t17}')
t18 = torch.rand((2,3,4)) #生成包含指定形状的张量 2*3*4
print(f't18的值为：{t18}')

# 5.3 使用randn()来创建随机张量
#随机生成小数，小数满足标准正态分布，均值为0，标准差为1
t19 = torch.randn(4)  # 创建一个包含4个元素的随机张量，元素满足标准正态分布
print(f't19的值为：{t19}')
t20 = torch.randn((2,3,4)) # 创建一个包含指定形状的随机张量，元素满足标准正态分布
print(f't20的值为：{t20}')

"""
总结：张量是PyTorch中最基本的数据结构，类似于多维数组。可以通过torch.tensor()函数、numpy数组、容器、Tensor来创建张量。
     还可以创建指定类型的张量，如IntTensor、FloatTensor、DoubleTensor。还可以创建线性张量，如arange()、linspace()。
     还可以创建随机张量，如randint()、rand()、randn()。

randint()、rand()、randn()三者之间的区别：
    randint()生成的是随机整数，范围在low到high之间（左闭右开）
    rand()生成的是随机小数，范围在0到1之间（左闭右开）
    randn()生成的是随机小数，满足标准正态分布，均值为0，标准差为1
    
"""

print("=" * 50)

# 6.创建指定值的张量
"""
   PyTorch 常用创建指定值张量的函数总结
=====================================
1. torch.ones()        -> 创建全1张量
2. torch.ones_like()   -> 根据已有张量创建全1
3. torch.zeros()       -> 创建全0张量
4. torch.zeros_like()  -> 根据已有张量创建全0
5. torch.full()        -> 创建指定值张量
6. torch.full_like()   -> 根据已有张量创建指定值
7. torch.eye()        -> 创建单位矩阵（对角线为1）
8. torch.randperm()   -> 随机排列（打乱索引）
=====================================
"""
#先创建一个随机张量方便使用
a = torch.rand(2,3,4)
print(a)

t21 = torch.ones(size=(2,3,4)) # 创建全1张量 参数： shape
print(f'ones--->：{t21}')
t22 = torch.ones_like(a) # 创建全1张量 参数： tensor
print(f'ones_like--->：{t22}')

# 创建全0张量
t23 = torch.zeros(size=(2,3,4)) # 创建全0张量 参数： shape
print(f'zeros--->：{t21}')
t24 = torch.zeros_like(a)       # 创建全0张量 参数： tensor
print(f'zeros_like--->：{t24}')

# 创建指定值的张量 参数： shape, value
t25 = torch.full((2,3,4),2) # 创建指定值的张量 参数： shape, value
print(f'full--->：{t25}')
print(f'full--->：{t25}')
t26 = torch.full_like(a,2) # 创建指定值的张量 参数： tensor, value
print(f'full_like--->：{t26}')


# 创建单位矩阵 参数： n
t27= torch.eye(4,4)  # 创建单位矩阵，参数为行数或列数，返回一个方阵
print(f'eye--->：{t27}')














