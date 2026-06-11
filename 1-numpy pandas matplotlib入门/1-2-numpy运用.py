# -*- coding: utf-8 -*-
# @Time    : 2026/6/3 07:13
# @Author  : WanLi
# @File    : 1-1-numpy运用.py


# -*- coding: utf-8 -*-
# @Time : 2026/06/03
# @Author : Wan
# @File : numpy_common_methods.py

"""
NumPy 常用方法大全（适合机器学习 / 深度学习入门）

学习顺序建议：

1. 创建数组
2. 数组属性
3. 修改形状
4. 索引切片
5. 数组运算
6. 统计方法
7. 随机数
8. 数学函数
9. 矩阵运算
"""

import numpy as np

print("=" * 50)
print("1. 创建数组")
print("=" * 50)

# -------------------------------
# 1.1 array()
# -------------------------------
a = np.array([1, 2, 3])
print("array()：", a)

# 二维数组
b = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("二维数组：")
print(b)

# -------------------------------
# 1.2 zeros()
# 创建全0数组
# -------------------------------
print("\nzeros()")
print(np.zeros((2, 3)))

# -------------------------------
# 1.3 zeros_like()
# 按已有数组形状创建全0数组
# -------------------------------
print("\nzeros_like()")
print(np.zeros_like(b))

# -------------------------------
# 1.4 ones()
# 创建全1数组
# -------------------------------
print("\nones()")
print(np.ones((2, 3)))

# -------------------------------
# 1.5 ones_like()
# 按已有数组形状创建全1数组
# -------------------------------
print("\nones_like()")
print(np.ones_like(b))

# -------------------------------
# 1.6 full()  参数：(shape(行, 列), value（ 填充值 ）)
# 全部填指定值
# -------------------------------
print("\nfull()")
print(np.full((2, 3), 999))

# -------------------------------
# 1.7 full_like()
# 按已有数组形状填充值
# -------------------------------
print("\nfull_like()")
print(np.full_like(b, 888))

# -------------------------------
# 1.8 empty()
# 创建空数组（随机内存值）
# -------------------------------
print("\nempty()")
print(np.empty((2, 2)))

# -------------------------------
# 1.9 arange()
# 类似 Python range
# -------------------------------
print("\narange()")

print(np.arange(10))
print(np.arange(1, 10))
print(np.arange(1, 10, 2))

# -------------------------------
# 1.10 linspace()
# 等间距生成数据
# -------------------------------
print("\nlinspace()")
print(np.linspace(0, 10, 5))

# -------------------------------
# 1.11 eye()
# 单位矩阵
# -------------------------------
print("\neye()")
print(np.eye(3))


print("\n" + "=" * 50)
print("2. 数组属性")
print("=" * 50)

arr = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# shape：形状（重点）
print("shape：", arr.shape)

# ndim：维度
print("ndim：", arr.ndim)

# size：元素总数
print("size：", arr.size)

# dtype：数据类型
print("dtype：", arr.dtype)


print("\n" + "=" * 50)
print("3. 修改形状")
print("=" * 50)

x = np.arange(12)

print("原数组：")
print(x)

# reshape()
print("\nreshape(3,4)")
print(x.reshape(3, 4))

# 自动计算
print("\nreshape(-1,3)")
print(x.reshape(-1, 3))

# flatten()
print("\nflatten()")
print(x.reshape(3, 4).flatten())

# 转置
print("\ntranspose / T")
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(matrix.T)


print("\n" + "=" * 50)
print("4. 索引与切片")
print("=" * 50)

arr = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# 取元素
print(arr[0])

# 第一行第二列
print(arr[0, 1])

# 切片
print(arr[:, 1])

# 第一列
print(arr[:, 0])

# 第二列以后
print(arr[:, 1:])


print("\n" + "=" * 50)
print("5. 数组运算")
print("=" * 50)

a = np.array([1, 2, 3])

print("a + 1")
print(a + 1)

print("a - 1")
print(a - 1)

print("a * 2")
print(a * 2)

print("a / 2")
print(a / 2)

print("平方")
print(a ** 2)

print("开方")
print(np.sqrt(a))


print("\n" + "=" * 50)
print("6. 统计方法")
print("=" * 50)

nums = np.array([1, 2, 3, 4, 5])

print("max：", np.max(nums))
print("min：", np.min(nums))
print("sum：", np.sum(nums))
print("mean：", np.mean(nums))
print("median：", np.median(nums))

# 方差
print("var：", np.var(nums))

# 标准差
print("std：", np.std(nums))


print("\n" + "=" * 50)
print("7. 随机数（重点）")
print("=" * 50)

# rand：0~1随机数
print("rand()")
print(np.random.rand(3))

# randint：随机整数
print("\nrandint()")
print(np.random.randint(1, 10))

# randn：标准正态分布
print("\nrandn()")
print(np.random.randn(5))


print("\n" + "=" * 50)
print("8. 数学函数")
print("=" * 50)

arr = np.array([1, 2, 3])

print("abs()")
print(np.abs([-1, -2, 3]))

print("sqrt()")
print(np.sqrt([1, 4, 9]))

print("exp()")
print(np.exp([1, 2]))

print("log()")
print(np.log([1, 2, 3]))


print("\n" + "=" * 50)
print("9. 矩阵运算（重点）")
print("=" * 50)

a = np.array([
    [1, 2],
    [3, 4]
])

b = np.array([
    [5, 6],
    [7, 8]
])

# 矩阵乘法
print("矩阵乘法 @")
print(a @ b)

# dot
print("\ndot()")
print(np.dot(a, b))


print("\n" + "=" * 50)
print("学习建议")
print("=" * 50)

print("""
第一阶段必须掌握：

1. np.array()
2. np.zeros() / zeros_like()
3. np.ones() / ones_like()
4. np.full() / full_like()
5. np.arange()
6. np.linspace()
7. shape / ndim / dtype
8. reshape()
9. 索引切片
10. mean() / sum()
11. np.random.randn()

""")