# -*- coding: utf-8 -*-
# @Time    : 2026/2/ 12:08
# @Author  : WanLi
# @File    : 1-numpy_test.py


# 导包
import numpy as np

# 创建numpy数组
my_list = [[10,20],[30,40]]
print(my_list)
print(type(my_list))

# 将列表转换为numpy数组
my_array = np.array(my_list)
print(my_array)
print(type(my_array))

# 数组的运算
print(my_array + 2)
print(my_array - 2)
print(my_array * 2)
print(my_array / 2)

print("====="*20)
print(np.max(my_array))
print(np.min(my_array))
print(np.mean(my_array))
print(np.sum(my_array))