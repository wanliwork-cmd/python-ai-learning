"""
模块的导入方式
    方式1：import 模块名
    方式2：from 模块名 import 函数名
"""

# 方式1：import 模块名(python内置的模块非常非常多，这里仅仅是使用random演示而已，在实际开发中，可以自定义模块)
import random

# 随机生成1-10之间的随机整数
print(random.randint(1, 10))

# 返回0-1之间的随机浮点数
print(random.random())

# 方式2：from 模块名 import 函数名
from random import randint,random
print(randint(1, 10))
print(random())
