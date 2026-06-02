"""
输入输出函数的使用
输入函数
    概念：接收用户从键盘的输入数据
    语法：
        语法1：
            变量名=input("提示词")
输出函数：
    概念：将程序中的数据输出到屏幕上
    语法：
        语法2：
            print(变量名)
            print(<输出项列表>,sep="")
            print(<输出项列表>,sep="分隔符",end="结束符")
"""
# 输入函数的使用
# 语法1：演示输入函数的使用
# TODO 批量注释的快捷键（选中需要注释的代码：CTRL+/）
# number = input("请输入一个数字：")
# print(number)
# # 打印数据的类型
# print(type(number))
#
# # 假如需要将输入的数据转换成int或者float类型, 需要使用int()或者float()函数进行“强转”
# """
# <class 'str'>
# Traceback (most recent call last):
#     number2 = int(number)
#               ^^^^^^^^^^^
# ValueError: invalid literal for int() with base 10: '我爱你'
# """
# number2 = int(number)
# print(number2)
# print(type(number2))

# 语法2：演示输出函数的使用
a = 100
b = 200
c = 300
d = 400

# 打印每个变量的值
# print(a)
# print(b)
# print(c)
# print(d)
print(a, b, c, d) # 一次输出多个变量，默认以空格进行间隔
print(a, b, c, d, sep=",") # 一次输出多个变量，使用逗号进行间隔
print(a, b, c, d, sep=",", end="|") # 一次输出多个变量，使用逗号进行间隔

# 查看python的源代码：ctrl按住不放手+鼠标左键点击函数名
