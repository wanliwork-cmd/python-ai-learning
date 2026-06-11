"""
    变量的概念：程序中被命名的存储单元，用于存储数据，简单来说，变量就是一个带名字的容器，可以存放数据
    变量的语法：
        变量名 = 变量值
    变量命名的规范：
        1：变量名只能是字母、数字、下划线
        2：变量的首字母不能是数字
        3：严格区分大小写
        4：不能使用内置的名字（import、public...,如何判断：可以写上以后看一下语法是否报错）
"""
# 变量的定义
a = 9
b = 4
c = 3.56
d = True
e = "Tom"  # 如果变量值是字符串类型，则一定要加上双引号包含

print(type(a)) # Ctrl+D 复制该行代码, 输出：<class 'int'>
print(type(b)) # <class 'int'>
print(type(c)) # <class 'float'>
print(type(d)) # <class 'bool'>
print(type(e)) # <class 'str'>

# 变量的操作
print(a + b) # 加， 13
print(a - b) # 减，5
print(a * b) # 乘，36
print(a / b) # 除，2.25
print(a // b) # 整除幂，2
print(a % b) # 取模，1=9-N*4=余数
# print(a + e) # 会报错吗？会：TypeError: unsupported operand type(s) for +: 'int' and 'str'
print(a * e) # 会报错吗？不会，输出：TomTomTomTomTomTomTomTomTom