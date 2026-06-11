"""
    运算符的分类：
        1：算数运算符
            +、-、*、/、//整除幂、%取模
        2：比较运算符
            >、<、>=、<=、!=、==
        3：逻辑运算符
            and 左右两边两个条件都成立，则成立
            or  左右两边两个条件有一个成立，则成立
            not 一个条件，对结果取反
"""
# 变量的定义
a = 9
b = 4
c = 3.56
d = True
e = "Tom"  # 如果变量值是字符串类型，则一定要加上双引号包含

# 1：算术运算符
print(a + b) # 加， 13
print(a - b) # 减，5
print(a * b) # 乘，36
print(a / b) # 除，2.25
print(a // b) # 整除幂，2
print(a % b) # 取模，1=9-N*4=余数
# print(a + e) # 会报错吗？会：TypeError: unsupported operand type(s) for +: 'int' and 'str'
print(a * e) # 会报错吗？不会，输出：TomTomTomTomTomTomTomTomTom

print("--------------------------------------------------------------")
# 2：比较运算符
print(a > b) # True(真) False(假)，输出True
print(a < b) # False
print(a == b) # False
print(a != b) # True

print("--------------------------------------------------------------")
# 3：逻辑运算符
print(a > b and a > 100) # 左右两边两个条件都成立，则成立, 输出False
print(a > b and b < 10) # True

print(a > b or a > 100) # 左右两边两个条件有一个成立，则成立，输出True
print(a < b or a > 100) # False

print(not a > b) # 一个条件，对结果取反，输出False

# 三目运算符，条件判断的时候才用到（if... else...）
