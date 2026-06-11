"""
匿名函数:
语法:
    lambda 参数列表 : 函数体
要求:
    函数体只有一行代码.不能换行.
"""

# 使用def 来定义求和函数.
def sum(a,b):
    return a + b

# 使用lambda 来定义 求和函数
print(lambda a,b : a + b) # <function <lambda> at 0x000001F3DC2B9CA0>
temp_func = lambda a,b : a + b

#使用函数
print(sum(10,20))       #30
print(temp_func(10,20))#30

# todo --------------匿名函数的使用场景-----------------
# todo --------------给函数传递参数的使用使用.(参数就是函数)-----------------
# todo --------------列表容器的sort()函数-----------------

my_list = [1,3,2,4]
my_list.sort()
print(my_list)#  [1, 2, 3, 4]

# 定义大函数
# def compare(d):# 传入字典
#     return d['age'] # 返回字典中的年龄给 sort

my_list2 = [{'name':'tom','age':18},{'name':'aima','age':19},{'name':'lili','age':16}]
# my_list2.sort() # TypeError: '<' not supported between instances of 'dict' and 'dict'
# my_list2.sort(key=compare)
my_list2.sort(key=lambda d:d['age']) # todo  匿名函数的使用场景- 当参数传递.
print(my_list2) # [{'name': 'lili', 'age': 16}, {'name': 'tom', 'age': 18}, {'name': 'aima', 'age': 19}]
















