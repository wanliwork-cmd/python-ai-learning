# 1、定义一个Person类
class Person(object):
    pass# 略过... 不让语法报错.临时撑起类的身体.

# 2、实例化Person类，生成p1对象
p1 = Person()

# 3、为p1对象添加属性
p1.name = '老王'
p1.age = 23
p1.address = '北京市昌平区'

# 4.查看类的属性////???
print(p1.name)# 老王
print(p1.age)# 23
print(p1.address)# 北京市昌平区
# print(p1.id) # AttributeError: 'Person' object has no attribute 'id'
