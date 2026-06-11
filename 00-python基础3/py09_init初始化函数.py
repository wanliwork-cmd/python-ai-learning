class Person(object):

    # 这是一个魔法函数: (不需要你调用.系统会在合适的时候自动调用.[你得定义])
    def __init__(self,name,age):
        self.name = name
        self.age = age

# 测试我们的类有没有属性..
# p1 = Person() # TypeError: __init__() missing 2 required positional arguments: 'name' and 'age'
p2 = Person('小明',19)
print(p2.name)
print(p2.age)
#-------------------------------------
p3 = Person('小强',21)
print(p3.name)
print(p3.age)