class Cat(object):
    # 属性
    # 方法
    def eat(self):
        print(F'self的引用值:{self}')# self的引用值:<__main__.Cat object at 0x000002C00AA49A60>
        print('猫吃鱼...')

# 证明self就是调用者本人???
c1 = Cat()
print(F'c1的引用值:{c1}')# 打印出的是对象的引用: c1的引用值:<__main__.Cat object at 0x000002C00AA49A60>

c1.eat()