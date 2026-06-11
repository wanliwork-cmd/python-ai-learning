# class 类名(object):
#     # 属性
#     # 方法

class Cat(object):
    # 属性
    name = 'tom' # 这个变量是类变量

    # 方法
    def eat(self):
        print('猫吃鱼...')

    def sleep(self):
        print('猫白天睡觉...')

# 给类进行实例化---> 创建该类的对象.
# 对象名 = 类名()
c1 = Cat()
c2 = Cat()

# 使用(调用)对象
# 对象名.属性
print(c1.name) # 第一只猫叫tom
print(c2.name) # 第二只猫也叫tom

# 对象名.方法()
c1.eat()
c2.eat()

