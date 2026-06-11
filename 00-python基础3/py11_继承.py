class Animal(object):
    def eat(self):
        print('动物要吃饭...')

class Cat(Animal):
    def sleep(self):
        print('猫白天睡觉')


# 子类可以使用父类公共的功能和属性.
c1 = Cat()
c1.eat() # 动物要吃饭...
c1.sleep() # 猫白天睡觉