class Animal(object):
    def eat(self):
        print('动物要吃饭...')

class Cat(Animal):
    def eat(self):
        print("猫吃鱼....")
        

class Dog(Animal):
    pass
    # def eat(self):
    #     print("狗吃肉....")

# 演示方法/函数 重写效果:  当父类的功能不能满足子类的需求.子类会定义一个和父类一样的方法.这就叫重写.
c1 = Cat()
c1.eat() # 猫吃鱼....

d1 = Dog()
d1.eat()#狗吃肉....