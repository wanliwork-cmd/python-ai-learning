class Car(object):
    def __init__(self, brand, model, color):
        self.brand = brand # 品牌
        self.model = model # 款式
        self.color = color # 颜色

    def run(self):
        print('i can run')

# 定义汽油车
class GasolineCar(Car):
    def run(self):
        print('i can run with gasoline')

# 电动汽车
class ElectricCar(Car):
    def __init__(self, brand, model, color, battery):
        super().__init__(brand, model, color)
        # 电池属性
        self.battery = battery

    def run(self):
        print(f'i can run with electric，i have a {self.battery} + " kWh battery"')

# 使用电动汽车
e1 = ElectricCar('MI','越野车','绿色','10000')
print(e1.brand)
print(e1.model)
print(e1.color)
print(e1.battery)
e1.run()


