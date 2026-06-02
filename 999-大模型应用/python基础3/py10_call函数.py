# 加算器类
class Adder(object):
    def __init__(self, value=0):
        self.data = value

    # 调用对象的默认函数()
    def __call__(self, x):
        print("call执行.....")
        return self.data + x

# 创建计算器对象
add = Adder(10)

#使用计算器
print(add(1))  # 1
print(add(2))  # 2

