"""
全局变量: 定义在py文件(模块)中,在整个模块都可以使用这个变量.在另一模块也能使用这个变量.
局部变量: 定义在函数内部.只能在函数中使用.函数执行完毕后局部变量就会立刻消失.
"""

g_number = 100

def show_number():
    number1 = 200 # 局部变量
    print(f"局部变量number1的值是{number1}")
    print(f"全局变量g_number的值是{g_number}")

# 局部位置使用全局没有问题.
show_number()

#全局位置使用局部变量
# print(number1) # NameError: name 'number1' is not defined

# 如果局部和全局变量名一模一样.这个时候有什么现象.
age = 18
def show_age():
    age = 20
    print(f'我的年龄是{age}')# 如果局部和全局变量名一模一样那么在函数内优先使用局部变量.

# 查看show_age的结果
show_age() # 我的年龄是20


# !!!在局部位置修改全局变量!!!
g_number2 = 500
def change_number():
    # todo 如果想在这里修改全局变量,需要加上global关键词来表示这个变量用的是全局的.
    global g_number2
    g_number2 = 1000 # 这里是局部位置.也就是你在这里定义了一个局部变量而已.
    print(f"修改后的全局变量值是:{g_number2}") # 修改后的全局变量值是:1000

#测试修改函数的效果:
change_number()
print(f"修改后的全局变量值是:{g_number2}") # 修改后的全局变量值是:500/1000