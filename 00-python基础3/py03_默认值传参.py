# 定义函数的时候可以给参数赋一个默认值,此时调用者可以不给该参数传递值.也可以传递值覆盖默认值.
def show_info(name,age,sex="男"):
    print(F"我的名字是{name},年龄是{age},性别是{sex}")

# 调用方式
show_info('tom',19)# 我的名字是tom,年龄是19,性别是男
show_info('rose',18,'女')#我 的名字是rose,年龄是18,性别是女
