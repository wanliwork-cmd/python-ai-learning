#1.按照位置传递参数
def show_info(name,age,sex):
    print(F"我的名字是{name},年龄是{age},性别是{sex}")

# todo 位置传参要求: 参数位置和个数都要一致.
show_info('tom',19,'男')

# 2.关键词传参 --继续用上一个函数
# todo 传参时候除了值还要带上参数名:name='tom'
show_info(age=19,name='jerry',sex='男') # 顺序不一样.但是个数是一致的.
show_info('rose',sex='男',age=20) # 顺序不一样.但是个数是一致的.

