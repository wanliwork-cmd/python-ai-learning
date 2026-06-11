# 可变参数又叫不定长参数,它是在参数的个数不确定的情况下使用的.
"""
*args   :  单值不定长容器 -->  元组
**kwargs:  键值对不定长  -->  字典
"""

# 定义函数,求任意个整数的和.
def get_sum(*args):
    # * 表示不定长符号,args才是参数名称.
    print(args) # (1, 2, 3)
    print(type(args)) # <class 'tuple'>
    sum = 0
    for i in args:
        sum += i # sum = sum + i
    print(f"求和的结果是{sum}")

#调用函数
get_sum(1,2,3)



# **kwargs:  键值对不定长  -->  字典
def show_info(**kwargs):
    print(kwargs) # {'name': 'tom', 'age': 19}
    print(type(kwargs))# <class 'dict'>
    print("我的个人信息有:")
    for k in kwargs:
        print(f"\t{k}:{kwargs[k]}")

show_info(name='tom',age=19) # 调用必须给参数名==如果调用默认值参数.













