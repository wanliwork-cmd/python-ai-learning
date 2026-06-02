"""
需求：写一个网吧登录的程序，年满18岁可以上网，不满18岁不可以上网
if条件结构：
    执行的代码内容
"""
# # 定义一个年龄变量
# age = 16
# if age>=18: # if：如果条件满足，执行if后的代码
#     print("可以上网")
# else:   # 否则
#     print("不可以上网")

# print(type(age>=18))
# 定义一个年龄变量
age = 5
if age<3: # if：如果条件满足，执行if后的代码
    print("学前班")
# elif age>=3 and age<6: # 否则如果
elif age<6: # 否则如果
    print("幼儿园")
# elif age>=6 and age<12:
elif age<12:
    print("小学生")
# elif age>=12 and age<18:
elif age<18:
    print("中学生")
else:
    print("成年人")

# 三目运算符
x = 10
y = 20
# 标准写法
if x>y:
    print(x)
else:
    print(y)

# 三目运算符的简化写法
# 表达式： value_if_true if condition else value_if_false
# 这个表达式会先判断condition，如果为True，则返回value_if_true， 否则返回value_if_false
max_value = x if x > y else y

print(max_value)