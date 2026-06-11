"""
if的嵌套语法
需求：进火车站先判断是否有车票，有车票在判断是否携带了违禁品

if 有车票:
    if 携带了违禁品:
        print("不可以进火车站")
    else:
        print("可以进火车站")
else:
    print("不可以进火车站")
"""

# 定义一个变量，判断是否有车票（True表示有车票，False表示没有车票）
has_ticket = True

# 定义一个变量，判断是否携带了违禁品（True表示携带了违禁品，False表示没有携带违禁品）
has_prohibited_item = False

# if has_ticket == True: # 判断是否有车票（有）
if has_ticket: # if 的条件满足（返回值是布尔类型：True或者False），执行if后的代码
    print("已经购买了车票")
    if has_prohibited_item: # 判断是否携带了违禁品（有）
        print("不可以进入火车站")
    else: # 否则else
        print("可以进入火车站")
else: # 否则else
    print("没有购买车票")

"""
可不可以无限嵌套？
    可以无限嵌套，但是不建议超过3层（可读性太差）
"""

