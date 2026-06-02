"""
需求：人与机器完成猜拳游戏
1：玩家输入自己的猜拳（1：剪刀，2：石头，3：布）
2：机器随机出拳（1：剪刀，2：石头，3：布）
3：判断玩家和机器的猜拳结果，判断是否赢了
"""
import random

# 玩家输入自己的猜拳（input）
player = input("请输入自己的猜拳（1：剪刀，2：石头，3：布）：")
print(player)
# print(type(player))

# 机器随机出拳
computer = str(random.randint(1,3))
print(computer)
# print(type(computer))

if player == computer: # if "1" == 1
    print("平局")
elif player == "1" and computer == "2":
    print("玩家猜拳剪刀，机器猜拳石头，玩家输了")
elif player == "1" and computer == "3":
    print("玩家猜拳剪刀，机器猜拳布，玩家赢了")
elif player == "2" and computer == "1":
    print("玩家猜拳石头，机器猜拳剪刀，玩家赢了")
elif player == "2" and computer == "3":
    print("玩家猜拳石头，机器猜拳布，玩家输了")
elif player == "3" and computer == "1":
    print("玩家猜拳布，机器猜拳剪刀，玩家输了")
elif player == "3" and computer == "2":
    print("玩家猜拳布，机器猜拳石头，玩家赢了")
else:
    print("输入的数字有误")
