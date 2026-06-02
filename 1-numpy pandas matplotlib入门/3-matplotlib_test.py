# -*- coding: utf-8 -*-
# @Time    : 2026/2/11 12:11
# @Author  : WanLi
# @File    : 3-matplotlib_test.py

# 导入随机数生成模块
import random

# 导入matplotlib绘图库
import matplotlib.pyplot as plt


# 注意：通常不需要手动指定backend
# matplotlib会自动选择最适合的backend
# 只有在特殊环境下才需要取消下面的注释

# 如果遇到显示问题，可以尝试取消下面一行的注释
# matplotlib.use('TkAgg')  # 或者 'Qt5Agg', 'MacOSX' 等

# 彻底解决中文显示问题的函数
def set_chinese_font():
    """设置matplotlib支持中文字体"""
    import platform

    system = platform.system()

    if system == "Darwin":  # macOS
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'Arial Unicode MS']
    elif system == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']

    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 调用字体设置函数
set_chinese_font()

# 准备测试数据
# 生成1到12的月份列表
month = [i for i in range(1, 13)]  # 十二个月份

# 生成12个0-100之间的随机销量数据
# 正确的列表推导式：对range(12)进行12次迭代，每次生成一个随机整数
sales = [random.randint(0, 100) for _ in range(12)]  # 十二个随机销量

# 打印数据用于调试验证
print("月份:", month)
print("销量:", sales)

# 使用matplotlib绘制折线图
plt.figure(figsize=(10, 6))  # 设置图表大小为10x6英寸
plt.plot(month, sales,  # x轴为月份，y轴为销量
         marker='o',  # 数据点标记为圆形
         linewidth=2,  # 线条宽度为2
         markersize=8)  # 标记点大小为8

# 设置图表标题和坐标轴标签（现在中文可以正常显示了）
plt.title('月度销售数据分析图表')  # 图表标题
plt.xlabel('月份 (1-12月)')  # x轴标签
plt.ylabel('销售额 (单位: 万元)')  # y轴标签

# 添加网格线，提高可读性
plt.grid(True, alpha=0.3)  # 显示网格，透明度为0.3

# 设置x轴刻度为整数月份
plt.xticks(month)

# 显示最终图表
plt.show()