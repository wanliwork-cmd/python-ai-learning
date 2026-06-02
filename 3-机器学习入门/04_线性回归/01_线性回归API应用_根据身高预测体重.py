# -*- coding: utf-8 -*-
# @Time    : 2026/5/26 21:19
# @Author  : WanLi
# @File    : 01_线性回归API应用_根据身高预测体重.py

# 1.导包  线性回归模型
from sklearn.linear_model import LinearRegression

# 2.准备数据
# x_train: 身高  y_train: 体重  x_test: 预测数据
x_train = [ [160], [166], [172], [174], [180] ]
y_train = [ 56.3,60.6,65.1,68.5,75 ]
x_test = [ [176] ]

# 3.准备模型
model = LinearRegression()

# 4.模型训练 训练的目的是：找到最优的斜率k和截距b
model.fit(x_train, y_train)
# 打印结果
print("模型最优的斜率:k是", model.coef_)
print("模型最优的截距:b是", model.intercept_)


# 5.模型预测结果
# 手动预测
y_pred_manual = model.coef_[0] * x_test[0][0] + model.intercept_
print(f'手动预测结果:,{y_pred_manual}')
y_pred = model.predict(x_test)
print(f'模型预测结果:,{y_pred}')
