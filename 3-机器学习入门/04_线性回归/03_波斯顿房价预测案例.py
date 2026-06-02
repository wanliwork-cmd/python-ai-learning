# -*- coding: utf-8 -*-
# @Time    : 2026/5/28 07:52
# @Author  : WanLi
# @File    : 03_波斯顿房价预测案例.py

#1.导包
import pandas as pd
import numpy as np
from fontTools.cu2qu.ufo import font_to_quadratic
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression

#2.准备数据
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
#2.1数据切割
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

#2.2特征的标准化数据
ss = StandardScaler()
new_x_train = ss.fit_transform(X_train)  # 训练集用fit_transform()训练并转换
new_x_test = ss.transform(X_test)  #测试集用transform()转换，因为前面已经用fit_transform()训练过，所以这里用transform()即可
#3.准备模型
# invscaling: 动态调整学习率  constant：固定学习率
model = SGDRegressor(loss='squared_error', learning_rate='invscaling',eta0=0.01)  #梯度下降模型
#model = LinearRegression()  #线性回归模型
#4.模型训练
model.fit(new_x_train, y_train)
print(f'训练后k参数权重：{model.coef_}')  # 模型的系数
print(f'训练后b参数偏置：{model.intercept_}')  # 模型的截距

#5.模型预测
y_pred = model.predict(new_x_test)
#6.模型评估  误差越小越好
print('====================================================')
print(f'平均绝对误差：{mean_absolute_error(y_test,y_pred)}')
print(f'均方误差：{mean_squared_error(y_test,y_pred)}')
print(f'均方根误差：{root_mean_squared_error(y_test,y_pred)}')