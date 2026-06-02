# -*- coding: utf-8 -*-
# @Time    : 2026/5/28 12:36
# @Author  : WanLi
# @File    : 03_混淆矩阵_准确率_精确率_召回率_f1分数.py

#1.导包
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1.读取文件获取数据
data =  pd.read_csv("data/breast-cancer-wisconsin.csv")
print(data.shape,data.ndim)

#2.数据预处理  注意：数据中有“?”无效字符，需要先转换为numpy中的nan，然后使用dropna()删除，或者使用fillna()填充
# inplace=True 默认值为false 返回新的数据，True是在原有数据上修改
new_data = data.replace("?", np.nan).dropna() # type: pd.DataFrame
print(new_data.shape,new_data.ndim)
#2.1 分别获取特征和标签
x = new_data.iloc[:,1:-1]  #获取特征
y = new_data.iloc[:,-1]    #获取标签
print(x.shape,x.ndim)
print(y.shape,y.ndim)
#2.2 使用train_test_split切割成4部分
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# 3 特征处理标准化
ss = StandardScaler() #标准化模型对特征数据的处理
new_x_train = ss.fit_transform(X_train) #训练模型并进行标准化处理
new_x_test = ss.transform(X_test) #进行标准化处理
# 3.1 创建模型  逻辑回归模型
lr_model = LogisticRegression() # 创建逻辑回归模型
# 4 模型训练
lr_model.fit(new_x_train, y_train) # 模型训练

# 5 模型评估准确率
y_pred = lr_model.predict(new_x_test) # 模型预测返回标签结果
print(y_pred)
#手动计算
print(y_test.tolist())

#在混淆矩阵中，需要指定正类标签为4，即恶性肿瘤（cancerous tumors），因为默认正类标签为1，而这里标签为2或4，所以需要指定正类标签为4
#参数：pos_label：正类标签
#以下是混淆矩阵的API
print(f'准确率：{accuracy_score(y_test, y_pred)}')
print(f'精确率：{precision_score(y_test, y_pred,pos_label=4)}')
print(f'召回率：{recall_score(y_test, y_pred,pos_label=4)}')
print(f'f1分数：{f1_score(y_test, y_pred,pos_label=4)}')

#打印混淆矩阵
print(confusion_matrix(y_test, y_pred))