# -*- coding: utf-8 -*-
# @Time    : 2026/5/27 20:49
# @Author  : WanLi
# @File    : 02_获取波斯顿房价预测数据.py
#from sklearn.datasets import load_boston

import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
print(f'特征：{data}')
print(f'标签：{target}')
