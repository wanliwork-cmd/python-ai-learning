# -*- coding: utf-8 -*-
# @Time    : 2026/6/3 07:19
# @Author  : WanLi
# @File    : 2-2-pandas运用.py


# -*- coding: utf-8 -*-
# Pandas 全面基础 + 实战示例
# Author: Wan
# Description: Pandas常用功能一站式入门

import pandas as pd
import numpy as np

print("========== 1. Series基础 ==========")

s = pd.Series([10, 20, 30, 40])
print("Series:\n", s)

s2 = pd.Series([10, 20, 30], index=["a", "b", "c"])
print("自定义索引Series:\n", s2)

print("\n========== 2. DataFrame基础 ==========")

data = {
    "name": ["Tom", "Jack", "Mary", "Lucy"],
    "age": [18, 20, 22, 24],
    "score": [90, 85, 88, 92]
}

df = pd.DataFrame(data)
print("DataFrame:\n", df)

print("\n========== 3. 读取与保存数据 ==========")

# 读取CSV
# df = pd.read_csv("data.csv")

# 保存CSV
df.to_csv("demo_output.csv", index=False)
print("已保存CSV文件")

print("\n========== 4. 查看数据 ==========")

print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())

print("\n========== 5. 选择数据 ==========")

print(df["name"])          # 列
print(df[["name", "score"]])

print(df.loc[0])           # 按标签
print(df.iloc[1])          # 按位置

print("\n========== 6. 条件过滤 ==========")

print(df[df["age"] > 20])
print(df[(df["age"] > 18) & (df["score"] > 85)])

print("\n========== 7. 添加/修改列 ==========")

df["pass"] = df["score"] >= 90
print(df)

df["bonus"] = df["score"] * 0.1
print(df)

print("\n========== 8. 缺失值处理 ==========")

df.loc[1, "score"] = np.nan
print(df)

print("是否有空值:\n", df.isnull())

df_fill = df.fillna(df["score"].mean())
print("填充后:\n", df_fill)

df_drop = df.dropna()
print("删除空值:\n", df_drop)

print("\n========== 9. 排序 ==========")

print(df.sort_values(by="score", ascending=False))

print("\n========== 10. 分组聚合 ==========")

df_group = pd.DataFrame({
    "class": ["A", "A", "B", "B"],
    "name": ["Tom", "Jack", "Mary", "Lucy"],
    "score": [90, 80, 85, 95]
})

print(df_group.groupby("class")["score"].mean())

print("\n========== 11. 合并数据 ==========")

df1 = pd.DataFrame({
    "id": [1, 2, 3],
    "name": ["A", "B", "C"]
})

df2 = pd.DataFrame({
    "id": [1, 2, 4],
    "score": [88, 92, 75]
})

print(pd.merge(df1, df2, on="id", how="inner"))
print(pd.merge(df1, df2, on="id", how="left"))

print("\n========== 12. 去重 ==========")

df_dup = pd.DataFrame({
    "name": ["A", "A", "B", "C"],
    "score": [90, 90, 80, 70]
})

print(df_dup.drop_duplicates())

print("\n========== 13. 字符串处理 ==========")

df_str = pd.DataFrame({
    "name": [" tom ", "jack", "MARY"]
})

df_str["name"] = df_str["name"].str.strip().str.lower()
print(df_str)

print("\n========== 14. 时间处理 ==========")

df_time = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=5),
    "value": [10, 20, 30, 40, 50]
})

print(df_time)
print(df_time.set_index("date"))

print("\n========== 15. 简单统计 ==========")

print("均值:", df["age"].mean())
print("最大值:", df["score"].max())
print("最小值:", df["score"].min())

print("\n========== 完成 ==========")