# 1.导包
import pandas as pd

# 忽略所有警告
import warnings
warnings.filterwarnings('ignore')

# 2.读取文件
df = pd.read_csv('data/students.csv')
print(df)
print('---------------------------------------')
# df.info()
# 3.fillna()填充数据
# 方式1: 直接填充0  这种情况容易造成数据异常,本来是成人,你填充0,整体年龄下降了
new_df1 = df.fillna(0)
print(new_df1)
print('---------------------------------------')
# 方式2: 填充平均年龄
new_df2 = df.fillna(df['age'].mean())
print(new_df2)
print('---------------------------------------')
# 方式3: 填充当前缺失值的前一个数据
new_df3 = df.fillna(method='ffill')
print(new_df3)
print('---------------------------------------')
# 方式4: 填充当前缺失值的后一个数据
new_df4 = df.fillna(method='bfill')
print(new_df4)
