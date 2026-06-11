# -*- coding: utf-8 -*-
# @Time    : 2026/6/3 07:50
# @Author  : WanLi
# @File    : 02_张量元素的类型转换.py

# -*- coding: utf-8 -*-
"""
PyTorch 张量元素类型转换完整总结
========================================
包含：
1. type()（查看）
2. dtype（核心属性）
3. 快捷转换方法
4. to() 通用转换
5. 常见dtype对照
========================================
"""

import torch

print("\n================ 1. 创建基础张量 ================")

data = torch.tensor([1, 2, 3])
print("data:", data)
print("dtype:", data.dtype)

print("\n================ 2. type()（仅查看，不推荐转换） ================")

print("data.type():", data.type())
# 例：torch.LongTensor（旧写法，仅用于查看）

print("\n================ 3. dtype（推荐核心属性） ================")

print("data.dtype:", data.dtype)
# 现代标准：torch.int64 / float32 等

print("\n================ 4. 快捷类型转换（最常用） ================")

print("float():", data.float())     # float32
print("double():", data.double())   # float64
print("half():", data.half())       # float16

print("int():", data.int())         # int32
print("long():", data.long())       # int64
print("short():", data.short())     # int16

print("\n================ 5. to() 通用转换（推荐最强方式） ================")

print("to(float32):", data.to(torch.float32))
print("to(float64):", data.to(torch.float64))
print("to(int64):", data.to(torch.int64))

print("\n================ 6. dtype + device 一起转换 ================")

x = torch.tensor([1, 2, 3])

x2 = x.to(dtype=torch.float32, device="cpu")
print("x2:", x2)
print("dtype:", x2.dtype)
print("device:", x2.device)

print("\n================ 7. 常见dtype对照表 ================")

print("""
torch.float32  -> 默认浮点（训练用）
torch.float64  -> 高精度（慢）
torch.float16  -> 半精度（GPU省显存）
torch.int32    -> 普通整数
torch.int64    -> 索引 / label（最常用）
torch.int16    -> 小整数
torch.bool     -> mask / 逻辑判断
""")

print("\n================ 8. 核心总结（重点） ================")

print("""
✔ dtype = 数据类型（现代标准）
✔ to()   = 类型 + 设备（最强推荐）
✔ float/int/long = 快捷写法
✔ type() = 仅查看旧类型（不用于开发）

🔥 实战规则：
- 训练数据：float32
- 标签/索引：int64
- GPU训练：优先 float16（half）
""")

print("\n================ 完成 =================")