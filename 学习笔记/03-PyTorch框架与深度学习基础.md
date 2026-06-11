# 03 - PyTorch框架与深度学习基础

> 日期：2025年11月7日 ~ 11月9日  
> 核心内容：PyTorch框架、张量操作、自动微分、梯度下降

---

## 一、深度学习简介

深度学习是机器学习的一个子领域，核心思想是通过多层神经网络从数据中自动学习特征表示。

**深度学习 vs 传统机器学习**：
- 传统ML：需要人工设计特征（Feature Engineering）
- 深度学习：自动从原始数据中学习特征（Feature Learning）

**典型应用场景**：图像识别、自然语言处理、语音识别、推荐系统、自动驾驶等。

---

## 二、PyTorch 基础

### 2.1 什么是PyTorch

PyTorch是由Facebook（Meta）开发的深度学习框架，特点：
- **动态计算图**：代码即模型，调试方便
- **Pythonic**：与Python生态无缝集成
- **GPU加速**：自动支持CUDA加速

### 2.2 张量（Tensor）——PyTorch的核心数据结构

张量是PyTorch中的基本数据类型，可以理解为多维数组（类似NumPy的ndarray，但支持GPU计算）。

#### 创建张量

```python
import torch

# 从数据创建
t = torch.tensor([1, 2, 3])

# 创建随机张量
t = torch.randn(3, 4)       # 3行4列，标准正态分布
t = torch.rand(3, 4)         # 均匀分布 [0, 1)
t = torch.randint(0, 10, (3, 4))  # 随机整数

# 创建线性张量
t = torch.arange(0, 10, step=2)   # [0, 2, 4, 6, 8]
t = torch.linspace(0, 1, 5)       # [0, 0.25, 0.5, 0.75, 1.0]

# 创建指定值张量
t = torch.zeros(3, 4)       # 全0
t = torch.ones(3, 4)        # 全1
t = torch.full((3, 4), 5.0) # 全5
t = torch.eye(3)            # 单位矩阵
```

#### 张量属性

```python
t = torch.randn(3, 4)
print(t.shape)    # 形状：torch.Size([3, 4])
print(t.dtype)    # 数据类型：torch.float32
print(t.device)   # 所在设备：cpu
print(t.ndim)     # 维度数：2
```

#### 张量类型转换

```python
# 元素类型转换
t = t.float()     # 转为float32
t = t.double()    # 转为float64
t = t.int()       # 转为int32
t = t.to(torch.float16)  # 通用转换方法

# 张量形状操作
t = torch.arange(12)
t = t.reshape(3, 4)     # 重塑为3行4列
t = t.view(3, 4)        # 类似reshape
t = t.unsqueeze(0)      # 增加维度：(1, 3, 4)
t = t.squeeze()          # 去除维度为1的维度
t = t.permute(1, 0)     # 转置
t = t.transpose(0, 1)   # 交换维度
```

#### 张量拼接

```python
a = torch.ones(2, 3)
b = torch.zeros(2, 3)

# 沿维度0拼接（纵向）
c = torch.cat([a, b], dim=0)  # shape: (4, 3)

# 沿维度1拼接（横向）
c = torch.cat([a, b], dim=1)  # shape: (2, 6)

# 堆叠（新增维度）
c = torch.stack([a, b], dim=0)  # shape: (2, 2, 3)
```

#### 张量索引操作

```python
t = torch.arange(12).reshape(3, 4)

# 基本索引
t[0]          # 第0行
t[0, 1]       # 第0行第1列
t[0:2]        # 第0到1行（切片）

# 高级索引
t[t > 5]      # 布尔索引：大于5的元素
t[:, [0, 2]]  # 选取第0列和第2列
```

---

## 三、自动微分模块（Autograd）

自动微分是PyTorch的核心功能，用于神经网络的反向传播（计算梯度）。

### 3.1 核心概念

- **`requires_grad=True`**：标记张量需要计算梯度
- **`backward()`**：触发反向传播，计算所有梯度
- **`.grad`**：存储计算出的梯度值
- **计算图**：PyTorch自动记录所有操作，构建有向无环图（DAG）

### 3.2 单次梯度下降示例

```python
import torch

# 1. 初始化权重（requires_grad=True是关键开关）
w = torch.tensor([20, 30], requires_grad=True, dtype=torch.float32)

# 2. 定义损失函数：loss = 2*w²
loss = 2 * w ** 2
# loss = [2*20², 2*30²] = [800, 1800]

# 3. 反向传播（先sum转标量，再backward）
loss.sum().backward()
# 梯度计算：∂L/∂w = 4*w → [80, 120]
print(w.grad)  # tensor([80., 120.])

# 4. 更新权重：W_new = W_old - lr * grad
lr = 0.01
w.data = w.data - lr * w.grad
# w_new = [20-0.8, 30-1.2] = [19.2, 28.8]
print(w.data)  # tensor([19.2, 28.8])
```

### 3.3 多轮梯度下降

```python
w = torch.tensor([20, 30], requires_grad=True, dtype=torch.float32)
lr = 0.01

for epoch in range(100):
    # 每次迭代必须重新计算损失（计算图是一次性的）
    loss = 2 * w ** 2
    
    # 梯度清零（PyTorch默认会累加梯度！）
    if w.grad is not None:
        w.grad.zero_()
    
    # 反向传播
    loss.sum().backward()
    
    # 更新权重（使用.data避免修改计算图）
    w.data = w.data - lr * w.grad
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss={loss.sum().item():.4f}, w={w.data}")
```

**重要注意事项**：
1. 每次迭代都要**重新计算**损失函数（计算图用完即释放）
2. 每次反向传播前必须**梯度清零**（否则梯度会累加）
3. 使用 `w.data` 更新权重，避免破坏计算图

### 3.4 前向传播与反向传播

```
前向传播：输入数据 → 经过各层计算 → 得到预测值 → 计算损失
反向传播：损失 → 通过链式法则逐层计算梯度 → 更新参数
```

**链式法则**：复合函数的导数 = 各层导数的乘积

---

## 四、PyTorch 设备管理

```python
# 检查是否有GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前设备：{device}")

# 将数据和模型移到GPU
model = MyModel().to(device)
data = data.to(device)
```

**常见错误**：`Expected all tensors to be on the same device`
- 原因：参与运算的张量不在同一设备上（CPU vs GPU）
- 解决：所有数据和模型必须 `.to(device)` 到同一设备

---

## 五、关键概念总结

| 概念 | 说明 |
|------|------|
| 张量 Tensor | PyTorch的基本数据结构，支持GPU加速 |
| 计算图 | 自动记录操作，用于反向传播 |
| requires_grad | 标记是否需要梯度计算 |
| backward() | 触发反向传播，计算梯度 |
| 梯度累加 | PyTorch默认累加梯度，需手动清零 |
| 学习率 lr | 控制参数更新步长，过大发散，过小收敛慢 |
| 损失函数 | 衡量预测值与真实值的差距 |
