# AI 全栈工程师学习路径与实践项目集

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.12-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Scikit--Learn-1.7-F7931E?logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Transformers-HuggingFace-yellow?logo=huggingface&logoColor=white" alt="Transformers">
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-green" alt="Ollama">
  <img src="https://img.shields.io/badge/Streamlit-WebApp-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
</p>

> 本仓库系统性地记录了从 Python 基础到深度学习、大模型应用的 **AI 全栈工程师** 完整学习路径，涵盖理论学习、代码实践与工程化落地，并附带完整的 **投满分文本分类** 实战项目。

---

## 📋 目录

- [项目概述](#项目概述)
- [技术栈](#技术栈)
- [仓库结构](#仓库结构)
- [核心模块详解](#核心模块详解)
  - [Python 基础与数据科学](#1-python-基础与数据科学)
  - [机器学习](#2-机器学习)
  - [深度学习与 PyTorch](#3-深度学习与-pytorch)
  - [自然语言处理](#4-自然语言处理-nlp)
  - [LLM 大模型应用](#5-llm-大模型应用)
  - [投满分文本分类实战项目](#6-投满分文本分类实战项目tmf)
- [学习笔记体系](#学习笔记体系)
- [面试资源](#面试资源)
- [环境搭建](#环境搭建)
- [快速开始](#快速开始)
- [License](#license)

---

## 项目概述

本仓库是一个面向 **AI/ML 工程师** 岗位的系统性学习资源库，设计思路遵循：

```
Python 基础 → 数据科学工具链 → 经典机器学习 → 深度学习 → NLP → 大模型应用 → 工程化落地
```

**核心特色：**

| 特色 | 说明 |
|------|------|
| 🎯 目标导向 | 以 AI 工程师岗位能力模型为导向组织内容 |
| 📝 理论+实践 | 每个知识模块均配套学习笔记与可运行代码 |
| 🏗️ 工程化思维 | 核心项目包含完整的训练/推理/部署/API服务化链路 |
| 📊 渐进式难度 | 从基础语法到大模型蒸馏，难度梯度平滑 |
| 🔧 工具链完整 | 覆盖 NumPy、Pandas、Scikit-learn、PyTorch、HuggingFace、Ollama 等主流工具 |

---

## 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| **编程语言** | Python 3.10+ | 主力开发语言 |
| **数据科学** | NumPy, Pandas, Matplotlib, OpenPyXL | 数据处理与可视化 |
| **机器学习** | Scikit-learn | 经典 ML 算法实现 |
| **深度学习** | PyTorch 2.x | 神经网络构建与训练 |
| **NLP** | FastText, HuggingFace Transformers | 文本分类与预训练模型微调 |
| **大模型** | Ollama, Qwen, DeepSeek, 阿里百炼 | LLM 推理与应用开发 |
| **模型优化** | 知识蒸馏, 模型剪枝, 量化 | 模型压缩与部署优化 |
| **Web 框架** | Streamlit, Flask | 可视化界面与 API 服务化 |

---

## 仓库结构

```
00_python_code/
│
├── 00-python基础1/          # Python 语法基础（变量、运算符、数据类型）
├── 00-python基础2/          # 控制流与模块（条件、循环、模块导入）
├── 00-python基础3/          # 面向对象编程（类、继承、多态）
│
├── 1-numpy pandas matplotlib入门/  # 数据科学三剑客入门
├── 2-pandas进阶/                    # Pandas 高级操作（缺失值、分组聚合、透视表）
│
├── 3-机器学习入门/          # Scikit-learn 经典算法
│   ├── 01_sklearn/          #   sklearn 基础
│   ├── 02_knn算法/          #   KNN 分类与回归
│   ├── 03_归一化和标准化/    #   特征工程
│   ├── 04_线性回归/          #   线性回归（身高预测、房价预测）
│   ├── 05_逻辑回归/          #   逻辑回归与评估指标
│   └── 06_聚类算法/          #   K-Means 聚类
│
├── 4-深度学习/01-PyTorch框架/  # PyTorch 框架系统学习
│
├── 5-NLP自然语言处理基础/    # NLP 核心技术
│   ├── 01_文本预处理/        #   分词、向量化、文本分析
│   ├── 02_RNN及其变体/       #   RNN / LSTM / GRU / 注意力机制
│   ├── 03_Transformer/       #   Transformer 架构完整实现
│   └── 04_迁移学习/          #   FastText 与预训练模型
│
├── 5-投满分文本分类任务项目/TMF/  # ⭐ 核心实战项目
│   ├── 01-rf/                #   随机森林基线方案
│   ├── 02-fasttext/          #   FastText 高效方案
│   ├── 03-bert/              #   BERT 微调方案
│   ├── 04-LLM/               #   大模型推理方案
│   ├── 05-distill/           #   知识蒸馏优化
│   └── 06-prune/             #   模型剪枝压缩
│
├── 00-ollama入门/            # Ollama 本地大模型部署
├── 0-0-学习笔记/             # 21 篇系统学习笔记
├── 0-0-AI面试题/             # 精选 200 道 AI 面试题
└── 0-0-面试模板以及注意事项/   # 简历模板与面试技巧
```

---

## 核心模块详解

### 1. Python 基础与数据科学

> 夯实编程功底，掌握数据科学核心工具链。

| 模块 | 内容 | 关键知识点 |
|------|------|-----------|
| Python 基础 1-3 | 语法基础 → 控制流 → OOP | 变量、运算符、条件、循环、函数、类、继承、多态 |
| NumPy | 多维数组与数值计算 | 张量创建、广播机制、索引切片、数值运算 |
| Pandas | 表格数据处理 | 缺失值处理、分组聚合、透视表、RFM 模型 |
| Matplotlib | 数据可视化 | 折线图、柱状图、散点图、多图布局 |

### 2. 机器学习

> 基于 Scikit-learn 框架，系统掌握经典 ML 算法及评估体系。

- **KNN 算法**：分类任务与回归任务的完整实现
- **特征工程**：Min-Max 归一化与 Z-Score 标准化的原理及适用场景
- **线性回归**：基于身高预测体重、波士顿房价预测等实战案例
- **逻辑回归**：癌症预测案例，配合混淆矩阵、精确率、召回率、F1-Score 等评估指标
- **K-Means 聚类**：无监督学习聚类分析

### 3. 深度学习与 PyTorch

> 从张量操作到自动微分，完整掌握 PyTorch 深度学习框架。

```
张量创建与操作 → 数值运算 → 自动微分 → 前向/反向传播 → 线性回归模型实战
```

核心知识点：

- 张量的创建、类型转换、索引、形状变换、拼接与堆叠
- `autograd` 自动微分机制与梯度计算
- 前向传播与反向传播的完整流程
- 手动实现线性回归模型

### 4. 自然语言处理（NLP）

> 覆盖 NLP 核心技术栈，从传统方法到 Transformer 架构。

| 方向 | 内容 |
|------|------|
| **文本预处理** | 分词（jieba）、停用词、词袋模型、TF-IDF、Word2Vec、文本数据分析 |
| **RNN 系列** | RNN → LSTM → GRU 的演进，注意力机制原理，英译法实战案例 |
| **Transformer** | 输入层、组件层（Multi-Head Attention、FFN）、编码器、解码器、完整模型实现 |
| **迁移学习** | FastText 训练、预训练模型（BERT 系列）加载与微调 |

### 5. LLM 大模型应用

> 掌握本地大模型部署与 API 调用，具备 LLM 应用开发能力。

- **Ollama 本地部署**：在 Linux 环境私有化部署大模型（Qwen），配合 ChatBox 使用
- **Streamlit 聊天应用**：基于 Ollama + Qwen 构建交互式聊天机器人（支持历史记录）
- **API 集成**：阿里百炼平台、DeepSeek API 的调用与集成
- **LangChain**：链式调用、Agent 工具、RAG 检索增强生成

### 6. 投满分文本分类实战项目（TMF）

> ⭐ **仓库核心项目** — 以"投满分"文本分类任务为载体，完整演示从传统 ML 到 LLM 的全链路解决方案。

```
数据 EDA → 预处理 → 多模型训练 → 评估对比 → 模型压缩 → API 服务化 → Web 界面
```

#### 方案矩阵

| 方案 | 模型 | 特点 | 适用场景 |
|------|------|------|---------|
| **01-rf** | TF-IDF + 随机森林 | 训练快、可解释性强 | 资源受限、快速迭代 |
| **02-fasttext** | FastText | 极速训练、轻量部署 | 海量数据、线上实时 |
| **03-bert** | BERT 微调 | 高精度、预训练知识迁移 | 精度优先 |
| **04-LLM** | DeepSeek / 阿里百炼 | 零样本/少样本推理 | 标注数据稀缺 |
| **05-distill** | BERT(Teacher) → BiLSTM(Student) | 知识蒸馏，精度与速度平衡 | 生产环境部署 |
| **06-prune** | BERT + 剪枝 | 模型压缩，减小体积 | 边缘设备部署 |

#### 工程化能力

每个方案均包含完整的工程化链路：

```python
config.py              # 统一配置管理
data_preprocessing.py  # 数据清洗与预处理
*_train.py             # 模型训练脚本
*_predict_service.py   # 推理服务封装
api_flask_server.py    # Flask REST API 服务化
app_streamlit.py       # Streamlit Web 可视化界面
```

---

## 学习笔记体系

`0-0-学习笔记/` 目录包含 **21 篇** 系统化的技术笔记，覆盖 AI 工程师所需的全部核心知识：

| 编号 | 主题 | 核心内容 |
|------|------|---------|
| 01-02 | Ollama 与大模型入门 | 本地部署、Streamlit 聊天应用 |
| 03-04 | 深度学习基础 | PyTorch 框架、神经网络、反向传播 |
| 05-06 | RNN 与 Prompt 工程 | 循环神经网络、提示词设计方法论 |
| 07-08 | 项目实战 | 金融行业评估、企业级大模型平台 |
| 09-10 | AI 应用平台 | Coze、Dify 低代码平台 |
| 11-12 | LangChain 与 RAG | 开发工具链、检索增强问答系统 |
| 13-14 | Function Call 与微调 | 函数调用、GPT-2 微调 |
| 15-16 | BERT 与 NLP 项目 | 新零售微调、评论分类与信息抽取 |
| 17-18 | 大模型进阶 | Qwen 微调、知识蒸馏 |
| 19-21 | 综合能力 | 多模态大模型、论文导读、面试技巧 |

---

## 面试资源

| 资源 | 路径 | 说明 |
|------|------|------|
| 精选面试题 | `0-0-AI面试题/精选200道面试题.md` | 200 道高频 AI 面试题汇总 |
| 参考简历 | `0-0-面试模板以及注意事项/重点：参考简历/` | 4 份高质量简历模板（PDF） |
| 面试技巧 | `0-0-学习笔记/21-简历优化与面试技巧.md` | 简历优化策略与面试方法论 |

---

## 环境搭建

### 系统要求

- Python >= 3.10
- macOS / Linux（推荐 Ubuntu 22.04+）
- CUDA 11.8+（如需 GPU 加速训练）

### 安装步骤

```bash
# 1. 克隆仓库
git clone <repo-url>
cd 00_python_code

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4.（可选）安装 Ollama 本地大模型
# 参考：https://ollama.com
ollama pull qwen2.5
```

### 核心依赖

```
numpy>=2.0,<2.5
pandas~=2.3.3
matplotlib~=3.10.8
scikit-learn~=1.7.2
torch~=2.12.0
streamlit~=1.54.0
ollama==0.3.3
openpyxl~=3.1.5
```

---

## 快速开始

### 运行文本分类项目（推荐）

```bash
# 进入项目目录
cd 5-投满分文本分类任务项目/TMF/03-bert

# 训练 BERT 模型
python bert_train_and_eval.py

# 启动 Flask API 服务
python api_flask_server.py

# 或启动 Streamlit Web 界面
streamlit run app_streamlit.py
```

### 运行 Ollama 聊天机器人

```bash
cd 00-ollama入门

# 启动 Streamlit 聊天应用
streamlit run Streamlit_demo_history.py
```

---

## License

本项目仅供学习参考使用。
