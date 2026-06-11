# 15 - 新零售行业评价决策系统（BERT + PET / P-Tuning）

> 对应课程日期：2025年12月28日 ~ 2026年1月6日

---

## 一、项目背景

### 1.1 业务场景

新零售行业中，用户在电商平台留下大量评论文本，企业需要对这些评论进行**自动分类**，识别出评论所属的商品类别（如：电脑、水果、平板、衣服、酒店、洗浴、书籍、蒙牛、手机、电器），从而支撑商品分析、用户画像、舆情监控等决策。

### 1.2 技术方案

本项目采用**两种Prompt-Tuning方式**实现文本分类：
1. **BERT + PET（Pattern Exploiting Training）**：使用硬模板 + Verbalizer映射
2. **BERT + P-Tuning**：使用可学习的连续向量（soft prompt）代替离散模板

两种方式都基于BERT的**MLM（Masked Language Model）** 任务形式，将分类问题转换为填空问题。

---

## 二、BERT + PET方式

### 2.1 核心思想

PET将文本分类任务转换为**掩码语言模型（MLM）的填空任务**：

- 原始数据：`衣服\t不好看，不值这个价钱`
- 通过模板转换：`这是一条[MASK][MASK]评论：不好看，不值这个价钱。`
- BERT预测 `[MASK][MASK]` 位置的词 → 得到子标签（如"衣服"）
- 通过Verbalizer映射到主标签

### 2.2 数据格式

**训练数据**（`train.txt`，每行格式为 `标签\t文本`）：
```
电脑	(1)这款笔记本外观感觉挺漂亮的...
水果	什么苹果啊，都没有苹果味...
衣服	不好看，不值这个价钱
酒店	房间超级小，根本就不值688元的价格...
```

**Prompt模板**（`prompt.txt`）：
```
这是一条{MASK}评论：{textA}。
```

**Verbalizer映射**（`verbalizer.txt`，主标签→子标签列表）：
```
电脑	电脑
水果	水果,苹果,香蕉,榴莲,西瓜
平板	平板
衣服	衣服
酒店	酒店
洗浴	洗浴
书籍	书籍
蒙牛	蒙牛
手机	手机
电器	电器
```

### 2.3 项目结构

```
01_Bert_PET微调/
├── pet_config.py              # 项目配置（设备、路径、超参数）
├── data/
│   ├── train.txt              # 训练集（63条）
│   ├── dev.txt                # 验证集（590条）
│   ├── prompt.txt             # Prompt模板定义
│   └── verbalizer.txt         # 标签词映射表
├── data_handle/
│   ├── template.py            # HardTemplate硬模板类
│   ├── data_preprocess.py     # 数据预处理（convert_example）
│   └── data_loader.py         # DataLoader封装
├── utils/
│   ├── verbalizer.py          # Verbalizer标签映射工具
│   ├── common_utils.py        # mlm_loss + convert_logits_to_ids
│   └── metirc_utils.py        # 评估指标（accuracy/precision/recall/F1）
├── train.py                   # 训练主脚本
├── inference.py               # 推理脚本
└── bert-base-chinese/         # BERT预训练模型文件
```

### 2.4 硬模板（HardTemplate）

**HardTemplate类**负责将文本按模板格式编码：

```python
class HardTemplate(object):
    def __init__(self, prompt: str):
        # prompt格式: "这是一条{MASK}评论：{textA}。"
        self.prompt = prompt
        self.inputs_list = []    # 拆解后的各part列表
        self.custom_tokens = set()  # {'MASK', 'textA'}
        self.prompt_analysis()   # 解析prompt模板

    def __call__(self, inputs_dict, tokenizer, mask_length, max_seq_len=512):
        """
        输入: {"textA": "这个手机也太卡了", "MASK": "[MASK]"}
        输出: {
            'input_ids': [101, ..., 102, 0, 0, ...],  # CLS + 编码 + SEP + padding
            'token_type_ids': [0, 0, ...],
            'attention_mask': [1, 1, ..., 0, 0, ...],
            'mask_position': [5, 6]  # MASK token在序列中的位置索引
        }
        """
        # 1. 拼接模板字符串: "这是一条[MASK][MASK]评论：这个手机也太卡了。"
        # 2. tokenizer编码，padding到max_seq_len
        # 3. 找到[MASK] token的位置索引 → mask_position
```

**关键输出**：`mask_position`记录了`[MASK]`在序列中的位置，后续需要从该位置提取模型预测结果。

### 2.5 数据预处理

```python
def convert_example(examples, tokenizer, max_seq_len, mask_length,
                    hard_template, train_mode=True):
    """
    将原始文本数据转换为模型可接收的格式
    输出:
        input_ids: 编码后的token id序列
        token_type_ids: 句子类型标记
        attention_mask: 注意力掩码
        mask_positions: [MASK]位置索引
        mask_labels: 真实标签的token id（仅训练模式）
    """
    for example in examples['text']:
        label, content = example.strip().split('\t')  # 切割标签和文本
        # 用硬模板编码
        encoded = hard_template(
            inputs_dict={'textA': content, 'MASK': '[MASK]'},
            tokenizer=tokenizer, max_seq_len=max_seq_len,
            mask_length=mask_length)
        # 标签编码: "衣服" → [6132, 3302]（token id），截断/补齐到mask_length
```

### 2.6 Verbalizer（标签词映射器）

**核心功能**：建立主标签和子标签之间的映射关系。

```python
class Verbalizer:
    def __init__(self, verbalizer_file, tokenizer, max_label_len):
        # 加载映射: {'水果': ['水果','苹果','香蕉','榴莲','西瓜'], ...}
        self.label_dict = self.load_label_dict(verbalizer_file)

    def find_sub_labels(self, label):
        """主标签 → 所有子标签的token id"""
        # '水果' → {'sub_labels': ['水果','苹果',...], 'token_ids': [[3717,3362],[5741,3362],...]}

    def find_main_label(self, sub_label, hard_mapping=True):
        """子标签 → 主标签"""
        # '苹果' → {'label': '水果', 'token_ids': [3717, 3362]}
        # 若子标签不在列表中，通过最大公共子串(hard_mapping)找最相似的主标签
```

**hard_mapping机制**：当模型预测出一个不在子标签列表中的词时，通过计算**最大公共子串长度**来匹配最相似的主标签，提升鲁棒性。

### 2.7 MLM损失函数

```python
def mlm_loss(logits, mask_positions, sub_mask_labels, criterion, device):
    """
    计算MASK位置的交叉熵损失
    logits: [batch, seq_len, vocab_size] 模型输出
    mask_positions: [batch, mask_num] MASK位置索引
    sub_mask_labels: 每个样本对应的子标签token id列表（变长）
    """
    for single_logits, single_sub_labels, single_positions in zip(...):
        # 1. 提取MASK位置的logits: (mask_num, vocab_size)
        single_mask_logits = single_logits[single_positions]
        # 2. 扩展以匹配所有子标签: (sub_label_num * mask_num, vocab_size)
        single_mask_logits = single_mask_logits.repeat(len(single_sub_labels), 1, 1)
        # 3. 计算交叉熵损失
        cur_loss = criterion(single_mask_logits, single_sub_labels)
        cur_loss = cur_loss / len(single_sub_labels)  # 子标签数归一化
    loss = loss / batch_size
    return loss
```

### 2.8 训练流程

```python
def model2train():
    # 1. 获取数据加载器
    train_dataloader, dev_dataloader = get_data()

    # 2. 加载BERT MLM模型（带MaskedLM头）
    model = AutoModelForMaskedLM.from_pretrained('bert-base-chinese')

    # 3. 创建Verbalizer
    verbalizer = Verbalizer(verbalizer_file, tokenizer, max_label_len=2)

    # 4. 创建优化器（AdamW + 参数分组）和学习率调度器（linear warmup）
    optimizer = torch.optim.AdamW(grouped_params, lr=5e-5)
    lr_scheduler = get_scheduler('linear', optimizer, warmup_steps, total_steps)

    # 5. 训练循环
    for epoch in range(epochs):
        for batch in train_dataloader:
            # 前向传播
            logits = model(input_ids, token_type_ids, attention_mask).logits
            # 通过verbalizer获取子标签
            sub_labels = verbalizer.batch_find_sub_labels(mask_labels)
            # 计算MLM损失
            loss = mlm_loss(logits, mask_positions, sub_labels, criterion)
            # 反向传播
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # 每valid_steps步评估一次，保存F1最高的模型
            if global_step % valid_steps == 0:
                acc, precision, recall, f1 = evaluate_model(...)
                if f1 > best_f1:
                    model.save_pretrained(save_dir)
```

### 2.9 推理流程

```python
def inference(contents):
    # 1. 用模板编码输入文本
    tokenized_output = convert_example(examples, tokenizer, ..., train_mode=False)
    # 2. 模型前向传播
    logits = model(input_ids, ...).logits
    # 3. 提取MASK位置预测的最大概率token
    predictions = convert_logits_to_ids(logits, mask_positions)
    # 4. 通过Verbalizer映射到主标签
    predictions = verbalizer.batch_find_main_label(predictions, hard_mapping=True)
    return predictions  # ['酒店', '酒店', '洗浴', ...]
```

---

## 三、BERT + P-Tuning方式

### 3.1 与PET的区别

| 方面 | PET（Hard Prompt） | P-Tuning（Soft Prompt） |
|------|---------------------|--------------------------|
| 模板类型 | 人工定义的离散文本模板 | 可学习的连续向量（Embedding） |
| 模板参数 | 固定不变 | 参与训练更新 |
| 优点 | 简单直观，无需额外参数 | 自动学习最优模板，灵活性高 |
| 缺点 | 模板质量依赖人工设计 | 需要额外训练参数 |

### 3.2 P-Tuning核心实现

P-Tuning使用一个**Prompt Encoder**（通常是LSTM或MLP）生成连续的prompt向量，替换掉人工定义的模板token：

```python
# 核心思想：
# 原始PET: [CLS] 这是一条 [MASK] 评论：文本内容 [SEP]
# P-Tuning: [CLS] [P1][P2][P3]... [MASK] 文本内容 [SEP]
# 其中 [P1][P2][P3]... 是可学习的连续向量，由PromptEncoder生成
```

**训练时**：
- BERT大部分参数冻结
- Prompt Encoder的参数参与训练
- Verbalizer的标签映射保持不变

### 3.3 项目配置对比

```python
# P-Tuning版本的配置（ptune_config.py）
class ProjectConfig:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    pre_model = 'bert-base-chinese'
    max_seq_len = 512
    batch_size = 8
    learning_rate = 5e-5
    max_label_len = 2
    epochs = 10
    # 新增: prompt encoder相关参数
    prompt_encoder = 'lstm'  # 或 'mlp'
    reparameterization = True  # 是否使用重参数化
```

---

## 四、关键知识点总结

### 4.1 Prompt-Tuning文本分类流程

```
原始文本 → 模板转换(添加[MASK]) → BERT编码 → 提取[MASK]位置logits
    → Verbalizer映射(子标签→主标签) → 预测类别
```

### 4.2 核心组件

| 组件 | 作用 |
|------|------|
| **HardTemplate** | 将文本+MASK模板编码为input_ids，并记录mask_position |
| **Verbalizer** | 建立主标签↔子标签映射，支持hard_mapping兜底 |
| **mlm_loss** | 只计算MASK位置的交叉熵损失，支持变长子标签 |
| **convert_logits_to_ids** | 从模型输出中提取MASK位置最大概率的token |

### 4.3 训练要点

- 使用 `AutoModelForMaskedLM`（带MLM头的BERT），而非分类头
- 损失函数只计算`[MASK]`位置，而非所有位置
- 评估指标：accuracy、precision、recall、F1
- 保存F1最高的模型权重
- 学习率warmup + linear decay调度策略

### 4.4 数据量特点

- 训练集仅63条，验证集590条 → **典型的小样本场景**
- Prompt-Tuning的优势在此场景下尤为突出：利用BERT预训练的MLM能力，只需极少的标注数据即可完成分类任务
