# RNN 循环神经网络与大模型开发入门

> 日期：2025年11月16日  
> 主题：RNN 循环神经网络、jieba 分词、词嵌入、歌词生成案例、语言模型发展、HuggingFace、评估指标（BLEU/ROUGE/PPL）

---

## 一、中文分词 - jieba

在进行自然语言处理之前，首先需要对中文文本进行分词处理。`jieba` 是最常用的中文分词库。

### 1.1 基本用法

```python
import jieba

content = "北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。"

# 精确模式分词（最常用）
words = jieba.lcut(content)
# 结果: ['北京', '冬奥', '的', '进度条', '已经', '过半', '，', '不少', '外国', '运动员', ...]

# 搜索引擎模式分词（适合搜索引擎建立索引）
words_search = jieba.lcut_for_search(content)
```

- `lcut()`：精确模式，返回分词列表
- `lcut_for_search()`：搜索模式，会将较长的词进一步切分

---

## 二、RNN 循环神经网络

RNN（Recurrent Neural Network）是专门处理**序列数据**的神经网络，能够捕捉前后文之间的依赖关系。

### 2.1 词嵌入层（Embedding）

词嵌入是将离散的词汇索引转换为连续的低维向量表示，使语义相近的词在向量空间中距离更近。

```python
import torch.nn as nn

# 创建词嵌入层
# num_embeddings: 词汇表大小（有多少个不同的词）
# embedding_dim: 词向量维度（每个词用多少个数字表示）
ebd = nn.Embedding(num_embeddings=1000, embedding_dim=128)

# 将词的索引转换为词向量
word_vec = ebd(torch.tensor(5))  # 获取索引为5的词对应的128维向量
```

### 2.2 RNN 循环层

RNN 的核心特点：每个时间步的输出不仅依赖当前输入，还依赖**上一个时间步的隐藏状态**，从而实现对序列信息的记忆。

```python
import torch
import torch.nn as nn

# 创建 RNN 层
rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1)
# input_size: 词向量维度
# hidden_size: 隐藏状态维度（特征个数）
# num_layers: 隐藏层层数（默认1）

# 输入数据：(序列长度, 批次大小, 词向量维度)
inputs = torch.randn(5, 32, 128)  # 5个token, 32条数据, 每个token 128维

# 初始隐藏状态：(层数, 批次大小, 隐藏维度)
hidden = torch.zeros(1, 32, 256)

# 前向传播
output, new_hidden = rnn(inputs, hidden)
# output 形状: (5, 32, 256) - 每个时间步的输出
# new_hidden 形状: (1, 32, 256) - 最终的隐藏状态
```

**输入维度说明**：

| 维度 | 含义 | 示例 |
|------|------|------|
| 第1维 | 序列长度（句子中 token 数） | 5 |
| 第2维 | 批次大小（一次处理多少条） | 32 |
| 第3维 | 词向量维度（= input_size） | 128 |

---

## 三、RNN 综合案例：歌词生成

### 3.1 项目流程

1. **数据准备**：读取歌词 → jieba 分词 → 构建词汇表 → 词索引替换
2. **自定义 Dataset**：实现滑动窗口取数据（y 相对 x 整体后移 1 位）
3. **搭建 RNN 模型**：Embedding → RNN → Linear 输出
4. **训练与生成**

### 3.2 数据准备

```python
import jieba

def create_data():
    original_words = []  # 每行歌词分词后的结果
    unique_words = []    # 去重后的词汇表

    with open("data/jaychou_lyrics.txt", mode="r", encoding="UTF-8") as f:
        while True:
            line = f.readline()
            if line == "":
                break
            words = jieba.lcut(line)
            original_words.append(words)
            for word in words:
                if word not in unique_words:
                    unique_words.append(word)

    # 构建词→索引的字典
    word_dict = {word: index for index, word in enumerate(unique_words)}

    # 将所有歌词替换为索引序列
    corpus_idx = []
    for line_words in original_words:
        for word in line_words:
            corpus_idx.append(word_dict.get(word))
        corpus_idx.append(word_dict.get(" "))  # 用空格索引分隔每行

    return unique_words, word_dict, corpus_idx, len(unique_words)
```

### 3.3 自定义 Dataset

```python
from torch.utils.data import Dataset

class LyricsDataset(Dataset):
    def __init__(self, corpus_idx, num_chars):
        self.corpus_idx = corpus_idx
        self.num_chars = num_chars       # 每次取连续多少个词
        self.word_count = len(corpus_idx)
        self.number = self.word_count // self.num_chars

    def __getitem__(self, index):
        start = min(max(0, index), self.word_count - self.num_chars - 1)
        end = start + self.num_chars
        x = self.corpus_idx[start:end]       # 输入序列
        y = self.corpus_idx[start+1:end+1]   # 目标序列（后移1位）
        return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        return self.number
```

**核心思想**：y_train 就是 x_train 整体往后移动 1 位，模型学习根据前文预测下一个词。

### 3.4 RNN 模型搭建

```python
class LyricsRNNModel(nn.Module):
    def __init__(self, unique_word_count):
        super().__init__()
        # 词嵌入层：索引 → 128维向量
        self.ebd = nn.Embedding(num_embeddings=unique_word_count, embedding_dim=128)
        # RNN 循环层
        self.rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1)
        # 全连接输出层
        self.output = nn.Linear(in_features=256, out_features=unique_word_count)

    def forward(self, inputs, hidden):
        embed = self.ebd(inputs)              # 词嵌入
        out, hidden = self.rnn(embed.transpose(0, 1), hidden)  # RNN（需转置维度）
        output = self.output(out.reshape(-1, out.shape[-1]))   # 全连接输出
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, 256)
```

**注意 `transpose(0,1)`**：Embedding 输出形状为 `(batch, seq_len, emb_dim)`，但 RNN 要求输入形状为 `(seq_len, batch, emb_dim)`，因此需要交换前两个维度。

### 3.5 训练与生成

```python
# 训练
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for x, y in dataloader:
    hidden = model.init_hidden(batch_size=5)
    output, hidden = model(x, hidden)
    y = torch.transpose(y, 0, 1).reshape(-1)  # 展平为一维
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 生成歌词
def predict(start_word, sentence_length):
    word_idx = word_to_index[start_word]
    hidden = model.init_hidden(batch_size=1)
    for _ in range(sentence_length):
        output, hidden = model(torch.tensor([[word_idx]]), hidden)
        word_idx = torch.argmax(output).item()
        print(unique_words[word_idx], end='')
```

---

## 四、HuggingFace Transformers 使用

HuggingFace 是最流行的 NLP 预训练模型库，提供丰富的预训练模型和简单的 API。

### 4.1 安装

```bash
pip install transformers
pip install sentencepiece
```

### 4.2 中译英示例

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

model_name = 'liam168/trans-opus-mt-zh-en'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 创建翻译管道
translate = pipeline('translation_zh_to_en', model=model, tokenizer=tokenizer)
result = translate('我想要学习大模型相关的任务')
```

---

## 五、语言模型发展史

### 5.1 基于统计的语言模型（2000s 前）

**核心思想**：利用 N-gram 模型估计词序列概率。假设一个词出现的概率只依赖于前 N-1 个词。

- **链式法则**：\(P(S) = P(w_1) \cdot P(w_2|w_1) \cdot P(w_3|w_1,w_2) \cdots\)
- **马尔可夫假设**：当前词只依赖前 N-1 个词
- **优点**：简单高效
- **缺点**：数据稀疏、忽略长距离上下文

### 5.2 基于神经网络的语言模型

**解决了统计模型的问题**：
- 通过词向量解决数据稀疏（语义相似的词向量距离近）
- 能捕获更复杂的上下文关系

**三层结构**：
1. **输入层**：one-hot 编码 → 词向量
2. **隐藏层**：非线性变换，学习上下文表示（Tanh 激活）
3. **输出层**：生成下一个词的概率分布

**局限**：长序列建模能力有限、可能出现梯度消失

### 5.3 基于 Transformer 的语言模型

通过**自注意力机制**（Self-Attention）解决长序列依赖问题。

**两阶段训练**：
1. **预训练**：在大规模无标注语料上学习通用语言表示
   - MLM（掩码语言模型）：如 BERT，随机遮挡 15% 的 token 并预测
   - NSP（下一句预测）：判断两个句子是否连续
   - CLM（因果语言模型）：如 GPT，根据上文预测下一个词
2. **微调**：在特定 NLP 任务上用少量标注数据适配

### 5.4 大语言模型（2020 至今）

本质：基于 Transformer 架构，在海量数据上训练的超大规模概率预测模型。

**"大"在何处**：
- 巨大的参数规模（数十亿~万亿）
- 海量的训练数据
- 极深的网络层数

**核心能力**：根据给定上文（Prompt），预测下一个最可能的词，循环生成完整文本。

---

## 六、模型评估指标

### 6.1 BLEU（机器翻译评估）

- **全称**：Bilingual Evaluation Understudy
- **原理**：基于 n-gram 匹配计算机器翻译与参考翻译的相似度
- **取值**：[0, 1]，越接近 1 越好，一般 >0.5 即可用
- **应用**：机器翻译任务

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

candidate = ["This", "is", "some", "generated", "text"]
references = [
    ["This", "is", "a", "reference", "text"],
    ["This", "is", "another", "reference", "textEEEE"]
]

smoothie = SmoothingFunction().method4  # 平滑函数，防止高阶 n-gram 分数为 0

bleu_1 = sentence_bleu(references, candidate, weights=(1,0,0,0), smoothing_function=smoothie)
bleu_2 = sentence_bleu(references, candidate, weights=(0.5,0.5,0,0), smoothing_function=smoothie)
bleu_4 = sentence_bleu(references, candidate, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoothie)
```

**n-gram 匹配说明**：
- 1-gram：单个词匹配（精确度 = 匹配词数/总词数）
- 2-gram：连续 2 词匹配
- 高阶 n-gram 匹配更严格，使用平滑函数避免分数归零

### 6.2 ROUGE（生成文本评估）

- **全称**：Recall-Oriented Understudy for Gisting Evaluation
- **原理**：基于**召回率**比较生成文本与参考文本
- **指标**：ROUGE-1（词汇覆盖）、ROUGE-2（短语结构）、ROUGE-L（句子连贯性）
- **应用**：新闻摘要、自动摘要等生成类任务

```python
from rouge import Rouge

rouge = Rouge()
generated = "这 是 一些 生成 文本"
reference = "这 是 另 一个 参考 文本"

scores = rouge.get_scores(generated, reference)
# scores[0]["rouge-1"]["p"]  # 精确率
# scores[0]["rouge-1"]["r"]  # 召回率
# scores[0]["rouge-1"]["f"]  # F1 分数
```

### 6.3 PPL 困惑度（语言模型评估）

- **全称**：Perplexity
- **原理**：衡量模型对测试集的预测能力，即模型预测样本的"好坏程度"
- **取值**：[0, +∞)，越接近 0 越好
- **应用**：语言模型、概率生成模型

**计算过程**：
1. 计算每个句子中所有词概率的累乘
2. 求平均负对数概率：\(-\frac{1}{N} \log_2 P(W)\)
3. 困惑度 = \(2^{\text{平均负对数概率}}\)

```python
import math

sentences = [['I', 'have', 'a', 'pen'], ['He', 'has', 'a', 'book']]
unigram = {'I': 1/12, 'have': 1/12, 'a': 3/12, 'pen': 1/12, 'He': 1/12, 'has': 2/12, 'book': 1/12}

total_ppl = 0
for sentence in sentences:
    prob = 1
    for word in sentence:
        prob *= unigram[word]
    avg_neg_log = -math.log(prob, 2) / len(sentence)
    ppl = 2 ** avg_neg_log
    total_ppl += ppl

perplexity = total_ppl / len(sentences)
```

### 6.4 评估指标总结

| 指标 | 适用任务 | 取值范围 | 含义 |
|------|---------|---------|------|
| BLEU | 机器翻译 | [0,1]，越高越好 | n-gram 匹配精确度 |
| ROUGE | 摘要生成 | [0,1]，越高越好 | 召回率导向的文本覆盖度 |
| PPL | 语言模型 | [0,+∞)，越低越好 | 模型预测的困惑程度 |
