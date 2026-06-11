# Prompt-Tuning 与提示词工程

> 日期：2025年11月21日~22日  
> 主题：Prompt-Tuning 方法入门与进阶、提示词工程应用（金融文本分类/信息抽取/文本匹配）、In-Context Learning

---

## 一、Prompt-Tuning 方法概述

### 1.1 什么是 Prompt-Tuning

Prompt-Tuning（提示微调）是一种**不修改模型参数**，通过设计合适的提示词（Prompt）来引导预训练大模型完成特定任务的方法。

**核心思想**：不训练模型，而是"教"模型如何理解任务。

### 1.2 与传统微调的区别

| 方法 | 是否修改模型参数 | 数据需求 | 适用场景 |
|------|---------------|---------|---------|
| 全量微调（Fine-Tuning） | 是 | 大量标注数据 | 需要深度适配的任务 |
| Prompt-Tuning | 否 | 少量示例即可 | 快速验证、资源有限 |
| In-Context Learning | 否 | 几个示例 | 零样本/少样本任务 |

### 1.3 常用 Prompt 设计模式

1. **Zero-Shot**（零样本）：直接描述任务，不给示例
2. **Few-Shot**（少样本）：先给几个正确示例，再提问（In-Context Learning）
3. **角色设定**：通过 System Prompt 给模型指定角色
4. **格式约束**：明确指定输出格式（如 JSON、列表等）

---

## 二、Ollama 在 Windows 上的高效配置

### 2.1 安装与环境配置

1. 从 [ollama.com](https://ollama.com/) 下载安装包
2. 安装后**先退出** Ollama（右键图标 → Quit），再配置环境变量
3. 配置模型存储路径（避免占用 C 盘）：
   - 新建系统变量 `OLLAMA_MODELS`
   - 值设置为其他盘路径，如 `D:\Work\ollama\models`

### 2.2 常用命令

```bash
ollama run <模型名称>    # 运行模型（首次会下载）
ollama list              # 查看已安装的模型
ollama ps                # 查看正在运行的模型
```

**显存要求**：7B 模型至少 8GB 显存，13B 至少 16GB 显存

### 2.3 Python 调用方式

```python
# 方式1：直接调用
import ollama
response = ollama.chat(model='qwen2.5:7b',
                       messages=[{'role': 'user', 'content': '你好'}])
print(response['message']['content'])

# 方式2：Client 对象（支持远程调用）
from ollama import Client
client = Client(host='http://127.0.0.1:11434')
response = client.chat(model='qwen2.5:7b', messages=[...])

# 方式3：流式输出
stream = ollama.chat(model='qwen2.5:7b', messages=[...], stream=True)
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
```

---

## 三、提示词工程实战 - 金融文本分类

### 3.1 任务说明

将金融领域文本自动分类到指定类别：`['新闻报道', '财务报告', '公司公告', '分析师报告']`

### 3.2 Prompt 设计要点

- 通过 **System Prompt** 指定模型角色（"你是一个文本分类器"）
- 使用 **Few-Shot**（In-Context Learning）：先展示每个类别的示例
- 明确指定输出格式

### 3.3 代码实现

```python
import ollama

# 1. 提供所有类别及每个类别的示例
class_examples = {
    '新闻报道': '今日，股市经历了一轮震荡，受到宏观经济数据和全球贸易紧张局势的影响。',
    '财务报告': '本公司年度财务报告显示，去年公司实现了稳步增长的盈利。',
    '公司公告': '本公司高兴地宣布成功完成最新一轮并购交易。',
    '分析师报告': '最新的行业分析报告指出，科技公司的创新将成为未来增长的主要推动力。'
}

# 2. 构建 Few-Shot 前置对话（In-Context Learning）
def init_prompts():
    class_list = list(class_examples.keys())
    pre_history = [
        {"role": "system", "content": f"现在你是一个文本分类器，你需要按照要求将我给你的句子分类到：{class_list}类别中。"}
    ]
    # 添加每个类别的示例对话
    for _type, example in class_examples.items():
        pre_history.append({"role": "user", "content": f'"{example}"是 {class_list} 里的什么类别？'})
        pre_history.append({"role": "assistant", "content": _type})
    return {'class_list': class_list, 'pre_history': pre_history}

# 3. 推理
def inference(sentences, custom_settings):
    for sentence in sentences:
        sentence_with_prompt = f'"{sentence}"是 {custom_settings["class_list"]} 里的什么类别？'
        response = ollama.chat(
            model='qwen2.5:7b',
            messages=[*custom_settings['pre_history'],
                      {"role": 'user', "content": sentence_with_prompt}]
        )
        print(f"句子: {sentence}")
        print(f"分类结果: {response['message']['content']}")
```

**关键技巧**：`*custom_settings['pre_history']` 使用 Python 的解包操作符将历史对话展开。

---

## 四、提示词工程实战 - 金融信息抽取

### 4.1 任务说明

从金融文本中抽取指定实体：`['日期', '股票名称', '开盘价', '收盘价', '成交量']`，以 JSON 格式输出。

### 4.2 Prompt 设计要点

- System Prompt 指定角色（"你是一个信息抽取助手"）
- 给出抽取示例（Few-Shot），包含 JSON 输出格式
- 处理"原文中未提及"的情况

### 4.3 代码实现

```python
import json
import ollama
import re

# 定义抽取的实体属性
schema = {
    '金融': ['日期', '股票名称', '开盘价', '收盘价', '成交量'],
}

# Prompt 模板
IE_PATTERN = "{}\n\n提取上述句子中{}的实体，并按照JSON格式输出，上述句子中不存在的信息用['原文中未提及']来表示，多个值之间用','分隔。"

# 提供示例
ie_examples = {
    '金融': [{
        'content': '2023-01-10，股市震荡。股票古哥-D[EOOE]美股今日开盘价100美元，最终以102美元收盘，成交量达到520000。',
        'answers': {
            '日期': ['2023-01-10'],
            '股票名称': ['古哥-D[EOOE]美股'],
            '开盘价': ['100美元'],
            '收盘价': ['102美元'],
            '成交量': ['520000'],
        }
    }]
}

# 后处理模型输出（提取 JSON）
def clean_response(response):
    if '```json' in response:
        res = re.findall(r'```json(.*?)```', response, re.DOTALL)
        if len(res) and res[0]:
            response = res[0]
    try:
        return json.loads(response)
    except:
        return response
```

**重要**：大模型有时会在 JSON 外包裹 ` ```json ... ``` `，需要后处理提取。

---

## 五、提示词工程实战 - 金融文本匹配

### 5.1 任务说明

判断两段金融文本的语义是否相似。

### 5.2 Prompt 设计要点

- System Prompt："你需要帮助我完成文本匹配任务，当我给你两个句子时，你需要回答我这两句话语义是否相似。只需要回答是否相似，不要做多余的回答。"
- 提供正反两方面的示例

### 5.3 代码实现

```python
import ollama

# 提供相似/不相似的示例
examples = {
    '是': [
        ('公司ABC发布了季度财报，显示盈利增长。', '财报披露，公司ABC利润上升。'),
    ],
    '不是': [
        ('黄金价格下跌，投资者抛售。', '外汇市场交易额创下新高。'),
    ]
}

def init_prompts():
    pre_history = [
        {"role": "system", "content": "现在你需要帮助我完成文本匹配任务，当我给你两个句子时，你需要回答我这两句话语义是否相似。只需要回答是否相似，不要做多余的回答。"}
    ]
    for key, sentence_pairs in examples.items():
        for s1, s2 in sentence_pairs:
            pre_history.append({"role": "user", "content": f'句子一: {s1}\n句子二: {s2}\n上面两句话是相似的语义吗？'})
            pre_history.append({"role": "assistant", "content": key})
    return {'pre_history': pre_history}

# 推理
for s1, s2 in sentence_pairs:
    prompt = f'句子一: {s1}\n句子二: {s2}\n上面两句话是相似的语义吗？'
    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[*custom_settings["pre_history"],
                  {"role": 'user', "content": prompt}]
    )
    print(f"结果: {response['message']['content']}")
```

---

## 六、Prompt 设计最佳实践总结

### 6.1 核心原则

1. **明确任务定义**：告诉模型"你要做什么"
2. **角色设定**：用 System Prompt 定义模型角色
3. **提供示例**：用 Few-Shot 展示正确的输入输出格式
4. **格式约束**：明确指定输出格式（JSON、列表、单个词等）
5. **后处理**：处理模型可能输出的多余内容（如 markdown 代码块）

### 6.2 In-Context Learning 模式

```
System: 你是一个 [角色]，你需要 [任务描述]。
User: [示例输入1]
Assistant: [示例输出1]
User: [示例输入2]
Assistant: [示例输出2]
User: [实际输入]
→ 模型输出结果
```

### 6.3 三种金融 NLP 任务对比

| 任务 | 输入 | 输出 | Prompt 关键点 |
|------|------|------|-------------|
| 文本分类 | 一段文本 | 类别标签 | 列举所有类别 + 每类一个示例 |
| 信息抽取 | 一段文本 | JSON 实体 | 指定抽取字段 + JSON 格式 + 未提及处理 |
| 文本匹配 | 两段文本 | 是/否 | 正反示例 + 限制只回答是/否 |

### 6.4 Ollama 聊天消息结构

```python
messages = [
    {"role": "system", "content": "系统指令/角色设定"},
    {"role": "user", "content": "用户输入"},
    {"role": "assistant", "content": "助手回复"},
    {"role": "user", "content": "用户追问"},
    # ... 可继续交替
]
```

- `system`：系统级指令，设定模型角色和行为
- `user`：用户的输入
- `assistant`：模型的回复（用于构建对话历史和 Few-Shot 示例）
