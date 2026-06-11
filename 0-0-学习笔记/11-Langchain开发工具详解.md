# Langchain 开发工具详解

> 日期：2025年12月8日~12日  
> 主题：Langchain 框架核心模块（Models、Prompts、Chains、Agents、Memory、Indexes）、知识库问答

---

## 一、Langchain 概述

**Langchain** 是一个用于开发大模型应用的框架，提供了一套标准化的接口和工具链，帮助开发者快速构建基于 LLM 的应用。

### 1.1 核心模块

| 模块 | 功能 | 说明 |
|------|------|------|
| Models | 模型接口 | 统一封装各种 LLM 的调用方式 |
| Prompts | 提示词模板 | 管理和复用提示词 |
| Chains | 链式调用 | 将多个步骤串联成工作流 |
| Agents | 智能体 | 让模型自主选择工具完成任务 |
| Memory | 记忆 | 管理对话历史上下文 |
| Indexes | 索引/检索 | 知识库构建和向量检索 |

### 1.2 安装

```bash
pip install langchain
pip install langchain_community
pip install faiss-cpu        # 向量数据库
pip install ollama
```

---

## 二、Models 模块 - 模型接口

### 2.1 接入 Ollama 本地模型

```python
# 方式1：直接调用 Ollama
import ollama
response = ollama.chat(model='qwen3:8b',
                       messages=[{'role': 'user', 'content': '你好'}])
print(response['message']['content'])

# 方式2：通过 Langchain 封装调用
from langchain_community.llms import Ollama
model = Ollama(base_url="http://127.0.0.1:11434", model="qwen3:8b", temperature=0)
result = model.invoke("你好")

# 方式3：Embedding 模型
from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="bge-m3", temperature=0)
```

---

## 三、Prompts 模块 - 提示词管理

### 3.1 Zero-Shot 提示词模板

```python
from langchain import PromptTemplate

template = "我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字"

prompt = PromptTemplate(
    input_variables=["lastname"],
    template=template,
)

# 填充模板
prompt_text = prompt.format(lastname="王")
# 结果: "我的邻居姓王，他生了个儿子，给他儿子起个名字"

# 调用模型
result = model.invoke(prompt_text)
```

### 3.2 Few-Shot 提示词模板

```python
from langchain import PromptTemplate, FewShotPromptTemplate

# 定义示例
examples = [
    {"word": "开心", "antonym": "难过"},
    {"word": "高", "antonym": "矮"},
]

# 示例的模板
example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template="单词: {word}\n反义词: {antonym}\n",
)

# Few-Shot 模板
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="给出每个单词的反义词",
    suffix="单词: {input}\n反义词:",
    input_variables=["input"],
    example_separator="\n",
)

# 生成最终 prompt
prompt_text = few_shot_prompt.format(input="粗")
# 输出:
# 给出每个单词的反义词
# 单词: 开心  反义词: 难过
# 单词: 高    反义词: 矮
# 单词: 粗    反义词:

result = model.invoke(prompt_text)  # 输出: "细"
```

---

## 四、Chains 模块 - 链式调用

### 4.1 LLMChain

将 Prompt + LLM 封装成一条链：

```python
from langchain.chains import LLMChain

first_chain = LLMChain(llm=model, prompt=first_prompt)
result = first_chain.run("王")
```

### 4.2 SimpleSequentialChain

将多条链串联，前一条的输出作为后一条的输入：

```python
from langchain.chains import SimpleSequentialChain

# 第一条链：给邻居的儿子起名
first_prompt = PromptTemplate(
    input_variables=["lastname"],
    template="我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字",
)
first_chain = LLMChain(llm=model, prompt=first_prompt)

# 第二条链：给儿子起小名
second_prompt = PromptTemplate(
    input_variables=["child_name"],
    template="邻居的儿子名字叫{child_name}，给他起一个小名",
)
second_chain = LLMChain(llm=model, prompt=second_prompt)

# 串联两条链
overall_chain = SimpleSequentialChain(
    chains=[first_chain, second_chain],
    verbose=False
)

# 只需传入第一个参数
result = overall_chain.run("王")
# 输出: 王XX -> 小X
```

---

## 五、Agents 模块 - 智能体

### 5.1 Agent 概念

Agent 能够根据用户的问题，**自主选择工具**来完成任务。

```python
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools

# 实例化模型
llm = Ollama(model="qwen3:8b")

# 加载工具
tools = load_tools(["llm-math"], llm=llm)  # 数学计算工具
# tools = load_tools(["serpapi", "llm-math"], llm=llm)  # + 联网搜索

# 创建 Agent
agent = initialize_agent(
    tools, llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  # 显示思考过程
)

# Agent 自主工作
result = agent.run("解方程：3x + 4(x + 2) - 84 = y; 其中x为3，y是多少？")
```

### 5.2 Agent 工作流程

```
用户问题 → Agent 思考 → 选择工具 → 执行工具 → 获取结果 → 继续思考 → 最终回答
```

### 5.3 常用工具

| 工具名 | 功能 | 安装 |
|--------|------|------|
| `llm-math` | 数学计算 | 内置 |
| `serpapi` | 网络搜索 | `pip install google-search-results` |
| `python_repl` | 执行 Python 代码 | 内置 |

---

## 六、Memory 模块 - 对话记忆

管理多轮对话的上下文历史，让模型"记住"之前的对话。

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("你好，我叫小明")
memory.chat_memory.add_ai_message("你好小明，有什么可以帮你的？")
```

---

## 七、Indexes 模块 - 知识库检索

### 7.1 向量数据库构建

```python
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# 创建 Embedding 模型
embeddings = OllamaEmbeddings(model="bge-m3", temperature=0)

# 从文档构建向量库
db = FAISS.from_documents(documents, embeddings)
db.save_local("faiss/wuliu")
```

### 7.2 知识库问答实现

```python
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# 加载向量库
embeddings = OllamaEmbeddings(model="bge-m3", temperature=0)
db = FAISS.load_local("faiss/wuliu", embeddings, allow_dangerous_deserialization=True)

# 检索相关文档
question = '我的快递出发地是哪？预计几天到达？'
docs = db.similarity_search(question, k=2)

# 拼接上下文
related_content = "\n".join([doc.page_content for doc in docs])

# 构建 Prompt
PROMPT_TEMPLATE = """
基于以下已知信息，简洁和专业地回答用户的问题。不允许编造。
已知内容: {context}
问题: {question}"""

prompt = PromptTemplate(input_variables=["context", "question"], template=PROMPT_TEMPLATE)
my_prompt = prompt.format(context=related_content, question=question)

# 调用模型
model = Ollama(model="qwen3:8b")
result = model.invoke(my_prompt)
```

---

## 八、Langchain 核心模式总结

| 模式 | 代码示例 | 适用场景 |
|------|---------|---------|
| PromptTemplate | `PromptTemplate(template=..., input_variables=...)` | 参数化提示词 |
| FewShotPromptTemplate | `FewShotPromptTemplate(examples=..., ...)` | 少样本学习 |
| LLMChain | `LLMChain(llm=..., prompt=...)` | 单步骤任务 |
| SimpleSequentialChain | `SimpleSequentialChain(chains=[...])` | 多步骤串联 |
| Agent | `initialize_agent(tools, llm, ...)` | 需要工具调用 |
| FAISS | `FAISS.from_documents(docs, embeddings)` | 知识库检索 |
| similarity_search | `db.similarity_search(query, k=2)` | 向量相似度检索 |
