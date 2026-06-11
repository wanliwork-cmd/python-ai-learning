# 02 - Ollama + QWen/DeepSeek 私有化部署聊天机器人

> 日期：2025年11月4日  
> 核心内容：Python调用Ollama API、Streamlit框架、聊天机器人实现

---

## 一、项目概述：黑马智聊机器人

### 技术架构

- **后端模型**：Ollama平台 + Qwen/DeepSeek模型（自然语言处理）
- **前端界面**：Streamlit框架（Python Web应用）
- **对话交互**：用户通过网页输入文本 → Qwen模型处理 → 返回结果展示

### 开发环境依赖

```shell
pip install ollama
pip install langchain -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install langchain-community -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install streamlit==1.32.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 二、Python调用Ollama API

### 2.1 文本续写

```python
import ollama

response = ollama.chat(
    model='qwen2:0.5b',
    messages=[{'role': 'user', 'content': '从前有座山，山里有个庙，续写一下'}]
)
print(response['message']['content'])
```

### 2.2 自定义Host和Port

如果修改了Ollama的监听配置：

```python
import ollama

client = ollama.Client(host="http://192.168.1.100:11434")
response = client.chat(
    model='qwen2:0.5b',
    messages=[{'role': 'user', 'content': '你好'}]
)
print(response['message']['content'])
```

### 2.3 循环问答程序

```python
import ollama

while True:
    prompt = input("请输入您的问题：")
    response = ollama.chat(
        model='qwen2:0.5b',
        messages=[{'role': 'user', 'content': prompt}]
    )
    print(response['message']['content'])
```

### 2.4 AI生成代码

```python
import ollama

prompt = """
请为以下功能生成一段Python代码：
求两个数的最大公约数
"""
response = ollama.chat(
    model='qwen2:0.5b',
    messages=[{'role': 'user', 'content': prompt}]
)
print(response['message']['content'])
```

### 2.5 情感标签分类

```python
import ollama

prompt = """
你需要对用户的反馈进行原因分类。
分类包括：价格过高、售后支持不足、产品使用体验不佳、其他。
回答格式为：分类结果：xx。
用户的问题是：性价比不高，我觉得不值这个价钱。
"""
response = ollama.chat(
    model='qwen2:0.5b',
    messages=[{'role': 'user', 'content': prompt}]
)
print(response['message']['content'])
# 输出：分类结果：价格过高
```

---

## 三、Streamlit 框架入门

Streamlit是一个纯Python的Web框架，几分钟内就能把Python脚本变成可交互的网站，无需前端经验。

### 3.1 安装与运行

```shell
pip install streamlit==1.32.0
streamlit hello       # 验证安装，浏览器自动打开
streamlit run app.py  # 运行自己的应用
```

### 3.2 基础组件

```python
import streamlit as st
import pandas as pd

# 标题
st.title('Streamlit教程')

# 段落文本
st.write('Hello World')

# Markdown
"# 1级标题"
"## 2级标题"

# 图片
st.image('./avatar.jpg', width=400)

# 静态表格
st.table({'name': ['张三', '李四'], 'age': [18, 20]})

# 可交互表格（支持排序、搜索）
st.dataframe(pd.DataFrame({'name': ['张三', '李四'], 'age': [18, 20]}))

# 分割线
st.divider()
```

### 3.3 输入组件

```python
# 文本输入框
name = st.text_input('请输入你的名字：')

# 密码输入框
pwd = st.text_input('密码：', type='password')

# 数字输入框
age = st.number_input('年龄：', value=20, min_value=0, max_value=200, step=1)

# 多行文本框
paragraph = st.text_area("请输入内容：")
```

### 3.4 Chat 组件（核心）

```python
import streamlit as st

# 聊天输入框
prompt = st.chat_input("请输入您的问题: ")

# 用户消息
with st.chat_message('user'):
    st.write('Hello')

# AI助手消息
with st.chat_message('assistant'):
    st.write('Hello Human')
```

---

## 四、聊天机器人完整实现

### 4.1 后端工具模块（my_utils.py）

```python
import ollama

def get_response(messages):
    """向大模型发起请求，返回回复内容"""
    response = ollama.chat(
        model='deepseek-r1:8b',  # 或 qwen2.5:7b
        messages=messages[-50:]  # 取最近50条消息作为上下文
    )
    return response['message']['content']
```

### 4.2 前端主程序（my_chat.py）

```python
import streamlit as st
from my_utils import get_response

# 页面标题
st.title("黑马智聊机器人")

# 初始化会话状态（存储聊天记录）
if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {'role': 'assistant', 'content': '你好，我是黑马智聊机器人，有什么可以帮助你的么？'},
        {'role': 'user', 'content': '天空是什么颜色的？'},
        {'role': 'assistant', 'content': '天空是蓝色的。'},
    ]

# 渲染历史聊天记录
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# 聊天输入框
prompt = st.chat_input("请输入您要咨询的问题：")

if prompt:
    # 显示用户消息
    st.session_state['messages'].append({'role': 'user', 'content': prompt})
    st.chat_message("user").markdown(prompt)

    # 调用大模型获取回复
    with st.spinner("AI小助手正在思考中..."):
        content = get_response(st.session_state['messages'])

    # 显示AI回复
    st.session_state['messages'].append({'role': 'assistant', 'content': content})
    st.chat_message("assistant").markdown(content)
```

### 4.3 关键设计要点

1. **`st.session_state`**：Streamlit的会话状态管理，用于存储聊天记录，页面刷新不丢失
2. **`messages[-50:]`**：只取最近50条消息作为上下文，防止token超限
3. **`st.spinner()`**：显示加载动画，提升用户体验
4. **角色区分**：`user` 和 `assistant` 角色交替出现，保持对话上下文

### 4.4 运行方式

```shell
streamlit run my_chat.py
```

浏览器自动打开 `http://localhost:8501`，即可开始对话。

---

## 五、项目扩展思路

- 接入不同模型（Qwen/DeepSeek/Llama）进行效果对比
- 添加系统角色设定（System Prompt），定制AI人设
- 集成LangChain的Memory模块管理对话历史
- 添加知识库RAG功能，让机器人能回答特定领域问题
- 支持多轮对话中的Function Call（工具调用）
