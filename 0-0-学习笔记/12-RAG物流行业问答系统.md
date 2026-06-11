# 基于知识库 RAG 的物流行业信息问答系统

> 日期：2025年12月14日  
> 主题：RAG 原理、FAISS 向量数据库、文档加载与分段、知识库问答、Streamlit Web 界面

---

## 一、项目背景

大模型基于历史数据训练，**无法获取最新知识和企业私有知识**。解决方案有两种：
1. 基于私有知识微调大模型
2. **RAG（检索增强生成）**：基于向量数据库 + LLM 搭建本地知识库问答

本项目以物流行业为例，构建 RAG 系统实现本地知识库问答。

---

## 二、RAG 原理

```
文档上传 → 文本分段 → Embedding 向量化 → 存入向量数据库
                                                  ↓
用户提问 → 向量检索相似文档 → 召回 Top-K 片段 → 拼接 Prompt → LLM 生成回答
```

**核心思想**：先从知识库中检索相关内容，再将检索结果作为上下文提供给 LLM 生成回答。

---

## 三、环境配置

```bash
pip install faiss-cpu      # 向量数据库
pip install langchain       # LLM 框架
pip install langchain_community
pip install ollama          # 本地模型管理
pip install streamlit       # Web 界面
pip install PyMuPDF         # PDF 文档加载
```

---

## 四、代码实现

### 4.1 本地知识库构建

将 PDF 文档加载 → 分段 → 向量化 → 存入 FAISS：

```python
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def build_knowledge_base():
    # 1. 加载 PDF 文档
    loader = PyMuPDFLoader("物流信息.pdf")
    data = loader.load()

    # 2. 文本分段（chunk_size=50, chunk_overlap=20）
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20)
    split_docs = text_splitter.split_documents(data)

    # 3. 初始化 Embedding 模型
    embeddings = OllamaEmbeddings(model="bge-m3", temperature=0)

    # 4. 向量化并存入 FAISS
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local("./faiss/wuliu")
```

**关键参数说明**：
- `chunk_size=50`：每段 50 个字符
- `chunk_overlap=20`：相邻段重叠 20 个字符，确保语义连贯
- `model="bge-m3"`：Embedding 模型，将文本转为向量

### 4.2 构建本地问答 RAG 系统

```python
from langchain import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# 加载向量库
embeddings = OllamaEmbeddings(model="bge-m3", temperature=0)
db = FAISS.load_local("faiss/wuliu", embeddings, allow_dangerous_deserialization=True)

def get_related_content(related_docs):
    """从检索到的文档中提取内容"""
    return "\n".join([doc.page_content.replace("\n\n", "\n") for doc in related_docs])

def qa(question):
    # 1. 向量检索相似文档（Top-K=2）
    docs = db.similarity_search(question, k=2)
    related_content = get_related_content(docs)

    # 2. 构建 RAG Prompt
    PROMPT_TEMPLATE = """
    基于以下已知信息，简洁和专业地回答用户的问题。不允许编造。
    已知内容: {context}
    问题: {question}"""
    prompt = PromptTemplate(input_variables=["context", "question"], template=PROMPT_TEMPLATE)
    my_prompt = prompt.format(context=related_content, question=question)

    # 3. 调用 LLM 生成回答
    model = Ollama(model="qwen2.5:7b")
    return model.invoke(my_prompt)

result = qa("我的快递出发地是哪？预计几天到达？")
```

### 4.3 Streamlit Web 界面

```python
import streamlit as st
from langchain.chains import ConversationalRetrievalChain

st.set_page_config(page_title="物流行业信息咨询系统")
st.title("物流行业信息咨询系统")

# 创建对话检索链（支持多轮对话上下文）
chain = ConversationalRetrievalChain.from_llm(
    llm=Ollama(model="qwen2.5:7b"),
    retriever=db.as_retriever()
)

# 会话状态管理
if "messages" not in st.session_state:
    st.session_state.messages = []

# 展示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("请输入你的问题:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        result = chain.invoke({"question": prompt, "chat_history": chat_history})
        st.markdown(result["answer"])
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
```

---

## 五、RAG 系统优化要点

| 优化方向 | 方法 |
|---------|------|
| 分段策略 | 调整 chunk_size 和 chunk_overlap |
| Embedding 模型 | 选择更好的向量化模型（如 bge-m3） |
| 检索策略 | 混合检索（向量 + 全文） |
| Prompt 设计 | 明确"不允许编造"等约束 |
| 多轮对话 | 使用 ConversationalRetrievalChain |
| 重排序 | 对检索结果进行相关性重排序 |
