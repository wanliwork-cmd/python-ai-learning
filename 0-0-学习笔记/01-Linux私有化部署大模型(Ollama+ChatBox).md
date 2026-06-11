# 01 - 基于Linux平台的企业级私有化部署大模型

> 日期：2025年11月2日（下午+晚上）  
> 核心内容：私有化大模型概念、Ollama安装与使用、Ollama API、ChatBox搭建ChatBot

---

## 一、为什么要私有化大模型

企业使用公共大模型（如ChatGPT）存在 **数据隐私与安全风险**。典型案例：三星员工使用ChatGPT泄露芯片机密。

在金融、医疗、政府等行业，敏感数据不能外泄，因此需要 **私有大模型** —— 在企业自己的数据上训练，结果仅供内部使用。

### 主流私有化方案对比

| 特性 | Ollama | LM Studio |
|------|--------|-----------|
| **定位** | 开源的LLM本地运行框架 | 闭源的LLM工作站 |
| **特点** | 轻量、CLI操作、提供API、开源 | 集训练/部署/调试于一体、可视化 |
| **优势** | 开源+API支持，适合开发者 | 高度可定制、友好UI |
| **适用** | 学术研究、开发者原型设计 | 智能客服、NLP应用 |

**结论**：企业开发首选 **Ollama**（开源 + 提供API + 轻量级）。

---

## 二、Ollama 安装与使用

### 2.1 什么是Ollama

Ollama是一款开源软件，旨在简化大型语言模型的本地部署和运行。它将模型权重、配置和数据捆绑成一个Modelfile，支持热加载、多平台（Mac/Windows/Linux）、无复杂依赖、资源占用少。

- 官网：https://ollama.com/

### 2.2 安装方式

**Windows**：下载安装包 `OllamaSetup.exe`，以管理员身份运行安装。

**Linux 手动安装**：
```shell
# 解压到 /usr 目录
tar -xzf ollama-linux-amd64.tgz -C /usr

# 验证安装
ollama -v
```

**Linux 一键安装**（工作中推荐）：
```shell
curl -fsSL https://ollama.com/install.sh | sh
```

### 2.3 配置开机自启服务

创建服务文件 `/etc/systemd/system/ollama.service`：
```ini
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/bin/ollama serve
User=root
Group=root
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
```

生效并启动：
```shell
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama
```

### 2.4 运行第一个大模型

```shell
ollama run qwen2:0.5b
```

- 首次运行会自动下载模型（约352MB）
- 下载完成后直接进入对话模式
- 输入 `/bye` 退出对话

### 2.5 修改模型存储路径

默认路径：`~/.ollama/models`

修改为自定义路径，在 `/etc/profile` 中增加：
```shell
export OLLAMA_MODELS=/root/ollama
```

生效：
```shell
source /etc/profile
```

**持久化配置**：在 ollama.service 的 `[Service]` 段添加：
```ini
Environment="OLLAMA_MODELS=/root/ollama"
```

然后重启服务：
```shell
systemctl daemon-reload
systemctl restart ollama
```

---

## 三、Ollama 对话指令详解

### 3.1 对话内指令

| 指令 | 功能 |
|------|------|
| `/bye` | 退出对话（快捷键 Ctrl+D） |
| `/show info` | 查看模型基本信息（名称、大小、量化级别） |
| `/show license` | 查看许可信息 |
| `/show modelfile` | 查看模型源文件 |
| `/show parameters` | 查看内置参数 |
| `/show system` | 查看System角色设定 |
| `/show template` | 查看提示词模板 |
| `/clear` | 清除上下文记忆 |
| `/load <model>` | 对话中切换大模型 |
| `/save <name>` | 保存当前对话为新模型 |
| `/set system <string>` | 设置系统角色 |
| `/set format json` | 输出JSON格式 |
| `/set verbose` | 开启对话统计日志 |
| `/set parameter` | 设置模型参数 |
| `"""` | 多行输入模式 |

### 3.2 关键模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `temperature` | 温度值，越高越有创造性 | 0.8 |
| `top_k` | 多样性，越高答案越多样 | 40 |
| `top_p` | 保守度，越低越保守 | 0.9 |
| `num_predict` | 最大生成token数 | 128 |
| `num_ctx` | 上下文token大小 | 2048 |
| `seed` | 随机种子（固定种子=固定输出） | 0 |
| `repeat_penalty` | 重复惩罚强度 | 1.1 |
| `repeat_last_n` | 防重复回顾距离 | 64 |

### 3.3 终端快捷键

| 快捷键 | 功能 |
|--------|------|
| Ctrl + a/e | 移动到行头/行尾 |
| Ctrl + b/f | 移动到单词左边/右边 |
| Ctrl + k/u/w | 删除游标后/前/前一个单词 |
| Ctrl + c | 停止推理输出 |
| Ctrl + l | 清屏 |

---

## 四、Ollama 客户端命令

### 4.1 核心命令

```shell
# 运行模型（带提示词直接返回，不进入交互模式）
ollama run qwen2:0.5b "你好"

# 运行模型（带verbose统计）
ollama run qwen2:0.5b --verbose

# 查看模型信息（不运行模型）
ollama show qwen2:0.5b --template

# 从远程下载模型
ollama pull qwen2

# 查看本地模型列表
ollama list    # 或 ollama ls

# 查看当前运行中的模型
ollama ps

# 删除本地模型
ollama rm qwen2:0.5b
```

**注意事项**：
- `[:Version]` 不写则默认 `latest`
- Ollama的其他命令中不能使用模型ID，只能使用全名称
- `--keepalive` 参数可设置模型在内存中的存活时间

---

## 五、Ollama API 详解

### 5.1 开通远程访问

在 `/etc/profile` 中增加：
```shell
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_ORIGINS=*
```

在 `ollama.service` 的 `[Service]` 段添加：
```ini
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_ORIGINS=*"
```

开放防火墙端口：
```shell
firewall-cmd --zone=public --add-port=11434/tcp --permanent
firewall-cmd --reload
```

### 5.2 聊天对话接口（最重要）

**请求**：`POST /api/chat`

```json
{
  "model": "qwen2.5:0.5b",
  "messages": [
    {"role": "system", "content": "你是一个AI助手"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮你的？"},
    {"role": "user", "content": "介绍一下自己"}
  ],
  "stream": false,
  "options": {
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9
  }
}
```

**消息角色说明**：
- `system`：系统角色设定（定义AI行为方式）
- `user`：用户输入
- `assistant`：AI助手的历史回复

**返回结果关键字段**：
- `message.content`：AI回复内容
- `total_duration`：总耗时（纳秒）
- `prompt_eval_count`：提示词token消耗
- `eval_count`：响应token消耗
- `tool_calls`：工具调用（Function Call时使用）

### 5.3 向量化接口

**请求**：`POST /api/embeddings`

```json
{
  "model": "nomic-embed-text",
  "prompt": "要向量化的文本"
}
```

返回 `embedding` 数组，常用于RAG检索、模型微调等场景。

### 5.4 多模态对话

Ollama支持多模态模型（如LLaVA），可通过 `images` 字段传入Base64编码的图片：

```shell
ollama run llava --keepalive 1h
```

图片需要转为Base64字符串，通过 `images` 数组传入（支持多张图片）。

---

## 六、ChatBox 搭建 ChatBot

### 6.1 ChatBox 简介

ChatBox是一款AI对话工具，支持一键接入ChatGPT/Gemini/Claude/Ollama等多种平台，支持本地大模型、文档和图片聊天、代码生成与预览等功能。

### 6.2 集成 Ollama 步骤

1. **启动本地大模型**：
   ```shell
   ollama run qwen2 --keepalive 1h
   ```

2. **在ChatBox中配置**：
   - 模型提供商选择 "Ollama API"
   - API Host 填写 `http://127.0.0.1:11434`（或虚拟机IP）
   - 选择对应模型名称

3. **开始对话**：选择模型后即可与本地大模型聊天

### 6.3 ChatBox 支持的AI平台

- OpenAI (GPT系列)
- Google Gemini
- Anthropic Claude
- Ollama (本地大模型)
- ChatGLM (智谱)
- DeepSeek
- Qwen (通义千问)
- Moonshot AI (月之暗面)
- 01.AI (零一万物)
- 等等...

---

## 七、补充：Windows 安装 DeepSeek

```shell
# 下载模型
ollama pull deepseek-r1:1.5b

# 运行模型
ollama run deepseek-r1:1.5b
```

---

## 八、大模型分类总览

| 类型 | 用途 | 代表模型 |
|------|------|----------|
| **大语言模型** | 文本对话、生成 | Qwen、ChatGLM3、LLaMA3、DeepSeek |
| **文本嵌入模型** | 文本向量化 | text2vec、bge、nomic-embed-text |
| **重排模型** | 向量检索优化 | bge-reranker-large、bce-reranker |
| **多模态模型** | 图文理解 | Qwen-VL、LLaVA、DeepSeek-VL |
| **语音模型** | 语音识别/合成 | Whisper、VoiceCraft |
| **扩散模型** | 文生图/视频 | Stable Diffusion、AnimateDiff |

Ollama目前支持：大语言模型、文本嵌入模型、多模态模型。
