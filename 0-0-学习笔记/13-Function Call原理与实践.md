# Function Call 原理与实践

> 日期：2025年12月15日~18日  
> 主题：Function Call 原理、单函数调用、多函数调用、智谱 GLM-4 实现、工具定义与解析

---

## 一、Function Call 概述

### 1.1 什么是 Function Call

Function Call（函数调用）是大模型的一种能力，允许模型**自主决定调用外部工具/函数**来获取信息或执行操作，而不是仅凭自身知识回答。

### 1.2 核心流程

```
用户提问 → 模型分析 → 决定调用函数 → 返回函数名和参数
    → 程序执行函数 → 将结果返回模型 → 模型生成最终回答
```

**两次调用模型**：
1. **第一次调用**：模型分析用户问题，决定是否需要调用工具，返回工具名和参数
2. **执行工具**：程序根据模型指示执行对应函数
3. **第二次调用**：将工具执行结果返回模型，模型基于结果生成自然语言回答

### 1.3 tool_choice 策略

| 策略 | 说明 |
|------|------|
| `"auto"` | 模型自动决定是否调用工具（默认） |
| `"none"` | 不调用任何工具，只生成文本 |
| `{"type":"function","function":{"name":"xxx"}}` | 强制指定调用某个函数 |

---

## 二、单函数调用实现

### 2.1 定义工具（tools）

```python
import json
import requests

# 工具描述列表
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取指定城市的当前天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称，如'北京'"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# 实际的天气查询函数
def get_current_weather(location):
    """调用天气 API 获取天气信息"""
    # 这里调用实际的天气 API
    weather_data = {
        "location": location,
        "high_temperature": "高温 29℃",
        "low_temperature": "低温 18℃",
        "type": "晴"
    }
    return json.dumps(weather_data, ensure_ascii=False)
```

### 2.2 解析模型响应

```python
def parse_response(response):
    """解析模型的 tool_calls 响应，执行对应函数"""
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)

    # 根据函数名执行对应操作
    if function_name == "get_current_weather":
        result = get_current_weather(**function_args)
    return result
```

### 2.3 完整的 Function Call 流程

```python
from zhipuai import ZhipuAI

client = ZhipuAI(api_key="your_api_key")

def function_call_demo(user_input):
    messages = [
        {"role": "system", "content": "你是天气查询助手，根据地址回答天气情况"},
        {"role": "user", "content": user_input}
    ]

    # 第一次调用：模型决定调用哪个工具
    response = client.chat.completions.create(
        model="glm-4",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    # 检查模型是否要调用工具
    if response.choices[0].message.content is not None:
        # 直接回答了，不需要工具
        print(response.choices[0].message.content)
    else:
        # 模型要求调用工具
        # 1. 解析函数调用结果
        function_response = parse_response(response)

        # 2. 将模型的 tool_calls 消息加入历史
        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump())

        # 3. 将工具执行结果加入历史
        function_name = response.choices[0].message.tool_calls[0].function.name
        function_id = response.choices[0].message.tool_calls[0].id
        messages.append({
            "role": "tool",
            "tool_call_id": function_id,
            "name": function_name,
            "content": function_response,
        })

        # 4. 第二次调用：模型基于工具结果生成回答
        last_response = client.chat.completions.create(
            model="glm-4",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        print(last_response.choices[0].message.content)
```

---

## 三、多函数调用

### 3.1 定义多个工具

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取指定城市的当前天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "城市名称"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_flight_info",
            "description": "查询航班信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_no": {"type": "string", "description": "航班号"}
                },
                "required": ["flight_no"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_flight_price",
            "description": "查询航班价格",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_no": {"type": "string", "description": "航班号"}
                },
                "required": ["flight_no"]
            }
        }
    }
]
```

### 3.2 模型自主选择工具

当提供多个工具时，模型会根据用户问题**自动选择**合适的工具：

```python
# 问天气 → 模型调用 get_current_weather
main("查询深圳的天气怎么样?")

# 问航班 → 模型调用 get_flight_info
main("查询北京到深圳的航班号?")

# 问价格 → 模型调用 get_flight_price
main("查询BJSZ123航班号对应的价格?")

# 问笑话 → 模型不调用任何工具，直接回答
main("给我讲一个笑话?")
```

---

## 四、Function Call 与数据库集成

### 4.1 通过 Function Call 查询 MySQL

```python
import pymysql

# 定义 SQL 查询工具
sql_tools = [
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "查询数据库中的用户信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQL 查询语句"}
                },
                "required": ["sql"]
            }
        }
    }
]

def query_database(sql):
    """执行 SQL 查询"""
    conn = pymysql.connect(host='localhost', user='root', password='xxx', db='test')
    cursor = conn.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()
    conn.close()
    return json.dumps(results, ensure_ascii=False)
```

---

## 五、Function Call 关键总结

### 5.1 消息角色说明

| role | 说明 |
|------|------|
| `system` | 系统指令，定义助手角色 |
| `user` | 用户输入 |
| `assistant` | 模型的第一次响应（包含 tool_calls） |
| `tool` | 工具执行结果（包含 tool_call_id） |

### 5.2 应用场景

| 场景 | 工具示例 |
|------|---------|
| 天气查询 | 调用天气 API |
| 航班/酒店查询 | 调用预订系统 API |
| 数据库查询 | 执行 SQL |
| 数学计算 | 内置 math 工具 |
| 网络搜索 | 搜索引擎 API |
| 文件操作 | 读写文件函数 |
