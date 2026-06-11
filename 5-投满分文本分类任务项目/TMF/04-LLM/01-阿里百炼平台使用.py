# pip install openai
from openai import OpenAI

# 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models

# 1- 创建大模型调用客户端对象
client = OpenAI(
    api_key = "sk-bc4978e2a7b4499395e05bc1e81e4af1",
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 2- 与大模型进行通信
result = client.chat.completions.create(
    model="qwen3.5-flash",
    messages=[
        {"role":"system","content":"你是一个文本分类专家，能够根据新闻标题进行分类"},
        {"role":"user","content":"新闻标题是：武汉樱花开了。请告诉我属于什么类型的新闻"}
    ]
)

print(result.model_dump_json())