from openai import OpenAI

client = OpenAI(
    api_key="sk-9f1b3a446289459ea6cf8ca57ccbb604",
    base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个人工智能教授，会写代码，能够讲清楚理论"},
        {"role": "user", "content": "请解释Transformer的架构原理"},
    ],
    stream=False
)

print(response.choices[0].message.content)