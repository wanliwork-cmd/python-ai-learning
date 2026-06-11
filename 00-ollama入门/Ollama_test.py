# -*- coding: utf-8 -*-
# @Time    : 2026/1/30 23:53
# @Author  : WanLi
# @File    : Ollama_test.py

import subprocess
import sys
import os
import ollama

model = "deepseek-r1:8b"
def main():
    """主函数"""
    # 打印版本信息
    print(f"Ollama 版本: {ollama.list()}")

    # 你的主要业务逻辑

    response = ollama.chat(model=model, messages=[
        {
            'role': 'user',
            'content': '你是谁?',
        },
    ])
    print(response['message']['content'])


def main1():
    client = ollama.Client(host="http://127.0.0.1:11434")
    print(client.list())
    print(client.show(model))
    print(client.ps())

    while True:
        prompt = input('请输入问题:')
        #chat 和模型对话
        response = client.chat(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ]
        )
        print(response['message']['content'])

if __name__ == "__main__":
    main1()