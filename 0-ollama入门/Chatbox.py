# -*- coding: utf-8 -*-
# @Time    : 2026/1/31 22:12
# @Author  : WanLi
# @File    : Chatbox.py

import streamlit as st
import ollama
import time

#初始化一个列表存储历史信息
if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.title("万家智能聊天机器人")

# 分割线
st.divider()

prompt =  st.chat_input("请输入您的问题: ")

if prompt:
    # 将用户出入的信息存到历史中
    st.session_state["messages"].append({"role":"user","content":prompt})
    for message in st.session_state["messages"]:
        st.chat_message(message["role"]).markdown(message["content"])

    with st.spinner("思考中..."):
        time.sleep(1)
        response = ollama.chat(model="deepseek-r1:8b", messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])
        st.session_state["messages"].append({"role":"assistant","content":response["message"]["content"]})
        st.chat_message("assistant").markdown(response["message"]["content"])