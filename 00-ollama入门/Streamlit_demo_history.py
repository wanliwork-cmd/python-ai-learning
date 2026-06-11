# -*- coding: utf-8 -*-
# @Time    : 2026/1/31 21:07
# @Author  : WanLi
# @File    : Streamlit_demo_history.py

#加上历史记录

import streamlit as st
import time


st.title('带历史记录')

st.divider()

if "count" not in st.session_state:
    st.session_state.count = 1

if "messages" not in st.session_state:
    st.session_state.messages = []

prompt =  st.chat_input("请输入您的问题: ")

if prompt:
    # 用户输入的信息追加到列表中
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    with st.spinner("正在思考..."):
        time.sleep(1)
        response = f"这是一个测试响应:{st.session_state.count}"
        # 将ai返回的信息追加到列表中
        st.session_state.messages.append({'role': 'assistant', 'content': response})
        st.session_state.count = st.session_state.count + 1
        st.chat_message("assistant").markdown(response)

