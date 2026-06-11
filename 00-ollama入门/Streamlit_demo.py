# -*- coding: utf-8 -*-
# @Time    : 2026/1/31 12:46
# @Author  : WanLi
# @File    : Streamlit_demo.py

import streamlit as st
import time

if "count" not in st.session_state:
    st.session_state['count'] = 1
# 给与一个标题
st.title('测试标题')

# 分割线
st.divider()

#消息提示框
prompt = st.chat_input("请输入您的问题:")

# 构建消息容器
if prompt:
    # user assistant
    with st.chat_message("user"):
        st.write(prompt)

    #ai回答
    st.spinner("思考中...")
    time.sleep(1)
    st.chat_message('assistant').markdown(f"这是ai的回答:  {st.session_state['count']}")
    st.session_state['count'] = st.session_state['count'] + 1