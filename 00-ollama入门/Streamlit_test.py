# -*- coding: utf-8 -*-
# @Time    : 2026/1/31 12:23
# @Author  : WanLi
# @File    : Streamlit_test.py
import time

# streamlit api使用

import streamlit as st

#设置标题内容
st.title("streamlit_api_test")

#段落
st.write('这是段落内容')

#分割线
st.divider()


#聊天输入框
input_message =  st.chat_input("请输入内容")

#消息容器
#chat_message需要传入角色: 可用"assistant", "user", "ai",'human'
#每一个角色对应不同的聊天背景色和图标
st.chat_message('user').markdown('hello')

st.chat_message('assistant').markdown('hello')

st.chat_message('ai').markdown('hello')

st.chat_message('human').markdown('hello')

#等待提示框
with st.spinner('思考中...'):
    time.sleep(5)  #休眠5秒
    st.write('思考完毕...')
    if input_message:
        st.write(input_message)

