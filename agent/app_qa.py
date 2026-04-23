import streamlit as st
from rag import RagService
import config_data as config

st.title("智能客服")
st.divider()


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "你好，我是智能客服助手，有什么能帮助您。"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

for msg in st.session_state["messages"]:
    role = msg["role"]
    content = msg["content"]
    st.chat_message(role).write(content)
    
    
prompt = st.chat_input("请输入您的问题...")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})
    ai_res_list = []
    res = st.session_state["rag"].chain.stream({"input": prompt}, config.session_config)
    def capture(generater,captured_list):
        for item in generater:
            captured_list.append(item)
            yield item
    st.chat_message("assistant").write(capture(res, ai_res_list))
    st.session_state["messages"].append({"role": "assistant", "content": "".join(ai_res_list)})