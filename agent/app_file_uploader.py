import streamlit as st

from knowledge_base import KnowledgeBase

st.title("知识库更新服务")

uploader_file = st.file_uploader(
    "上传txt文件", 
    type=["txt"], 
    accept_multiple_files=False)

if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBase()
    
    
if uploader_file is not None:
    file_content = uploader_file.read().decode("utf-8")
    # st.text_area("文件内容", value=file_content, height=300)
    file_name = uploader_file.name
    text = uploader_file.getvalue().decode("utf-8")
    result = st.session_state["service"].upload_by_str(text, file_name)
    if result is not None:
        st.success(result)