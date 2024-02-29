import streamlit as st
import tempfile
from pathlib import Path
from langchain_openai import ChatOpenAI
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

index = st.session_state.get("index")


def on_change_file():
    if "index" in st.session_state:
        st.session_state.pop("index")


st.title("Q&A")

# PDFをアップロードする
pdf_file = st.file_uploader("PDFをアップロードしてください", type="pdf", on_change=on_change_file)

if pdf_file:
    with st.spinner(text="準備中..."):
        # ファイルを一時保存する
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(pdf_file.getbuffer())
            reader = SimpleDirectoryReader(input_files=[tmp.name])
            documents = reader.load_data()

            Settings.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            index = VectorStoreIndex.from_documents(documents=documents)

if index is not None:
    user_message = st.text_input(label="質問を入力してください")

    if user_message:
        with st.spinner(text="検索中..."):
            query_engine = index.as_query_engine()
            results = query_engine.query(user_message)
            st.write(results.response)
