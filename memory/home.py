import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.globals import set_debug, set_verbose

set_verbose(True) # フォーマットされたプロンプトなどが表示される
set_debug(True)   # LangChainの挙動が最も詳細に出力される

st.title("OpenAI Chat")

system_prompt = """
あなたはAIエージェントです。
"""

def create_chain():
    chat = ChatOpenAI(
        model_name="gpt-3.5-turbo",
    )
    # OpenAI Functions AgentのプロンプトにMemoryの会話履歴を追加するための設定
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{input}"),
        ]
    )

    # OpenAI Functions Agentが使える設定でMemoryを初期化
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    chain = ConversationChain(
      llm=chat,
      prompt=prompt,
      memory=memory,
    )

    return chain

# 一度だけ初期化。セッションで保持しないと毎回初期化されてしまう
if "chain" not in st.session_state:
    st.session_state.chain = create_chain()

# 会話履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# 保持されている会話履歴を表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Message to chatbot"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # チャットの実行
    response = st.session_state.chain.invoke({"input": prompt})
    with st.chat_message("system"):
        st.markdown(response["response"])
    st.session_state.messages.append({"role": "assistant", "content": response["response"]})

