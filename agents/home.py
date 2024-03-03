import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.globals import set_debug, set_verbose

set_verbose(True) # フォーマットされたプロンプトなどが表示される
set_debug(True)   # LangChainの挙動が最も詳細に出力される

st.title("OpenAI Chat")

system_prompt = """
あなたはAIエージェントです。
"""

def create_agent():
    chat = ChatOpenAI(
        model_name="gpt-3.5-turbo",
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    search = DuckDuckGoSearchRun()
    tools = [
        Tool(
            name="duckduckgo-search",
            func=search.run,
            description="最新の情報を取得したい時に使うWeb検索ツールです。",
        )
    ]

    agent = create_openai_functions_agent(chat, tools, prompt)
    memory = ConversationBufferMemory(return_messages=True)

    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        memory=memory,
        verbose=True,
    )

# 一度だけ初期化。セッションで保持しないと毎回初期化されてしまう
if "agent" not in st.session_state:
    st.session_state.agent = create_agent()

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
    response = st.session_state.agent.invoke({"input": prompt})
    output = response["output"]
    with st.chat_message("system"):
        st.markdown(output)
    st.session_state.messages.append({"role": "assistant", "content": output})

