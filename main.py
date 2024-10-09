import os
import json

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Thiết lập thư mục làm việc và đọc dữ liệu cấu hình
working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_db_dir"
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma(persist_directory=persist_directory,
                         embedding_function=embeddings)
    return vectorstore

def chat_chain(vectorstore):
    # Thiết lập template cho prompt
    prompt_template = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="MommyBaby là trợ lý ảo của bạn trong việc cung cấp thông tin về các sản phẩm sữa chất lượng.\n"
                 "Cuộc trò chuyện trước đây:\n{chat_history}\n"
                 "Người dùng: {question}\n"
                 "MommyBaby Assistant:"
    )

    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.3)
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )

    # Tạo một LLMChain với prompt template
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template
    )

    return chain  # Chỉ trả về chain

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Multi Doc Chat",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Multi Documents Chatbot")

# Khởi tạo trạng thái chat_history nếu chưa có
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Khởi tạo vectorstore nếu chưa có
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

# Khởi tạo conversational_chain nếu chưa có
if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = chat_chain(st.session_state.vectorstore)

# Hiển thị lịch sử trò chuyện
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Nhập câu hỏi từ người dùng
user_input = st.chat_input("Ask AI...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Định dạng lại lịch sử trò chuyện
    formatted_history = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.chat_history]
    )

    with st.chat_message("assistant"):
        # Gọi chain với lịch sử trò chuyện đã định dạng
        response = st.session_state.conversational_chain({"question": user_input, "chat_history": formatted_history})
        assistant_response = response["text"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
