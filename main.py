import os
import json
import streamlit as st
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from vectorize_documents import embeddings

# Thiết lập thư mục làm việc và đọc dữ liệu cấu hình
working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Tạo một từ điển để lưu trữ lịch sử các phiên trò chuyện
session_store = {}

def get_session_history(session_id):
    """Trả về lịch sử trò chuyện cho một phiên nhất định."""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()  # Khởi tạo đối tượng ChatMessageHistory
    return session_store[session_id]

def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_db_dir"
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectorstore

def chat_chain(vectorstore):
    # Thiết lập template cho prompt
    qa_system_prompt = """You are MommyBaby, a virtual assistant that provides nutritional advice about milk for mothers and babies. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
    
    # Tạo retrieval chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    documents_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(retriever, documents_chain)
    
    # Tạo runnable với lịch sử thông điệp
    conservation_chain = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,  
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    return conservation_chain

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="MommyBaby",
    page_icon="📚",
    layout="centered"
)

st.title("📚 MommyBaby Chatbot")

# Khởi tạo trạng thái chat_history nếu chưa có
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Khởi tạo vectorstore nếu chưa có
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

# Khởi tạo conversational_chain nếu chưa có
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = chat_chain(st.session_state.vectorstore)

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

    # Gọi retrieval chain với lịch sử trò chuyện
    with st.chat_message("assistant"):
        response = st.session_state.retrieval_chain.invoke(
            {"input": user_input}, 
            config={"configurable": {"session_id": "default"}}  # Sử dụng session_id cho mỗi phiên
        )
        
        # Kiểm tra xem phản hồi có chứa câu trả lời không
        if 'answer' in response:
            assistant_response = response["answer"]
            st.markdown(assistant_response)
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
