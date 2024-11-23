import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from api.vectorize_documents import embeddings

app = FastAPI(
    title="MommyBaby Chatbot API",
    description="A chatbot API for providing nutritional advice for mothers and babies.",
    version="1.0.0"
)

# Read config file and set environment variables
working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"./config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# In-memory session storage
session_store = {}
vectorstore = None
retrieval_chain = None  # Lazy initialization


def get_session_history(session_id):
    """Return chat history for a specific session."""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


def setup_vectorstore():
    """Initialize vectorstore."""
    persist_directory = f"./vector_db_dir"
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectorstore


def chat_chain(vectorstore):
    """Set up chat chain for retrieval-based responses."""
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

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    documents_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(retriever, documents_chain)

    conservation_chain = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conservation_chain


def initialize_resources():
    """Lazy initialization of resources."""
    global vectorstore, retrieval_chain
    if vectorstore is None:
        vectorstore = setup_vectorstore()
    if retrieval_chain is None:
        retrieval_chain = chat_chain(vectorstore)


# Define the request model
class QueryRequest(BaseModel):
    session_id: str 
    user_input: str


@app.post("/chat/")
async def chat(request: QueryRequest):
    """Process user input and retrieve response."""
    initialize_resources()
    chat_history = get_session_history(request.session_id)
    chat_history.add_message({"role": "user", "content": request.user_input})

    response = retrieval_chain.invoke(
        {"input": request.user_input},
        config={"configurable": {"session_id": request.session_id}},
    )

    if 'answer' in response:
        assistant_response = response["answer"]
        chat_history.add_message({"role": "assistant", "content": assistant_response})
        return {"answer": assistant_response}
    return {"answer": "Sorry, I couldn't generate a response."}
