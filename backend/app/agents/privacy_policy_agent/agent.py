# from openai.types import vector_store (Removed incorrect import)
from app.agents.kyc_agents import kyc_agent_results
from langchain_core.messages import SystemMessage,HumanMessage
from langgraph.graph import StateGraph, START, END

from app.config.privacy_policy_agent_config import CHUNK_OVERLAP
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
import os 
load_dotenv()

llm = ChatOpenAI(
    model="google/gemini-2.0-flash-001",
    temperature=0.7,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
    }
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma")
model_name = "BAAI/bge-small-en-v1.5"

# Lazy loaded — not initialized at import time
_embeddings = None
_vector_db = None
_retriever = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings(model_name=model_name)
    return _embeddings

def get_retriever():
    global _vector_db, _retriever
    if _retriever is not None:
        return _retriever
    try:
        embeddings = get_embeddings()
        if os.path.exists(PERSIST_DIR):
            print(f"Loading existing Vector DB from {PERSIST_DIR}...")
            _vector_db = Chroma(
                persist_directory=str(PERSIST_DIR),
                embedding_function=embeddings
            )
        else:
            pdf_path = os.path.join(BASE_DIR, "data", "privacy_policy_for_banking_simulation_app.pdf")
            if not os.path.exists(pdf_path):
                print(f"WARNING: PDF not found at {pdf_path}. Privacy agent will be unavailable.")
                return None
            else:
                print("No existing DB found. Ingesting PDF...")
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)
                _vector_db = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=PERSIST_DIR
                )
        if _vector_db is not None:
            _retriever = _vector_db.as_retriever(search_type='similarity', search_kwargs={'k': 4})
    except Exception as e:
        print(f"WARNING: Could not initialize vector DB: {e}. Privacy agent will be unavailable.")
    return _retriever


@tool
def rag_tool(query):
    """
    Retrieve relevant information from the pdf document.
    Use this tool when the user asks factual / conceptual questions
    that might be answered from the stored documents.
    """
    retriever = get_retriever()
    if retriever is None:
        return {
            'query': query,
            'context': ["Privacy policy document is not available at the moment."],
            'metadata': []
        }
    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]
    return {
        'query': query,
        'context': context,
        'metadata': metadata
    }


tools = [rag_tool]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState)->dict:
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

rag_bot = graph.compile()

def main():
    exit_list = ['exit', 'quit', 'thanks', 'thank you']
    print("Privacy Policy Agent is ready. Type 'exit' to quit.")
    while True:
        user_input = input("User: ").strip()
        if not user_input:
            continue
        if user_input.lower() in exit_list:
            print("Goodbye!")
            break
        result = rag_bot.invoke({
            "messages": [
                SystemMessage(
                    content="Use the pdf note, Explain if user wants, make concise if user wants, answer users question but must be related to pdf's content"
                ),
                HumanMessage(content=user_input)
            ]
        })
        print(f"AI: {result['messages'][-1].content}\n")

if __name__ == "__main__":
    main()