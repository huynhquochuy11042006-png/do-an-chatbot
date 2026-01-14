import google.generativeai as genai
genai.configure(api_key="AIzaSyC4G5cGpl2XCKmQkAcCl0IEt4tzp_lU3mk")

import streamlit as st
import json
import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Configure Gemini API key (thay bằng key thật của bạn từ Google AI Studio)
genai.configure(api_key="AIzaSyC4G5cGpl2XCKmQkAcCl0IEt4tzp_lU3mk")

# ================= CẤU HÌNH =================
JSON_FILE = "/content/drive/MyDrive/RAG/all_procedures_normalized.json"  # Đường dẫn file JSON (sau chunk rule-based)
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "dichvucong_rag"
GEMINI_MODEL = "gemini-2.5-flash"

@st.cache_resource
def get_embedding_function():
    EMBEDDING_MODEL = "BAAI/bge-m3"  # Model embedding tiếng Việt
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return embedding_function

@st.cache_resource
def load_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_func = get_embedding_function()

    try:
        collection = chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func  # cần để query đúng
        )
        st.success(f"Collection '{COLLECTION_NAME}' đã load từ {CHROMA_DB_PATH}")
    except Exception as e:
        st.error(f"Không tìm thấy collection '{COLLECTION_NAME}' trong {CHROMA_DB_PATH}: {e}")
        collection = None

    return collection
# --- Load collection 1 lần ---
collection = load_collection()

def query_rag(query: str, chat_history: list, top_k: int):
    # Retrieval với top_k động
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    context_parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context_parts.append(f"[{meta['hierarchy']}]\\n{doc}\\n(Nguồn: {meta['url']})")

    context = "\\n\\n".join(context_parts)

    prompt = f"""
    Bạn là trợ lý tư vấn thủ tục hành chính hỗ trợ cư trú tại Việt Nam. Trả lời ngắn gọn, chính xác, dễ hiểu, có dẫn nguồn. Chỉ trả lời dựa trên context, không thêm thông tin ngoài.

    Context:
    {context}

    Câu hỏi: {query}

    Trả lời bằng tiếng Việt, có đánh số nếu là danh sách, và trích dẫn nguồn rõ ràng (tên block, URL):
    """

    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt, stream=True)

    return response

# Giao diện chính
st.set_page_config(page_title="Chatbot hỗ trợ cư trú", layout="centered")
st.title("Chatbot hỗ trợ cư trú")

# Sidebar với top-k slider và thông tin
with st.sidebar:
    st.header("Cài đặt")
    top_k = st.slider("Top-k retrieval (số chunks lấy về)", min_value=1, max_value=10, value=3, step=1)
    st.header("Thông tin")
    st.write(f"Vector DB: {COLLECTION_NAME}")
    st.write(f"Số chunk: {collection.count() if collection else 0}")
    st.write(f"Model LLM: {GEMINI_MODEL}")
    st.write("Embedding: BAAI/bge-m3 (tối ưu tiếng Việt)")
    st.caption("Dữ liệu load từ file JSON. Nếu lỗi, kiểm tra đường dẫn JSON_FILE.")

# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input từ user
if prompt := st.chat_input("Hỏi về các thủ tục liên quan đến cư trú?"):
    # Thêm tin nhắn user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gọi RAG với top_k từ slider và stream response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            response = query_rag(prompt, st.session_state.messages, top_k)
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + " ")
            message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"Lỗi khi gọi Gemini: {str(e)}"
            message_placeholder.error(full_response)

    # Lưu response vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": full_response})
# Dán toàn bộ code trên vào đây
