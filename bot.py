import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import torch
import faiss
import json
import numpy as np

LLM_MODEL = "google/flan-t5-base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DATA_FILE = "final.json"
TOP_K = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_data
def load_contexts():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item["context"] for item in data if "context" in item]

@st.cache_resource
def build_faiss_index(contexts):
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedder.encode(contexts, convert_to_tensor=True, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.cpu().detach().numpy())
    return embedder, index, embeddings

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL).to(DEVICE)
    return tokenizer, model

def generate_answer(query, contexts, embedder, index, tokenizer, model):
    query_embedding = embedder.encode(query, convert_to_tensor=True).cpu().numpy()
    scores, retrieved_idxs = index.search(np.array([query_embedding]), k=TOP_K)
    retrieved_contexts = [contexts[i] for i in retrieved_idxs[0]]
    combined_context = "\n".join(retrieved_contexts)
    
    prompt = f"Answer the question based on the context.\n\nContext: {combined_context}\n\nQuestion: {query}"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=256)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# UI
st.set_page_config(page_title="🎓 College Chatbot", layout="centered")
st.title("🎓 College Chatbot")
st.markdown("Ask me anything about the college!")

user_input = st.text_input("You:", placeholder="e.g., What are library hours?")
if user_input:
    with st.spinner("Thinking..."):
        contexts = load_contexts()
        embedder, index, _ = build_faiss_index(contexts)
        tokenizer, model = load_model()
        response = generate_answer(user_input, contexts, embedder, index, tokenizer, model)
    st.markdown(f"**Bot:** {response}")
