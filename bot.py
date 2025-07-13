import json
import torch
import faiss
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIGURATION
# -----------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
TOP_K = 3

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_resource
def load_contexts():
    with open("final.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return [entry["context"] for entry in data]

# -----------------------------
# EMBEDDING + FAISS INDEX
# -----------------------------
@st.cache_resource
def build_index(contexts):
    embed_model = SentenceTransformer(EMBED_MODEL)
    embeddings = embed_model.encode(contexts, convert_to_tensor=True, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.cpu().numpy())
    return embed_model, index, embeddings

# -----------------------------
# LOAD LLM (Mistral)
# -----------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def generate_prompt(question, contexts):
    joined = "\n---\n".join(contexts)
    return f"<s>[INST] Use the following college context to answer the question.\n\nContext:\n{joined}\n\nQuestion: {question} [/INST]"

def get_answer(prompt, tokenizer, model):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="College Chatbot", layout="centered")
st.title("🎓 College Chatbot")
st.write("Ask me anything about the college!")

question = st.text_input("You:", placeholder="e.g. What are the library timings?")

if question:
    # Load everything
    contexts = load_contexts()
    embed_model, index, embeddings = build_index(contexts)
    tokenizer, model = load_model()

    # RAG Retrieval
    q_embedding = embed_model.encode(question, convert_to_tensor=True).cpu().numpy()
    D, I = index.search(q_embedding.reshape(1, -1), k=TOP_K)
    retrieved = [contexts[i] for i in I[0]]

    # Generate answer
    prompt = generate_prompt(question, retrieved)
    with st.spinner("Thinking..."):
        answer = get_answer(prompt, tokenizer, model)

    st.markdown("**Bot:**")
    st.success(answer)
