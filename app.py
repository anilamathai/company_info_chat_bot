import os
import numpy as np
from google import genai
from sentence_transformers import SentenceTransformer
import streamlit as st
import re
from dotenv import load_dotenv

# --- CONFIGURATION & API KEY SETUP --- #

# Load .env file immediately. This makes the key available via os.getenv().
load_dotenv() 

# 1. Determine API Key (Local or Cloud)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Fallback to Streamlit Secrets (for Streamlit Cloud deployment)
if not GEMINI_API_KEY and "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Function to safely create the Gemini Client
@st.cache_resource
def create_gemini_client(api_key):
    if not api_key:
        # This will only be hit if the key is missing from BOTH sources
        raise ValueError("FATAL ERROR: GEMINI_API_KEY not found in Environment Variable or Streamlit Secrets.")

    return genai.Client(api_key=api_key) 

# Initialize the client outside the function, with a try/except 
client = None
try:
    client = create_gemini_client(GEMINI_API_KEY)
except ValueError as e:
    # Store the error message in session state, but allow the rest of the script to run
    st.session_state["gemini_client_error"] = str(e)
    client = None

# -------------------------------------------------------------
# 2. Embedding Model: Initialize the local Sentence Transformer model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)
embedding_model = get_embedding_model()

# 3. LLM Model Settings
MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.7
MAX_TOKENS = 1000

SYSTEM_PROMPT = (
    "You are a knowledgeable and professional assistant. "
    "You provide accurate, helpful, and concise information based on the provided knowledge base. "
    "Always base your answers on the provided knowledge base and maintain a polite, professional tone. "
    "If information is not available in the knowledge base, politely say so."
)

# ------------------ KNOWLEDGE BASE SETUP ------------------ #

# Use Streamlit caching to load and process knowledge only once
@st.cache_data
def process_knowledge(file_path="new_knowledge.txt", chunk_size=200):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            knowledge = f.read()
    except FileNotFoundError:
        st.error(f"Knowledge file '{file_path}' not found. Please make sure the file exists.")
        return [], None
    
    # Clean the text
    knowledge = re.sub(r'\s+', ' ', knowledge)  # Remove extra whitespace
    knowledge = knowledge.strip()
    
    # Split knowledge
    words = knowledge.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    
    # Filter out very short chunks
    chunks = [chunk for chunk in chunks if len(chunk.split()) > 10]
    
    if not chunks:
        st.error("No valid chunks created from the knowledge file.")
        return [], None
    
    # Generate and cache embeddings
    chunk_embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    
    return chunks, chunk_embeddings

KNOWLEDGE_CHUNKS, CHUNK_EMBEDDINGS = process_knowledge()

# ------------------ CHAT FUNCTIONS ------------------ #

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return np.dot(a, b) / (norm_a * norm_b)

def chat_with_rag(user_input):
    # CRITICAL CHECK: Ensure client is initialized before using it
    if client is None:
        return st.session_state.get("gemini_client_error", "[Error: Gemini Client not initialized.]")

    # Check if knowledge base is loaded
    if not KNOWLEDGE_CHUNKS or CHUNK_EMBEDDINGS is None:
        return "Knowledge base is not properly loaded. Please check your knowledge file."

    messages = st.session_state.messages
    
    # RAG Logic
    question_embedding = embedding_model.encode(user_input, convert_to_numpy=True)
    similarities = [cosine_similarity(question_embedding, chunk_emb) for chunk_emb in CHUNK_EMBEDDINGS]
    top_indices = np.argsort(similarities)[-3:]
    relevant_chunks = "\n\n".join([KNOWLEDGE_CHUNKS[i] for i in reversed(top_indices)])

    # UPDATED FORMAT 
    contextual_user_prompt = f"Knowledge Base (Relevant Context):\n{relevant_chunks}\n\nUser Question: {user_input}"
    
    # Prepare contents for the Gemini API call
    gemini_contents = []
    # Add history from session state (skipping the system prompt at index 0)
    for msg in messages[1:]:
        role = 'model' if msg['role'] == 'assistant' else 'user'
        gemini_contents.append({'role': role, 'parts': [{'text': msg['content']}]})

    # Add the current RAG-augmented user prompt
    gemini_contents.append({'role': 'user', 'parts': [{'text': contextual_user_prompt}]})
    
    # LLM Call
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=gemini_contents,
            config={
                "system_instruction": SYSTEM_PROMPT, 
                "temperature": TEMPERATURE,
                "max_output_tokens": MAX_TOKENS,
            }
        )
    except Exception as e:
        return f"[Error: Gemini API Call Failed] Details: {e}"

    # Get assistant reply, add to history
    reply = response.text
    
    # Add original user input and the reply to the session history for display
    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": reply})

    return reply

# ------------------ STREAMLIT UI IMPLEMENTATION ------------------ #

st.set_page_config(page_title="Custom RAG Chatbot", layout="wide")
st.title("🤖 Custom Knowledge Chatbot")
# REMOVED: st.caption(f"Powered by **{MODEL}** (via Gemini API) and **Sentence-Transformers** (Local Embeddings)")

# DISPLAY API KEY ERROR FIRST 
if client is None:
    st.error(st.session_state.get("gemini_client_error", "An unknown error occurred during client initialization."))
    st.stop()

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# Display conversation history
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask questions about the knowledge base..."):
    
    # 1. DISPLAY CURRENT USER PROMPT IMMEDIATELY
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner(f"Asking {MODEL}..."):
            response = chat_with_rag(prompt)
            st.markdown(response)

# Display system metrics in the sidebar
st.sidebar.header("System Metrics")
st.sidebar.metric("Max Response Length", f"{MAX_TOKENS} tokens") 
st.sidebar.markdown("---")

# REMOVED: "Setup Status" and "Sample Knowledge Chunks" section
# REMOVED: if CHUNK_EMBEDDINGS is not None and len(KNOWLEDGE_CHUNKS) > 0: ...

# Add information about the current knowledge base
st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Current Knowledge:**")
st.sidebar.info("This chatbot is trained on your custom dataset. Ask specific questions about the content.")

if st.sidebar.button("🔄 Clear Chat History"):
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.rerun()