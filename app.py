import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 1. CONFIGURATION AND SECRETS HANDLING ---

# Set Streamlit page config
st.set_page_config(
    page_title="Gemini RAG Chatbot 💬",
    page_icon="🤖",
    layout="wide"
)

st.title("Gemini Pro RAG Assistant 🤖")
st.subheader("Chat with your PDFs using Google Gemini and Streamlit")

# Securely load the API key from Streamlit secrets (for deployment)
# or from environment variables (for local testing/manual setup)
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    # Fallback to os.getenv (for local testing with .env or manual setup)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("FATAL ERROR: GEMINI_API_KEY not found in Streamlit Secrets or Environment Variable. Please configure the key in the Streamlit App Settings.")
    st.stop()
    
# --- 2. CORE RAG FUNCTIONS (CACHED) ---

# Use st.cache_resource to cache resource-heavy operations
@st.cache_resource
def get_vector_store(pdf_docs):
    """Processes PDF documents, creates embeddings, and builds a FAISS vector store."""
    st.info("Processing document and generating knowledge base...")
    
    # Load documents from the uploaded files
    all_text = ""
    for pdf_file in pdf_docs:
        # Save uploaded file to a temporary location to be read by PyPDFLoader
        with open(f"./temp_{pdf_file.name}", "wb") as f:
            f.write(pdf_file.getbuffer())
        loader = PyPDFLoader(f"./temp_{pdf_file.name}")
        pages = loader.load()
        all_text += "\n".join(p.page_content for p in pages)
        os.remove(f"./temp_{pdf_file.name}") # Clean up temp file

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(all_text)
    
    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GEMINI_API_KEY)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    st.success("Knowledge Base Created! You can now chat.")
    return vector_store

@st.cache_resource
def get_conversation_chain(vector_store):
    """Creates the LangChain Conversational Retrieval Chain."""
    # Use Gemini Pro for the main model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.3,
        api_key=GEMINI_API_KEY
    )
    
    # Create the RAG chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    return conversation_chain

# --- 3. SESSION STATE MANAGEMENT AND UI ---

def initialize_session_state():
    """Initializes message history and chain/vector store objects."""
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

def handle_user_input(user_question):
    """Processes user input, runs the RAG chain, and updates the chat history."""
    if st.session_state.conversation_chain is None:
        st.warning("Please upload and process documents first.")
        return

    # Call the RAG chain
    with st.spinner("Generating response..."):
        try:
            response = st.session_state.conversation_chain.invoke(
                {"question": user_question, "chat_history": st.session_state.chat_history}
            )
        except Exception as e:
            st.error(f"An API error occurred: {e}")
            return
            
    # Update chat history
    st.session_state.chat_history.append((user_question, response["answer"]))
    
    # Display the result (The display loop is separate, below)

def display_chat_history():
    """Displays all messages in the chat history."""
    # Display newest messages first (reverse order)
    for question, answer in reversed(st.session_state.chat_history):
        # Assistant Message
        with st.chat_message("assistant"):
            st.markdown(answer)
        
        # User Message
        with st.chat_message("user"):
            st.markdown(question)

# --- 4. STREAMLIT LAYOUT ---

initialize_session_state()

# Sidebar for file upload
with st.sidebar:
    st.header("Your Documents")
    
    # File uploader allows multiple PDFs
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click 'Process'",
        accept_multiple_files=True,
        type=['pdf']
    )
    
    if st.button("Process Documents"):
        if pdf_docs:
            st.session_state.vector_store = get_vector_store(pdf_docs)
            st.session_state.conversation_chain = get_conversation_chain(st.session_state.vector_store)
            st.session_state.chat_history = [] # Clear history on new document upload
        else:
            st.warning("Please upload at least one PDF file.")

# Main Chat Interface
if st.session_state.vector_store:
    user_question = st.chat_input("Ask a question about your documents...")
    
    if user_question:
        handle_user_input(user_question)

display_chat_history()
