import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch
import os
import uuid
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="RickQuery", 
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for query history
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# Function to process PDF and create vector store
def process_pdf(pdf_file):
    os.makedirs('temp', exist_ok=True)
    
    temp_pdf_path = os.path.join('temp', f"{uuid.uuid4()}.pdf")
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_file.getbuffer())
    
    loader = PyPDFLoader(temp_pdf_path)
    docs = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    )
    
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    
    return vectorstore

# Load Advanced Question Answering Model
@st.cache_resource
def load_qa_model():
    try:
        # Use a more advanced model for better accuracy
        model_name = "google/flan-t5-large"
        
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        qa_pipeline = pipeline(
            "question-answering", 
            model=model, 
            tokenizer=tokenizer,
            max_seq_len=512
        )
        
        return qa_pipeline
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

# Enhanced Text Formatting Function
def format_answer(answer):
    # Split answer into paragraphs
    paragraphs = answer.split('\n')
    formatted_answer = ''
    
    for para in paragraphs:
        if para.strip():
            formatted_answer += f"<p style='text-align: justify; line-height: 1.6;'>{para.strip()}</p>"
    
    return formatted_answer

# Main Streamlit App
def main():
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main-title {
        color: #2c3e50;  /* Deep blue-gray color */
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .main-container {
        background-color: #f4f6f7;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    .query-input {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .answer-box {
        background-color: #e8f4f8;
        border-radius: 10px;
        padding: 20px;
        margin-top: 15px;
        border-left: 5px solid #3498db;
    }
    </style>
    """, unsafe_allow_html=True)

    # Main title with custom styling
    st.markdown(
        "<h1 class='main-title' style='text-align: center;'>🔍 RickQuery: Advanced PDF Insight Extractor</h1>", 
        unsafe_allow_html=True
    )

    # Sidebar configuration (previous implementation remains the same)
    st.sidebar.markdown("## 📤 PDF Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a PDF file", 
        type="pdf", 
        help="Upload a PDF to start querying"
    )

    # Main content area
    if uploaded_file is not None:
        with st.spinner('Processing PDF... (This might take a moment)'):
            try:
                # Process PDF
                vectorstore = process_pdf(uploaded_file)
                
                # Load Advanced QA Model
                qa_model = load_qa_model()
                
                if qa_model is not None:
                    # Query input with custom styling
                    st.markdown("<div class='query-input'>", unsafe_allow_html=True)
                    query = st.text_input(
                        "Enter your comprehensive question:", 
                        placeholder="Ask a detailed question about the PDF content"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Answer generation
                    if st.button("🔍 Extract Insights", type="primary"):
                        if query:
                            try:
                                # Enhanced context retrieval
                                docs = vectorstore.similarity_search(query, k=5)
                                context = " ".join([doc.page_content for doc in docs])
                                
                                # Generate answer with more context
                                result = qa_model({
                                    'question': query,
                                    'context': context
                                })
                                
                                # Format answer with paragraphs
                                formatted_answer = format_answer(result['answer'])
                                
                                # Display answer with enhanced styling
                                st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
                                st.markdown("### 💡 Extracted Insights:")
                                st.markdown(formatted_answer, unsafe_allow_html=True)
                                
                                # Confidence visualization
                                st.progress(result['score'])
                                st.markdown(f"**Insight Confidence:** {result['score']:.2%}", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                            except Exception as e:
                                st.error(f"Insight extraction error: {e}")
                        else:
                            st.warning("Please formulate a comprehensive question.")
                else:
                    st.error("Advanced model initialization failed.")
            
            except Exception as e:
                st.error(f"PDF processing complexity: {e}")
    
    else:
        # Attractive welcome message
        st.markdown("""
        <div class='main-container'>
        <h2 style='text-align: center; color: #2c3e50;'>Welcome to RickQuery 🚀</h2>
        <p style='text-align: center; color: #34495e;'>
        Unlock the power of your PDFs. Upload a document and dive deep into its contents!
        </p>
        </div>
        """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
