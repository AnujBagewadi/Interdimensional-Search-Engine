import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch
import os

# Page Configuration
st.set_page_config(page_title="RickQuery", page_icon="📄")
st.title('PDF Question & Answering')

# Function to process PDF and create vector store
def process_pdf(pdf_file):
    # Create temp directory
    os.makedirs('temp', exist_ok=True)
    
    # Save PDF temporarily
    temp_pdf_path = os.path.join('temp', "uploaded.pdf")
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_file.getbuffer())
    
    # Load PDF
    loader = PyPDFLoader(temp_pdf_path)
    docs = loader.load()
    
    # Split text
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create vector store with FAISS
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    
    return vectorstore

# Load Question Answering Model
@st.cache_resource
def load_qa_model():
    try:
        # Load pre-trained QA model
        model_name = "deepset/roberta-base-squad2"
        
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create QA pipeline
        qa_pipeline = pipeline(
            "question-answering", 
            model=model, 
            tokenizer=tokenizer
        )
        
        return qa_pipeline
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

# Streamlit App
def main():
    # Sidebar for PDF upload
    st.sidebar.header("Upload PDF")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        try:
            # Process PDF
            vectorstore = process_pdf(uploaded_file)
            
            # Load QA Model
            qa_model = load_qa_model()
            
            if qa_model is not None:
                # Query input
                query = st.text_input("Enter your question about the PDF:")
                
                # Answer generation
                if st.button("Get Answer"):
                    if query:
                        try:
                            # Retrieve relevant context
                            docs = vectorstore.similarity_search(query, k=3)
                            context = " ".join([doc.page_content for doc in docs])
                            
                            # Generate answer
                            result = qa_model({
                                'question': query,
                                'context': context
                            })
                            
                            # Display answer
                            st.markdown("### Answer:")
                            st.write(result['answer'])
                            
                            # Optional: Show confidence score
                            st.markdown(f"**Confidence:** {result['score']:.2%}")
                        
                        except Exception as e:
                            st.error(f"Answer generation error: {e}")
                    else:
                        st.warning("Please enter a question.")
            else:
                st.error("Failed to load QA model.")
        
        except Exception as e:
            st.error(f"PDF processing error: {e}")
    
    else:
        st.info("Please upload a PDF file to get started.")

# Run the app
if __name__ == "__main__":
    main()
