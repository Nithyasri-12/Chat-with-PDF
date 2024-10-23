import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import google.generativeai as genai

# Initialize environment and configurations
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Please set your GOOGLE_API_KEY in the .env file")
    st.stop()

# Configure Google Generative AI with the API key
genai.configure(api_key=api_key)

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split the text into manageable chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50000,
        chunk_overlap=1000
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Create and save a FAISS vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Create a conversational chain with custom prompt template."""
    prompt_template = """
    Use the following conversation history and context to answer the question. 
    If the answer is not in the provided context, just say "Answer is not available in the context."
    
    Chat History: {chat_history}
    Context: {context}
    Question: {question}
    
    Please provide a detailed answer based on the context and previous conversation:
    """
    
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        google_api_key=api_key
    )
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "chat_history"]
    )
    
    chain = load_qa_chain(
        model,
        chain_type="stuff",
        prompt=prompt
    )
    
    return chain

def process_user_input(user_question):
    """Process user questions and generate responses."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        new_db = FAISS.load_local("faiss_index", embeddings)
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Format chat history for context
        chat_history = "\n".join(st.session_state.chat_history)
        
        response = chain(
            {
                "input_documents": docs,
                "question": user_question,
                "chat_history": chat_history
            }
        )
        
        # Extract the response text
        assistant_response = response["output_text"]
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        
        # Update conversation history for context
        st.session_state.chat_history.append(f"Human: {user_question}")
        st.session_state.chat_history.append(f"Assistant: {assistant_response}")
        
        return assistant_response
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def main():
    """Main application function."""
    # Page configuration
    st.set_page_config("Multi PDF Chatbot", page_icon=":scroll:", layout="wide")
    st.header("Multi-PDF's 📚 - Chat Agent 🤖")

    # Sidebar configuration
    with st.sidebar:
        st.image("img/Robot.jpg")
        st.write("---")
        
        st.subheader("📁 PDF File's Section")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files & Click on Process",
            accept_multiple_files=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Process PDFs"):
                if not pdf_docs:
                    st.warning("Please upload PDFs first!")
                    return
                    
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        with col2:
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.experimental_rerun()
        
        st.write("---")
        
        # Display chat history in sidebar
        if st.session_state.chat_history:
            st.subheader("💬 Chat History")
            st.text_area(
                "Recent Conversations",
                value="\n".join(st.session_state.chat_history),
                height=300,
                disabled=True
            )

    # Main chat interface
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input - Outside of any container/column structure
    if prompt := st.chat_input("Ask a question about your PDFs..."):
        if not os.path.exists("faiss_index"):
            st.warning("Please upload and process PDF files first!")
            st.stop()
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = process_user_input(prompt)
                if response:
                    st.markdown(response)

if __name__ == "__main__":
    main()