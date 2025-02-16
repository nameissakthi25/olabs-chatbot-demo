import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

# Set USER_AGENT
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def load_and_process_urls(urls):
    # Load content from URLs
    loader = WebBaseLoader(urls)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore

def create_chain(vectorstore):
    # Initialize Ollama LLM
    llm = Ollama(model="llama3.2:1b")
    
    # Create memory with updated parameters
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Create chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": None}
    )
    
    return chain

def main():
    st.title("Context-Aware Chatbot")
    
    # URL input section
    urls = st.text_area(
        "Enter URLs (one per line) containing theory, procedure, and resources:",
        height=100
    )
    
    if st.button("Load Content"):
        if urls:
            url_list = [url.strip() for url in urls.split('\n') if url.strip()]
            with st.spinner("Loading and processing content..."):
                vectorstore = load_and_process_urls(url_list)
                st.session_state.chain = create_chain(vectorstore)
                st.success("Content loaded successfully!")
    
    # Chat interface
    if "chain" in st.session_state:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the loaded content"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chain({"question": prompt})
                    response_text = response["answer"]
                    st.markdown(response_text)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text}
                    )

if __name__ == "__main__":
    main() 