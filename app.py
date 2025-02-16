import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import speech_recognition as sr
import os

# Set USER_AGENT
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# Predefined URLs
URLS = [
    "https://www.olabs.edu.in/?sub=73&brch=8&sim=34&cnt=1",  # theory
    "https://www.olabs.edu.in/?sub=73&brch=8&sim=34&cnt=2",  # procedure
    "https://www.olabs.edu.in/?sub=73&brch=8&sim=34&cnt=345" # resources
]

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def listen_to_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Speak your question")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return text
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return None

def load_and_process_urls(urls):
    try:
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
    except Exception as e:
        st.error(f"Error loading content: {str(e)}")
        return None

def create_chain(vectorstore):
    try:
        # Initialize Ollama LLM
        llm = Ollama(model="llama3.2:1b")
        
        # Create memory
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
    except Exception as e:
        st.error(f"Error creating chain: {str(e)}")
        return None

def initialize_chain():
    if not st.session_state.initialized:
        with st.spinner("Loading content..."):
            vectorstore = load_and_process_urls(URLS)
            if vectorstore:
                chain = create_chain(vectorstore)
                if chain:
                    st.session_state.chain = chain
                    st.session_state.initialized = True
                    st.success("Content loaded successfully!")
                    return True
    return st.session_state.initialized

def main():
    st.title("Context-Aware Chatbot")
    
    # Initialize the chain
    if not initialize_chain():
        st.error("Failed to initialize the chatbot. Please refresh the page.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Voice input button
    if st.button("🎤 Speak Question"):
        question = listen_to_speech()
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chain({"question": question})
                    response_text = response["answer"]
                    st.markdown(response_text)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text}
                    )
    
    # Text input
    if prompt := st.chat_input("Type your question here"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chain({"question": prompt})
                response_text = response["answer"]
                st.markdown(response_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text}
                )

if __name__ == "__main__":
    main() 