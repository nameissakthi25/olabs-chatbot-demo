# Context-Aware Chatbot

A Streamlit-based chatbot that answers questions based on provided web content using LangChain and Ollama. The chatbot strictly responds to questions related to the content from the URLs you provide.

## Features

- Load content from multiple URLs
- Process and chunk content for efficient retrieval
- Context-aware responses using LLM
- Clean chat interface
- Conversation memory
- Strictly answers based on provided content

## Prerequisites

- Python 3.8 or higher
- Ollama
- Windows/Linux/MacOS

## Installation Guide

### 1. Install Ollama

Visit [Ollama's official website](https://ollama.ai/download) and download the appropriate version for your operating system:

- **Windows**: 
  - Download and install Windows version
  - You'll need WSL (Windows Subsystem for Linux)
  
- **MacOS**: 
  - Download and install Mac version
  
- **Linux**:
  ```bash
  curl -fsSL https://ollama.ai/install.sh | sh
  ```

### 2. Setup Project

1. Create a new directory for your project:
   ```bash
   mkdir chatbot-project
   cd chatbot-project
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/MacOS
   python -m venv venv
   source venv/bin/activate
   ```

3. Create the required files:
   - Create `app.py` and copy the provided code
   - Create `requirements.txt` and copy the dependencies

4. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Start Ollama service:
   ```bash
   ollama serve
   ```

2. Pull the Llama2 model (in a new terminal):
   ```bash
   ollama pull llama2
   ```

3. Run the Streamlit app (in a new terminal):
   ```bash
   streamlit run app.py
   ```

4. Open your browser and navigate to `http://localhost:8501`

## Usage

1. Enter URLs containing the content you want the chatbot to learn from (one URL per line)
2. Click "Load Content" and wait for the processing to complete
3. Start asking questions about the loaded content in the chat interface
4. The chatbot will respond based only on the information from the provided URLs

## Troubleshooting

### Common Issues and Solutions

1. **Ollama Connection Error**
   - Ensure Ollama service is running (`ollama serve`)
   - Check if the model is pulled (`ollama pull llama2`)

2. **Import Errors**
   - Verify all dependencies are installed (`pip install -r requirements.txt`)
   - Make sure you're using the virtual environment

3. **URL Loading Issues**
   - Check your internet connection
   - Verify the URLs are accessible
   - Ensure URLs are properly formatted

## System Requirements

- Minimum 8GB RAM (16GB recommended)
- At least 5GB free storage space
- Internet connection for URL loading and initial model download

## Limitations

- Only responds to questions about loaded content
- URL content must be publicly accessible
- Processing large amounts of content may take time 