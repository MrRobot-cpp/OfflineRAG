# 🤖 OfflineRAG System

A powerful offline Retrieval-Augmented Generation (RAG) system that allows you to chat with your PDF documents using local AI models.

## ✨ Features

- **📄 PDF Processing**: Upload and process PDF documents with intelligent text extraction
- **🔍 Vector Search**: Advanced FAISS-based semantic search for relevant content retrieval
- **🤖 Local AI**: Powered by Llama 3 through Ollama for complete privacy
- **💬 Interactive Chat**: Beautiful web interface for asking questions about your documents
- **🔒 Fully Offline**: No data leaves your computer - complete privacy protection
- **⚡ Fast & Efficient**: Optimized chunking and vector indexing for quick responses

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed on your system
2. **Ollama** installed and running with Llama 3 model

### Installation

1. **Clone or download** this repository
2. **Open terminal/command prompt** in the project directory
3. **Run the setup**:

```bash
# Windows (PowerShell)
.\run_rag.ps1

# Windows (Command Prompt)
run_rag.bat

# Manual setup
venv\Scripts\activate
pip install -r requirements.txt
python start_rag.py
```

### First Run

1. **Start the system** using one of the methods above
2. **Open your browser** and go to `http://localhost:5006`
3. **Upload a PDF** document using the file input
4. **Click "Load PDF"** to process the document
5. **Ask questions** about your document in the chat interface

## 📖 How to Use

### 1. Document Setup
- Click "Choose PDF File" to select your document
- Click "🚀 Load PDF" to process and index the document
- Wait for the "Loaded" status message

### 2. Asking Questions
- Type your question in the text input field
- Click "🔍 Ask AI" to get an AI-powered answer
- The system will search through your document and provide relevant answers

### 3. Managing Chat
- Use "🗑️ Clear" to clear the conversation history
- Upload new documents anytime to switch context

## 🔧 Configuration

### Ollama Setup

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Pull Llama 3 model**:
   ```bash
   ollama pull llama3
   ```
3. **Verify installation**:
   ```bash
   ollama list
   ```

### Customization

You can modify the following in `scripts/offline_rag.py`:

- **Chunk size**: Change `chunk_size` parameter in `load_pdf_chunks()`
- **Model**: Change `model` parameter in `LocalLlama()` class
- **UI Theme**: Modify the `THEME` dictionary
- **Retrieval count**: Change `k` parameter in `retrieve_top_k()`

## 🛠️ Troubleshooting

### Common Issues

**"Panel not found" error**:
- Ensure virtual environment is activated
- Run `pip install panel` in the venv

**"Ollama not found" error**:
- Install Ollama and ensure it's in your PATH
- Or update the `ollama_path` in the `LocalLlama` class

**"FAISS import error"**:
- Run `pip install faiss-cpu` in the virtual environment

**PDF processing fails**:
- Ensure the PDF contains readable text (not just images)
- Try with a different PDF file

**UI not responsive**:
- Check browser console for errors
- Ensure all dependencies are installed correctly

### Getting Help

1. Check the status messages in the web interface
2. Look at the terminal output for error details
3. Ensure all prerequisites are properly installed

## 📁 Project Structure

```
OfflineRAG/
├── scripts/
│   └── offline_rag.py      # Main application code
├── venv/                   # Virtual environment
├── requirements.txt        # Python dependencies
├── start_rag.py           # Startup script
├── run_rag.bat            # Windows batch file
├── run_rag.ps1            # PowerShell script
└── README.md              # This file
```

## 🔒 Privacy & Security

- **100% Offline**: All processing happens on your local machine
- **No Data Collection**: No information is sent to external servers
- **Local AI**: Uses your local Llama 3 model through Ollama
- **Secure**: Your documents never leave your computer

## 🎯 Use Cases

- **Research**: Ask questions about research papers and documents
- **Legal**: Query legal documents and contracts
- **Education**: Study materials and textbooks
- **Business**: Analyze reports and documentation
- **Personal**: Chat with your personal documents

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

---

**Made with ❤️ for the privacy-conscious AI community**
