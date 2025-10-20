# Offline RAG System - CLI Version

A command-line interface for the Offline RAG system that uses Llama 3, FAISS, and MiniLM for document question-answering without any UI dependencies.

## Features

- **PDF Processing**: Load and process PDF documents into searchable chunks
- **FAISS Vector Search**: Create and manage FAISS indices for fast similarity search
- **Llama 3 Integration**: Use local Llama 3 model via Ollama for AI responses
- **Index Management**: Save, load, and delete document indices
- **Command Line Interface**: Simple CLI for all operations

## Prerequisites

1. **Python 3.8+**
2. **Ollama**: Install from [ollama.ai](https://ollama.ai)
3. **Llama 3 Model**: Pull the model with `ollama pull llama3`

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Start the CLI
```bash
python scripts/offline_rag_cli.py
```

### Available Commands

1. **Load PDF file**: Process a PDF document and create a searchable index
2. **Load existing index**: Load a previously created index
3. **Ask question**: Ask questions about the loaded document
4. **List indices**: Show all available indices
5. **Delete index**: Remove an index
6. **Show status**: Display current system status
7. **Exit**: Quit the application

### Example Workflow

1. Start the CLI: `python scripts/offline_rag_cli.py`
2. Choose option 1 to load a PDF file
3. Enter the path to your PDF file
4. Wait for processing to complete
5. Choose option 3 to ask questions
6. Enter your question about the document
7. Get AI-powered answers based on the document content

## Testing

Run the test script to verify functionality:
```bash
python scripts/test_rag.py
```

## File Structure

```
scripts/
├── offline_rag_cli.py    # Main CLI application
├── faiss_manager.py      # FAISS index management
├── manage_indices.py     # Index management utility
└── test_rag.py          # Test script

faiss_index/              # Directory for stored indices
models/                   # Directory for models (if needed)
```

## Troubleshooting

### Ollama Issues
- Ensure Ollama is installed and running
- Check that the Llama 3 model is pulled: `ollama list`
- Verify Ollama is in your PATH or at the expected location

### PDF Processing Issues
- Ensure the PDF file is readable and not password-protected
- Check that PyPDF2 can extract text from the PDF
- Try with a different PDF file

### Memory Issues
- Large PDFs may require significant memory for processing
- Consider using smaller chunk sizes for very large documents

## Dependencies

- `faiss-cpu`: Vector similarity search
- `sentence-transformers`: Text embeddings
- `PyPDF2`: PDF text extraction
- `numpy`: Numerical operations
- `torch`: PyTorch for transformers
- `transformers`: Hugging Face transformers

## License

This project is open source. Please check the original license terms.
