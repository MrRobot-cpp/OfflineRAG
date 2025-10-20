# -*- coding: utf-8 -*-
"""
Offline RAG: Llama 3 + FAISS + MiniLM
FIXED VERSION - Complete UI and functionality fixes
"""
import os
import subprocess
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import panel as pn
import param
import tempfile
import shutil
from pathlib import Path
from scripts.faiss_manager import FAISSManager

# Configure Panel
pn.extension(sizing_mode="stretch_width", template="bootstrap")

# ===== COLOR THEME =====
THEME = {
    'background': '#0f0f23',
    'card_bg': '#1a1a2e', 
    'text_primary': '#ffffff',
    'text_secondary': '#b0b0b0',
    'accent_primary': '#00d4ff',
    'accent_secondary': '#4dd4ff',
    'success': '#00ff88',
    'warning': '#ffaa00',
    'error': '#ff4444',
    'border': '#2a2a3e'
}

# ===== IMPROVED LLM CLASS =====
class LocalLlama:
    def __init__(self, model="llama3"):
        self.model = model
        # Try multiple possible Ollama paths
        possible_paths = [
            r"C:\Users\PC\AppData\Local\Programs\Ollama\ollama.exe",
            r"C:\Program Files\Ollama\ollama.exe",
            "ollama"  # If in PATH
        ]
        
        self.ollama_path = None
        for path in possible_paths:
            if path == "ollama":
                # Check if ollama is in PATH
                try:
                    subprocess.run([path, "version"], capture_output=True, check=True)
                    self.ollama_path = path
                    break
                except:
                    continue
            elif os.path.exists(path):
                self.ollama_path = path
                break
        
        if not self.ollama_path:
            raise Exception("Ollama not found. Please install Ollama and ensure it's in your PATH or at the expected location.")
    
    def generate(self, prompt):
        try:
            result = subprocess.run(
                [self.ollama_path, "run", self.model, prompt],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"❌ Error: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "⏰ Request timed out (120s)"
        except Exception as e:
            return f"⚠️ Error: {str(e)}"

# ===== IMPROVED PDF PROCESSING =====
def load_pdf_chunks(file_path, chunk_size=500, overlap=50):
    try:
        reader = PdfReader(file_path)
        full_text = ""
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                full_text += f"Page {page_num + 1}:\n{text.strip()}\n\n"
        
        if not full_text.strip():
            return ["❌ No readable text found in PDF"]
        
        # Improved chunking with overlap
        chunks = []
        start = 0
        while start < len(full_text):
            end = start + chunk_size
            chunk = full_text[start:end]
            
            # Try to break at sentence boundary
            if end < len(full_text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > start + chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start = end - overlap if end < len(full_text) else end
        
        return chunks if chunks else ["❌ No valid chunks created"]
        
    except Exception as e:
        return [f"❌ PDF Error: {str(e)}"]

# ===== FAISS VECTOR STORE =====
def create_faiss_index(chunks):
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(chunks)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype('float32'))
        return index, model
    except Exception as e:
        raise Exception(f"Failed to create FAISS index: {str(e)}")

def retrieve_top_k(query, index, model, chunks, k=3):
    try:
        query_emb = model.encode([query])
        D, I = index.search(query_emb.astype('float32'), k)
        return [chunks[i] for i in I[0] if i < len(chunks)]
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []

# ===== IMPROVED RAG SYSTEM =====
class OfflineRAG(param.Parameterized):
    chat_history = param.List([])
    status = param.String("🟢 Ready")
    loaded_file = param.String("No file loaded")
    is_processing = param.Boolean(False)
    available_indices = param.List([])
    current_index_name = param.String("")
    
    def __init__(self, **params):
        super().__init__(**params)
        self.chunks = []
        self.index = None
        self.emb_model = None
        self.llm = None
        self.temp_file = None
        self.faiss_manager = FAISSManager()
        
        # Initialize LLM
        try:
            self.llm = LocalLlama()
        except Exception as e:
            self.status = f"🔴 LLM Error: {str(e)}"
        
        # Load available indices
        self.refresh_indices()
    
    def refresh_indices(self):
        """Refresh the list of available indices"""
        try:
            indices = self.faiss_manager.list_indices()
            self.available_indices = [f"{idx['name']} - {idx['filename']}" for idx in indices]
        except Exception as e:
            print(f"Error refreshing indices: {e}")
            self.available_indices = []
    
    def load_index(self, event):
        """Load an existing index"""
        if not hasattr(self, 'index_selector') or not self.index_selector.value:
            self.status = "🔴 No index selected"
            return
        
        try:
            self.is_processing = True
            self.status = "🟡 Loading index..."
            
            # Extract index name from selection
            index_name = self.index_selector.value.split(' - ')[0]
            
            # Load index
            index, model, chunks, metadata = self.faiss_manager.load_index(index_name)
            
            if index is None:
                self.status = "🔴 Failed to load index"
                self.is_processing = False
                return
            
            # Set loaded data
            self.index = index
            self.emb_model = model
            self.chunks = chunks
            self.loaded_file = metadata.get('filename', index_name)
            self.current_index_name = index_name
            self.chat_history = []
            
            self.status = f"🟢 Loaded: {self.loaded_file} ({len(self.chunks)} chunks)"
            self.is_processing = False
            
        except Exception as e:
            self.status = f"🔴 Error loading index: {str(e)}"
            self.is_processing = False
    
    def delete_index(self, event):
        """Delete selected index"""
        if not hasattr(self, 'index_selector') or not self.index_selector.value:
            self.status = "🔴 No index selected"
            return
        
        try:
            index_name = self.index_selector.value.split(' - ')[0]
            
            if self.faiss_manager.delete_index(index_name):
                self.refresh_indices()
                self.status = f"🟢 Deleted index: {index_name}"
                
                # Clear current data if we deleted the current index
                if self.current_index_name == index_name:
                    self.index = None
                    self.emb_model = None
                    self.chunks = []
                    self.loaded_file = "No file loaded"
                    self.current_index_name = ""
                    self.chat_history = []
            else:
                self.status = f"🔴 Failed to delete index: {index_name}"
                
        except Exception as e:
            self.status = f"🔴 Error deleting index: {str(e)}"
    
    def load_file(self, event):
        if not hasattr(self, 'file_input') or self.file_input.value is None:
            self.status = "🔴 No file selected"
            return
        
        try:
            self.is_processing = True
            self.status = "🟡 Processing PDF..."
            
            # Save uploaded file
            if self.temp_file and os.path.exists(self.temp_file):
                os.unlink(self.temp_file)
            
            self.temp_file = tempfile.mktemp(suffix='.pdf')
            with open(self.temp_file, 'wb') as f:
                f.write(self.file_input.value)
            
            self.loaded_file = self.file_input.filename
            self.chunks = load_pdf_chunks(self.temp_file)
            
            if self.chunks and "❌" in self.chunks[0]:
                self.status = f"🔴 {self.chunks[0]}"
                self.is_processing = False
                return
            
            self.index, self.emb_model = create_faiss_index(self.chunks)
            
            # Save index to disk
            index_name = self.loaded_file.replace('.pdf', '').replace(' ', '_')
            metadata = {
                'original_filename': self.loaded_file,
                'chunk_size': 500,
                'overlap': 50
            }
            
            if self.faiss_manager.save_index(self.index, self.emb_model, self.chunks, self.loaded_file, metadata):
                self.current_index_name = index_name
                self.refresh_indices()
            
            self.chat_history = []
            self.status = f"🟢 Loaded: {self.loaded_file} ({len(self.chunks)} chunks)"
            self.is_processing = False
            
        except Exception as e:
            self.status = f"🔴 Error: {str(e)}"
            self.is_processing = False
    
    def ask_question(self, event):
        if not hasattr(self, 'query_input'):
            return
            
        query = self.query_input.value
        if not query or not query.strip():
            self.chat_history.append(("", "🔴 Please enter a question"))
            return
        
        if not self.loaded_file or self.loaded_file == "No file loaded":
            self.chat_history.append((query, "🔴 Please load a PDF first"))
            return
        
        if not self.llm:
            self.chat_history.append((query, "🔴 LLM not available"))
            return
        
        try:
            self.is_processing = True
            self.status = "🟡 Searching documents..."
            
            top_chunks = retrieve_top_k(query, self.index, self.emb_model, self.chunks, k=3)
            
            self.status = "🟡 Generating AI response..."
            
            # Build context
            context = "\n\n".join(top_chunks) if top_chunks else "No specific relevant content found."
            
            # Improved prompt
            prompt = f"""You are a helpful AI assistant. Based on the following context from a document, please answer the question accurately and concisely.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Answer based only on the provided context
- If the context doesn't contain enough information, say "I cannot answer this based on the provided document"
- Be specific and cite relevant parts when possible
- Keep your answer clear and concise

ANSWER:"""
            
            answer = self.llm.generate(prompt)
            
            # Add to chat history
            self.chat_history.append((query, answer))
            self.status = "🟢 Answered"
            self.is_processing = False
            
            # Clear the input
            self.query_input.value = ""
            
        except Exception as e:
            error_msg = f"🔴 Error: {str(e)}"
            self.chat_history.append((query, error_msg))
            self.status = error_msg
            self.is_processing = False

    def clear_history(self, event):
        self.chat_history = []
        self.status = "🟢 History cleared"

# ===== INITIALIZE RAG =====
rag = OfflineRAG()

# ===== IMPROVED UI WIDGETS =====
file_input = pn.widgets.FileInput(
    accept='.pdf', 
    name='📁 Choose PDF File',
    height=40,
    styles={'background': THEME['card_bg'], 'color': THEME['text_primary'], 'border': f"2px solid {THEME['border']}"}
)
rag.file_input = file_input

load_btn = pn.widgets.Button(
    name="🚀 Load PDF", 
    button_type='primary', 
    width=150,
    height=40,
    disabled=False
)
load_btn.on_click(rag.load_file)

query_input = pn.widgets.TextInput(
    placeholder="💭 Ask anything about your document...",
    height=40,
    styles={'background': THEME['card_bg'], 'color': THEME['text_primary'], 'border': f"2px solid {THEME['border']}"}
)
rag.query_input = query_input

ask_btn = pn.widgets.Button(
    name="🔍 Ask AI", 
    button_type='success', 
    width=120,
    height=40
)
ask_btn.on_click(rag.ask_question)

clear_btn = pn.widgets.Button(
    name="🗑️ Clear", 
    button_type='warning', 
    width=120,
    height=40
)
clear_btn.on_click(rag.clear_history)

# ===== INDEX MANAGEMENT WIDGETS =====
index_selector = pn.widgets.Select(
    name="📚 Load Existing Index",
    options=rag.available_indices,
    height=40,
    styles={'background': THEME['card_bg'], 'color': THEME['text_primary']}
)
rag.index_selector = index_selector

load_index_btn = pn.widgets.Button(
    name="📂 Load Index", 
    button_type='primary', 
    width=120,
    height=40
)
load_index_btn.on_click(rag.load_index)

delete_index_btn = pn.widgets.Button(
    name="🗑️ Delete Index", 
    button_type='danger', 
    width=120,
    height=40
)
delete_index_btn.on_click(rag.delete_index)

refresh_indices_btn = pn.widgets.Button(
    name="🔄 Refresh", 
    button_type='light', 
    width=100,
    height=40
)
refresh_indices_btn.on_click(lambda event: rag.refresh_indices())

# ===== STATUS AND INFO PANELS =====
status_card = pn.pane.Markdown(
    pn.bind(lambda status: f"**Status:** {status}", rag.param.status),
    styles={
        'background': THEME['card_bg'],
        'color': THEME['text_primary'],
        'padding': '15px',
        'border-radius': '10px',
        'border': f"2px solid {THEME['accent_primary']}",
        'margin': '10px 0'
    }
)

file_info = pn.pane.Markdown(
    pn.bind(lambda file: f"**📄 File:** {file}", rag.param.loaded_file),
    styles={
        'background': THEME['card_bg'],
        'color': THEME['text_secondary'],
        'padding': '15px',
        'border-radius': '10px',
        'border': f"1px solid {THEME['border']}",
        'margin': '10px 0'
    }
)

# ===== IMPROVED CHAT DISPLAY =====
def create_chat_display(chat_history):
    if not chat_history:
        return pn.pane.Markdown(
            """
            <div style="text-align: center; padding: 40px; color: #b0b0b0;">
                <h3>💡 Welcome to Offline RAG!</h3>
                <p>1. Upload a PDF document</p>
                <p>2. Click "Load PDF" to process it</p>
                <p>3. Ask questions about your document</p>
            </div>
            """,
            styles={'background': THEME['card_bg'], 'border-radius': '10px', 'margin': '10px 0'}
        )
    
    chat_items = []
    for i, (q, a) in enumerate(chat_history):
        if q:  # User message
            chat_items.append(
                pn.pane.Markdown(
                    f"""
                    <div style="background: {THEME['accent_primary']}; color: white; padding: 15px; 
                                border-radius: 20px 20px 5px 20px; margin: 10px 0; margin-left: 20px;">
                        <strong>🧑‍💻 You:</strong> {q}
                    </div>
                    """,
                    styles={'margin': '5px 0'}
                )
            )
        
        # AI response
        chat_items.append(
            pn.pane.Markdown(
                f"""
                <div style="background: {THEME['card_bg']}; color: {THEME['text_primary']}; padding: 15px; 
                            border-radius: 20px 20px 20px 5px; border: 2px solid {THEME['accent_secondary']}; 
                            margin: 10px 0; margin-right: 20px;">
                    <strong>🤖 AI:</strong><br/>{a}
                </div>
                """,
                styles={'margin': '5px 0'}
            )
        )
    
    return pn.Column(*chat_items, scroll=True, height=500)

# ===== MAIN UI LAYOUT =====
header = pn.pane.Markdown(
    """
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #0f0f23, #1a1a2e); 
                border-radius: 15px; margin-bottom: 20px;">
        <h1 style="color: #00d4ff; margin: 0;">🤖 Offline RAG System</h1>
        <p style="color: #b0b0b0; margin: 10px 0 0 0;">Local AI powered by Llama 3 + FAISS + MiniLM</p>
    </div>
    """,
    styles={'margin-bottom': '20px'}
)

# Chat display with proper binding
chat_display = pn.bind(create_chat_display, rag.param.chat_history)

# Main layout
ui = pn.Column(
    header,
    pn.Row(
        pn.Column(
            pn.pane.Markdown("### 📁 Document Setup", styles={'color': THEME['accent_primary'], 'font-size': '18px'}),
            pn.Row(file_input, load_btn, margin=(0, 0, 10, 0)),
            file_info,
            status_card,
            pn.Spacer(height=20),
            pn.pane.Markdown("### 📚 Index Management", styles={'color': THEME['accent_primary'], 'font-size': '18px'}),
            pn.Row(index_selector, refresh_indices_btn, margin=(0, 0, 10, 0)),
            pn.Row(load_index_btn, delete_index_btn, margin=(0, 0, 10, 0)),
            width=400,
            styles={'background': THEME['card_bg'], 'padding': '20px', 'border-radius': '15px', 'margin': '10px'}
        ),
        pn.Column(
            pn.pane.Markdown("### 💬 Chat Interface", styles={'color': THEME['accent_primary'], 'font-size': '18px'}),
            pn.Row(query_input, ask_btn, clear_btn, margin=(0, 0, 10, 0)),
            pn.pane.Markdown("### 📜 Conversation", styles={'color': THEME['accent_primary'], 'font-size': '18px'}),
            chat_display,
            sizing_mode='stretch_width',
            styles={'background': THEME['card_bg'], 'padding': '20px', 'border-radius': '15px', 'margin': '10px'}
        )
    ),
    styles={
        'background': THEME['background'],
        'color': THEME['text_primary'],
        'padding': '20px',
        'min-height': '100vh',
        'font-family': 'Arial, sans-serif'
    },
    sizing_mode="stretch_width"
)

# Make it servable
ui.servable()

print("✅ OfflineRAG System Ready!")
print("🌐 Open your browser and go to: http://localhost:5006")
print("📚 Upload a PDF and start asking questions!")