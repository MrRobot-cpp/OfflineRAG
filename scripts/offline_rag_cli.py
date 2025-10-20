# -*- coding: utf-8 -*-
"""
Offline RAG: Llama 3 + FAISS + MiniLM
Command Line Interface Version - No UI Dependencies
"""
import os
import subprocess
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import tempfile
import shutil
from pathlib import Path
from scripts.faiss_manager import FAISSManager

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
                encoding='utf-8',
                errors='ignore',
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

# ===== CORE RAG SYSTEM =====
class OfflineRAG:
    def __init__(self):
        self.chunks = []
        self.index = None
        self.emb_model = None
        self.llm = None
        self.temp_file = None
        self.faiss_manager = FAISSManager()
        self.loaded_file = "No file loaded"
        self.current_index_name = ""
        
        # Initialize LLM
        try:
            self.llm = LocalLlama()
            print("✅ LLM initialized successfully")
        except Exception as e:
            print(f"🔴 LLM Error: {str(e)}")
            self.llm = None
    
    def load_pdf(self, file_path):
        """Load and process a PDF file"""
        try:
            print(f"🟡 Processing PDF: {file_path}")
            
            # Clean up previous temp file
            if self.temp_file and os.path.exists(self.temp_file):
                os.unlink(self.temp_file)
            
            # Create temp file if needed
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                return False
            
            self.temp_file = file_path
            self.loaded_file = os.path.basename(file_path)
            self.chunks = load_pdf_chunks(file_path)
            
            if self.chunks and "❌" in self.chunks[0]:
                print(f"🔴 {self.chunks[0]}")
                return False
            
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
                print(f"✅ Index saved: {index_name}")
            
            print(f"🟢 Loaded: {self.loaded_file} ({len(self.chunks)} chunks)")
            return True
            
        except Exception as e:
            print(f"🔴 Error: {str(e)}")
            return False
    
    def load_index(self, index_name):
        """Load an existing index"""
        try:
            print(f"🟡 Loading index: {index_name}")
            
            # Load index
            index, model, chunks, metadata = self.faiss_manager.load_index(index_name)
            
            if index is None:
                print("🔴 Failed to load index")
                return False
            
            # Set loaded data
            self.index = index
            self.emb_model = model
            self.chunks = chunks
            self.loaded_file = metadata.get('filename', index_name)
            self.current_index_name = index_name
            
            print(f"🟢 Loaded: {self.loaded_file} ({len(self.chunks)} chunks)")
            return True
            
        except Exception as e:
            print(f"🔴 Error loading index: {str(e)}")
            return False
    
    def ask_question(self, query):
        """Ask a question about the loaded document"""
        if not query or not query.strip():
            return "🔴 Please enter a question"
        
        if not self.loaded_file or self.loaded_file == "No file loaded":
            return "🔴 Please load a PDF first"
        
        if not self.llm:
            return "🔴 LLM not available"
        
        try:
            print("🟡 Searching documents...")
            
            top_chunks = retrieve_top_k(query, self.index, self.emb_model, self.chunks, k=3)
            
            print("🟡 Generating AI response...")
            
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
            print("🟢 Response generated")
            return answer
            
        except Exception as e:
            error_msg = f"🔴 Error: {str(e)}"
            print(error_msg)
            return error_msg
    
    def list_indices(self):
        """List available indices"""
        return self.faiss_manager.list_indices()
    
    def delete_index(self, index_name):
        """Delete an index"""
        return self.faiss_manager.delete_index(index_name)

# ===== COMMAND LINE INTERFACE =====
def main():
    print("🤖 Offline RAG System - Command Line Interface")
    print("=" * 50)
    
    # Initialize RAG system
    rag = OfflineRAG()
    
    if not rag.llm:
        print("❌ Cannot start without LLM. Please install Ollama and ensure it's running.")
        return
    
    while True:
        print("\n📋 Available Commands:")
        print("1. Load PDF file")
        print("2. Load existing index")
        print("3. Ask question")
        print("4. List indices")
        print("5. Delete index")
        print("6. Show status")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            file_path = input("Enter PDF file path: ").strip()
            if file_path:
                rag.load_pdf(file_path)
        
        elif choice == "2":
            indices = rag.list_indices()
            if not indices:
                print("❌ No indices found")
                continue
            
            print("\n📚 Available Indices:")
            for i, idx in enumerate(indices, 1):
                print(f"{i}. {idx['name']} - {idx['filename']} ({idx['num_chunks']} chunks)")
            
            try:
                idx_choice = int(input("\nEnter index number: ")) - 1
                if 0 <= idx_choice < len(indices):
                    rag.load_index(indices[idx_choice]['name'])
                else:
                    print("❌ Invalid index number")
            except ValueError:
                print("❌ Please enter a valid number")
        
        elif choice == "3":
            if not rag.loaded_file or rag.loaded_file == "No file loaded":
                print("❌ Please load a PDF or index first")
                continue
            
            query = input("\n💭 Enter your question: ").strip()
            if query:
                print("\n🤖 AI Response:")
                print("-" * 40)
                response = rag.ask_question(query)
                print(response)
                print("-" * 40)
        
        elif choice == "4":
            indices = rag.list_indices()
            if indices:
                print("\n📚 Available Indices:")
                for i, idx in enumerate(indices, 1):
                    print(f"{i}. {idx['name']} - {idx['filename']} ({idx['num_chunks']} chunks)")
                    print(f"   Created: {idx['created_at']}")
            else:
                print("❌ No indices found")
        
        elif choice == "5":
            indices = rag.list_indices()
            if not indices:
                print("❌ No indices found")
                continue
            
            print("\n📚 Available Indices:")
            for i, idx in enumerate(indices, 1):
                print(f"{i}. {idx['name']} - {idx['filename']}")
            
            try:
                idx_choice = int(input("\nEnter index number to delete: ")) - 1
                if 0 <= idx_choice < len(indices):
                    index_name = indices[idx_choice]['name']
                    confirm = input(f"Are you sure you want to delete '{index_name}'? (y/N): ").strip().lower()
                    
                    if confirm == 'y':
                        if rag.delete_index(index_name):
                            print(f"✅ Deleted index: {index_name}")
                            # Clear current data if we deleted the current index
                            if rag.current_index_name == index_name:
                                rag.index = None
                                rag.emb_model = None
                                rag.chunks = []
                                rag.loaded_file = "No file loaded"
                                rag.current_index_name = ""
                        else:
                            print(f"❌ Failed to delete index: {index_name}")
                    else:
                        print("❌ Deletion cancelled")
                else:
                    print("❌ Invalid index number")
            except ValueError:
                print("❌ Please enter a valid number")
        
        elif choice == "6":
            print(f"\n📊 Status:")
            print(f"   File: {rag.loaded_file}")
            print(f"   Index: {rag.current_index_name}")
            print(f"   Chunks: {len(rag.chunks)}")
            print(f"   LLM: {'✅ Available' if rag.llm else '❌ Not available'}")
        
        elif choice == "7":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
