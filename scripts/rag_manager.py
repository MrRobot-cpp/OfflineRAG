#!/usr/bin/env python3

import os
import sys
import subprocess
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import tempfile
import shutil
from pathlib import Path
import json
from datetime import datetime
from scripts.faiss_manager import FAISSManager

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

class EnhancedRAGManager:
    def __init__(self):
        self.faiss_manager = FAISSManager()
        self.llm = None
        self.current_index = None
        self.current_model = None
        self.current_chunks = []
        self.current_metadata = {}
        
        # Initialize LLM
        self._init_llm()
    
    def _init_llm(self):
        """Initialize the LLM"""
        try:
            from offline_rag_cli import LocalLlama
            self.llm = LocalLlama()
            print("✅ LLM initialized successfully")
        except Exception as e:
            print(f"🔴 LLM Error: {str(e)}")
            self.llm = None
    
    def add_file(self, file_path, index_name=None):
        """Add a file to the vector database"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return False
        
        # Generate index name if not provided
        if not index_name:
            index_name = file_path.stem.replace(' ', '_')
        
        print(f"🟡 Processing file: {file_path.name}")
        
        # Process based on file type
        if file_path.suffix.lower() == '.pdf':
            chunks = self._process_pdf(file_path)
        elif file_path.suffix.lower() in ['.txt', '.md']:
            chunks = self._process_text(file_path)
        else:
            print(f"❌ Unsupported file type: {file_path.suffix}")
            return False
        
        if not chunks or (chunks and "❌" in chunks[0]):
            print(f"❌ Failed to process file: {chunks[0] if chunks else 'Unknown error'}")
            return False
        
        # Create FAISS index
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(chunks)
            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(np.array(embeddings).astype('float32'))
            
            # Save to disk
            metadata = {
                'original_filename': file_path.name,
                'file_path': str(file_path),
                'file_type': file_path.suffix,
                'chunk_size': 500,
                'overlap': 50,
                'num_chunks': len(chunks)
            }
            
            if self.faiss_manager.save_index(index, model, chunks, file_path.name, metadata):
                print(f"✅ File added to vector DB: {index_name} ({len(chunks)} chunks)")
                return True
            else:
                print("❌ Failed to save index")
                return False
                
        except Exception as e:
            print(f"❌ Error creating index: {e}")
            return False
    
    def add_multiple_files(self, file_paths, index_name=None):
        """Add multiple files to a single index"""
        if not file_paths:
            print("❌ No files provided")
            return False
        
        print(f"🟡 Processing {len(file_paths)} files...")
        
        all_chunks = []
        file_info = []
        
        for file_path in file_paths:
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"⚠️ Skipping missing file: {file_path}")
                continue
            
            print(f"  Processing: {file_path.name}")
            
            # Process file
            if file_path.suffix.lower() == '.pdf':
                chunks = self._process_pdf(file_path)
            elif file_path.suffix.lower() in ['.txt', '.md']:
                chunks = self._process_text(file_path)
            else:
                print(f"  ⚠️ Skipping unsupported file type: {file_path.suffix}")
                continue
            
            if chunks and "❌" not in chunks[0]:
                # Add file info to each chunk
                for i, chunk in enumerate(chunks):
                    chunks[i] = f"[File: {file_path.name}]\n{chunk}"
                
                all_chunks.extend(chunks)
                file_info.append({
                    'filename': file_path.name,
                    'chunks': len(chunks)
                })
            else:
                print(f"  ❌ Failed to process: {file_path.name}")
        
        if not all_chunks:
            print("❌ No files were successfully processed")
            return False
        
        # Generate index name if not provided
        if not index_name:
            index_name = f"multi_file_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create combined index
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(all_chunks)
            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(np.array(embeddings).astype('float32'))
            
            # Save to disk
            metadata = {
                'original_filename': f"Multi-file index ({len(file_info)} files)",
                'file_info': file_info,
                'file_type': 'multi',
                'chunk_size': 500,
                'overlap': 50,
                'num_chunks': len(all_chunks)
            }
            
            if self.faiss_manager.save_index(index, model, all_chunks, f"multi_file_{len(file_info)}_files", metadata):
                print(f"✅ Multi-file index created: {index_name}")
                print(f"   Files: {len(file_info)}, Total chunks: {len(all_chunks)}")
                return True
            else:
                print("❌ Failed to save multi-file index")
                return False
                
        except Exception as e:
            print(f"❌ Error creating multi-file index: {e}")
            return False
    
    def _process_pdf(self, file_path):
        """Process PDF file"""
        try:
            reader = PdfReader(file_path)
            full_text = ""
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    full_text += f"Page {page_num + 1}:\n{text.strip()}\n\n"
            
            if not full_text.strip():
                return ["❌ No readable text found in PDF"]
            
            return self._chunk_text(full_text)
            
        except Exception as e:
            return [f"❌ PDF Error: {str(e)}"]
    
    def _process_text(self, file_path):
        """Process text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return ["❌ Empty file"]
            
            return self._chunk_text(content)
            
        except Exception as e:
            return [f"❌ Text Error: {str(e)}"]
    
    def _chunk_text(self, text, chunk_size=500, overlap=50):
        """Chunk text into smaller pieces"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > start + chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start = end - overlap if end < len(text) else end
        
        return chunks if chunks else ["❌ No valid chunks created"]
    
    def load_index(self, index_name):
        """Load an existing index"""
        try:
            print(f"🟡 Loading index: {index_name}")
            
            index, model, chunks, metadata = self.faiss_manager.load_index(index_name)
            
            if index is None:
                print("❌ Failed to load index")
                return False
            
            self.current_index = index
            self.current_model = model
            self.current_chunks = chunks
            self.current_metadata = metadata
            
            print(f"✅ Index loaded: {metadata.get('original_filename', index_name)}")
            print(f"   Chunks: {len(chunks)}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            return False
    
    def query(self, question, k=3):
        """Query the loaded index"""
        if not self.current_index or not self.current_model:
            print("❌ No index loaded. Please load an index first.")
            return None
        
        if not self.llm:
            print("❌ LLM not available")
            return None
        
        try:
            print("🟡 Searching documents...")
            
            # Search for relevant chunks
            query_emb = self.current_model.encode([question])
            D, I = self.current_index.search(query_emb.astype('float32'), k)
            top_chunks = [self.current_chunks[i] for i in I[0] if i < len(self.current_chunks)]
            
            print("🟡 Generating AI response...")
            
            # Build context
            context = "\n\n".join(top_chunks) if top_chunks else "No relevant content found."
            
            # Create prompt
            prompt = f"""You are a helpful AI assistant. Based on the following context from documents, please answer the question accurately and concisely.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer based only on the provided context
- If the context doesn't contain enough information, say "I cannot answer this based on the provided documents"
- Be specific and cite relevant parts when possible
- Keep your answer clear and concise

ANSWER:"""
            
            answer = self.llm.generate(prompt)
            return answer
            
        except Exception as e:
            print(f"❌ Query error: {e}")
            return None
    
    def list_indices(self):
        """List all available indices"""
        return self.faiss_manager.list_indices()
    
    def delete_index(self, index_name):
        """Delete an index"""
        return self.faiss_manager.delete_index(index_name)
    
    def interactive_query(self):
        """Interactive query mode"""
        if not self.current_index:
            print("❌ No index loaded. Please load an index first.")
            return
        
        print("\n💬 Interactive Query Mode")
        print("Type 'exit' to quit, 'help' for commands")
        print("-" * 40)
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() == 'exit':
                    break
                elif question.lower() == 'help':
                    print("\n📋 Available commands:")
                    print("  - Ask any question about your documents")
                    print("  - 'exit' - quit query mode")
                    print("  - 'help' - show this help")
                    continue
                elif not question:
                    continue
                
                print("\n🤖 AI Response:")
                print("-" * 40)
                response = self.query(question)
                if response:
                    print(response)
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\n👋 Exiting query mode...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main CLI interface"""
    print("🤖 Enhanced RAG Manager")
    print("=" * 40)
    
    rag = EnhancedRAGManager()
    
    if not rag.llm:
        print("❌ Cannot start without LLM. Please install Ollama and ensure it's running.")
        return
    
    while True:
        print("\n📋 Available Commands:")
        print("1. Add single file to vector DB")
        print("2. Add multiple files to vector DB")
        print("3. Load existing index")
        print("4. Query loaded index")
        print("5. Interactive query mode")
        print("6. List all indices")
        print("7. Delete index")
        print("8. Show current status")
        print("9. Exit")
        
        choice = input("\nEnter your choice (1-9): ").strip()
        
        if choice == "1":
            file_path = input("Enter file path: ").strip()
            if file_path:
                index_name = input("Enter index name (optional): ").strip() or None
                rag.add_file(file_path, index_name)
        
        elif choice == "2":
            print("Enter file paths (one per line, empty line to finish):")
            file_paths = []
            while True:
                path = input().strip()
                if not path:
                    break
                file_paths.append(path)
            
            if file_paths:
                index_name = input("Enter index name (optional): ").strip() or None
                rag.add_multiple_files(file_paths, index_name)
            else:
                print("❌ No files provided")
        
        elif choice == "3":
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
        
        elif choice == "4":
            if not rag.current_index:
                print("❌ No index loaded. Please load an index first.")
                continue
            
            question = input("\n❓ Enter your question: ").strip()
            if question:
                print("\n🤖 AI Response:")
                print("-" * 40)
                response = rag.query(question)
                if response:
                    print(response)
                print("-" * 40)
        
        elif choice == "5":
            rag.interactive_query()
        
        elif choice == "6":
            indices = rag.list_indices()
            if indices:
                print("\n📚 Available Indices:")
                for i, idx in enumerate(indices, 1):
                    print(f"{i}. {idx['name']} - {idx['filename']} ({idx['num_chunks']} chunks)")
                    print(f"   Created: {idx['created_at']}")
            else:
                print("❌ No indices found")
        
        elif choice == "7":
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
                            if rag.current_metadata.get('name') == index_name:
                                rag.current_index = None
                                rag.current_model = None
                                rag.current_chunks = []
                                rag.current_metadata = {}
                        else:
                            print(f"❌ Failed to delete index: {index_name}")
                    else:
                        print("❌ Deletion cancelled")
                else:
                    print("❌ Invalid index number")
            except ValueError:
                print("❌ Please enter a valid number")
        
        elif choice == "8":
            print(f"\n📊 Current Status:")
            print(f"   Index loaded: {'✅ Yes' if rag.current_index else '❌ No'}")
            if rag.current_metadata:
                print(f"   Current file: {rag.current_metadata.get('original_filename', 'Unknown')}")
                print(f"   Chunks: {len(rag.current_chunks)}")
            print(f"   LLM: {'✅ Available' if rag.llm else '❌ Not available'}")
        
        elif choice == "9":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()