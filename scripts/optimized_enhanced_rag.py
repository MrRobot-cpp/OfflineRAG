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
from faiss_manager import FAISSManager
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import diskcache as dc
from tqdm import tqdm

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

class OptimizedLocalLlama:
    """Optimized LLM with streaming support - NO TIMEOUTS"""
    
    def __init__(self, model_name="llama3"):
        self.model_name = model_name
        self._check_ollama()
    
    def _check_ollama(self):
        """Check if Ollama is available"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise Exception("Ollama not responding")
        except Exception as e:
            raise Exception(f"Ollama not available: {e}")
    
    def generate(self, prompt, stream=True):
        """
        Generate response with optional streaming
        
        Args:
            prompt: The prompt to send
            stream: If True, streams output (no timeout). If False, returns text.
        """
        if stream:
            return self._generate_stream(prompt)
        else:
            return self._generate_batch(prompt)
    
    def _generate_stream(self, prompt):
        """Stream output - NO TIMEOUT"""
        try:
            process = subprocess.Popen(
                ["ollama", "run", self.model_name],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Send prompt
            process.stdin.write(prompt + "\n")
            process.stdin.flush()
            process.stdin.close()
            
            # Stream output
            full_response = ""
            for line in process.stdout:
                print(line, end='', flush=True)
                full_response += line
            
            process.wait()
            return full_response.strip()
            
        except Exception as e:
            return f"Error: {e}"
    
    def _generate_batch(self, prompt):
        """Non-streaming generation (use for non-interactive)"""
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            return "Error: Request timed out"
        except Exception as e:
            return f"Error: {e}"


class EnhancedRAGManager:
    def __init__(self,
                 embedding_model="all-MiniLM-L6-v2",
                 chunk_size=500,
                 chunk_overlap=50,
                 llm_model="llama3.2:1b",
                 index_type="ivf_pq",
                 max_context_length=2000,
                 use_streaming=True):
        """
        Initialize Enhanced RAG Manager with optimization settings

        Args:
            embedding_model: Sentence transformer model
            chunk_size: Text chunk size in characters
            chunk_overlap: Overlap between chunks
            llm_model: Ollama model name
            index_type: FAISS index type ("flat_l2", "ivf_pq", "ivf_flat")
            max_context_length: Maximum context characters for generation
            use_streaming: Whether to use streaming generation (True) or batch (False)
        """
        self.faiss_manager = FAISSManager()
        self.llm = None
        self.current_index = None
        self.current_model = None
        self.current_chunks = []
        self.current_metadata = {}

        # Configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_model = llm_model
        self.embedding_model_name = embedding_model
        self.index_type = index_type
        self.max_context_length = max_context_length
        self.use_streaming = use_streaming

        # Initialize caches
        self.embedding_cache = dc.Cache('embedding_cache', size_limit=1e9)  # 1GB cache
        self.response_cache = dc.Cache('response_cache', size_limit=500e6)  # 500MB cache

        # Initialize LLM lazily (only when needed)
        self.llm = None
    
    def _init_llm(self):
        """Initialize the LLM with streaming support"""
        try:
            self.llm = OptimizedLocalLlama(self.llm_model)
            print(f"✅ LLM initialized successfully ({self.llm_model})")
        except Exception as e:
            print(f"🔴 LLM Error: {str(e)}")
            self.llm = None
    
    def add_file(self, file_path, index_name=None, batch_size=32):
        """
        Add a file to the vector database with batch processing
        
        Args:
            file_path: Path to file
            index_name: Optional index name
            batch_size: Batch size for embedding generation
        """
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
        
        # Create FAISS index with batch processing
        try:
            print(f"🟡 Creating embeddings for {len(chunks)} chunks...")
            model = SentenceTransformer(self.embedding_model_name)
            
            # OPTIMIZATION: Batch embedding generation
            embeddings = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_embeddings = model.encode(batch, show_progress_bar=False)
                embeddings.extend(batch_embeddings)
                
                progress = min(i + batch_size, len(chunks))
                print(f"   🔄 Embedded {progress}/{len(chunks)} chunks", end='\r')
            
            print(f"\n✅ Embeddings created")
            
            embeddings = np.array(embeddings).astype('float32')

            # Create optimized index
            index = self.faiss_manager.create_optimized_index(embeddings, self.index_type)

            # Save to disk
            metadata = {
                'original_filename': file_path.name,
                'file_path': str(file_path),
                'file_type': file_path.suffix,
                'chunk_size': self.chunk_size,
                'overlap': self.chunk_overlap,
                'num_chunks': len(chunks),
                'index_type': self.index_type
            }
            
            if self.faiss_manager.save_index(index, model, chunks, file_path.name, metadata):
                print(f"✅ File added to vector DB: {index_name} ({len(chunks)} chunks)")
                return True
            else:
                print("❌ Failed to save index")
                return False
                
        except Exception as e:
            print(f"❌ Error creating index: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def add_multiple_files(self, file_paths, index_name=None, batch_size=32):
        """
        Add multiple files to a single index with batch processing
        
        Args:
            file_paths: List of file paths
            index_name: Optional index name
            batch_size: Batch size for embedding generation
        """
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
            
            print(f"  📄 Processing: {file_path.name}")
            
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
                print(f"  ✅ Processed: {file_path.name} ({len(chunks)} chunks)")
            else:
                print(f"  ❌ Failed to process: {file_path.name}")
        
        if not all_chunks:
            print("❌ No files were successfully processed")
            return False
        
        # Generate index name if not provided
        if not index_name:
            index_name = f"multi_file_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create combined index with batch processing
        try:
            print(f"\n🟡 Creating embeddings for {len(all_chunks)} total chunks...")
            model = SentenceTransformer(self.embedding_model_name)
            
            # OPTIMIZATION: Batch embedding generation
            embeddings = []
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                batch_embeddings = model.encode(batch, show_progress_bar=False)
                embeddings.extend(batch_embeddings)
                
                progress = min(i + batch_size, len(all_chunks))
                print(f"   🔄 Embedded {progress}/{len(all_chunks)} chunks", end='\r')
            
            print(f"\n✅ All embeddings created")
            
            embeddings = np.array(embeddings).astype('float32')

            # Create optimized index
            index = self.faiss_manager.create_optimized_index(embeddings, self.index_type)

            # Save to disk
            metadata = {
                'original_filename': f"Multi-file index ({len(file_info)} files)",
                'file_info': file_info,
                'file_type': 'multi',
                'chunk_size': self.chunk_size,
                'overlap': self.chunk_overlap,
                'num_chunks': len(all_chunks),
                'index_type': self.index_type
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
            import traceback
            traceback.print_exc()
            return False
    
    def _process_pdf(self, file_path):
        """Process PDF file with progress tracking"""
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            print(f"   📖 Reading {total_pages} pages...")
            
            full_text = ""
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    full_text += f"Page {page_num + 1}:\n{text.strip()}\n\n"
                
                # Show progress every 10 pages
                if (page_num + 1) % 10 == 0 or (page_num + 1) == total_pages:
                    print(f"   📄 Extracted {page_num + 1}/{total_pages} pages", end='\r')
            
            print(f"\n   ✅ Text extraction complete")
            
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
    
    def _chunk_text(self, text, chunk_size=None, overlap=None):
        """Chunk text into smaller pieces using langchain splitter"""
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.chunk_overlap

        # Use RecursiveCharacterTextSplitter for better chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = splitter.split_text(text)
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
    
    def query(self, question, k=3, max_context_length=None, stream=None):
        """
        Query the loaded index with optimized context and caching

        Args:
            question: User's question
            k: Number of chunks to retrieve
            max_context_length: Max context characters (uses instance default if None)
            stream: If True/False, overrides instance setting. If None, uses instance setting.
        """
        if not self.current_index or not self.current_model:
            print("❌ No index loaded. Please load an index first.")
            return None

        if not self.llm:
            print("🟡 Initializing LLM...")
            self._init_llm()
            if not self.llm:
                print("❌ LLM not available")
                return None

        # Use instance defaults if not specified
        if max_context_length is None:
            max_context_length = self.max_context_length
        if stream is None:
            stream = self.use_streaming

        try:
            print("🟡 Searching documents...")

            # Search for relevant chunks
            query_emb = self.current_model.encode([question])
            D, I = self.current_index.search(query_emb.astype('float32'), k)
            top_chunks = [self.current_chunks[i] for i in I[0] if i < len(self.current_chunks)]

            # Build context
            context = "\n\n".join(top_chunks) if top_chunks else "No relevant content found."

            # OPTIMIZATION: Truncate context if too long
            if len(context) > max_context_length:
                context = context[:max_context_length] + "\n\n[Context truncated for performance...]"

            # Create cache key from question and context hash
            import hashlib
            cache_key = hashlib.md5(f"{question}|{context}".encode()).hexdigest()

            # Check response cache first
            cached_response = self.response_cache.get(cache_key)
            if cached_response:
                print("🟢 Using cached response...")
                if stream:
                    # For streaming, print cached response
                    print(cached_response)
                return cached_response

            print("🟡 Generating AI response...")

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

            # Generate response
            answer = self.llm.generate(prompt, stream=stream)

            # Cache the response (only if generation was successful)
            if answer and not answer.startswith("Error:"):
                self.response_cache[cache_key] = answer

            return answer

        except Exception as e:
            print(f"❌ Query error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def list_indices(self):
        """List all available indices"""
        return self.faiss_manager.list_indices()
    
    def delete_index(self, index_name):
        """Delete an index"""
        return self.faiss_manager.delete_index(index_name)
    
    def configure(self, **kwargs):
        """Update configuration settings"""
        if 'chunk_size' in kwargs:
            self.chunk_size = kwargs['chunk_size']
            print(f"✅ Chunk size: {self.chunk_size}")

        if 'chunk_overlap' in kwargs:
            self.chunk_overlap = kwargs['chunk_overlap']
            print(f"✅ Chunk overlap: {self.chunk_overlap}")

        if 'llm_model' in kwargs:
            self.llm_model = kwargs['llm_model']
            print(f"✅ LLM model: {self.llm_model}")
            # Reinitialize LLM
            self._init_llm()

        if 'index_type' in kwargs:
            self.index_type = kwargs['index_type']
            print(f"✅ Index type: {self.index_type}")

        if 'max_context_length' in kwargs:
            self.max_context_length = kwargs['max_context_length']
            print(f"✅ Max context length: {self.max_context_length}")

        if 'use_streaming' in kwargs:
            self.use_streaming = kwargs['use_streaming']
            print(f"✅ Streaming generation: {self.use_streaming}")
    
    def interactive_query(self):
        """Interactive query mode with streaming"""
        if not self.current_index:
            print("❌ No index loaded. Please load an index first.")
            return
        
        print("\n💬 Interactive Query Mode (Streaming)")
        print("Type 'exit' to quit, 'help' for commands")
        print("-" * 70)
        
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
                print("-" * 70)
                response = self.query(question, stream=True)
                if not response:
                    print("(No response generated)")
                print("-" * 70)
                
            except KeyboardInterrupt:
                print("\n👋 Exiting query mode...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main CLI interface"""
    print("\n" + "=" * 70)
    print("🚀 OPTIMIZED ENHANCED RAG MANAGER")
    print("=" * 70)
    print("\n✨ Optimizations:")
    print("  • Streaming responses (no timeouts)")
    print("  • Batch processing for large documents")
    print("  • Configurable context length")
    print("  • Progress tracking")
    print("=" * 70)
    
    rag = EnhancedRAGManager()

    # Note: LLM is initialized lazily when needed for queries
    print("💡 Note: LLM (Ollama) will be initialized when you perform queries.")
    print("   Make sure Ollama is installed and running for AI responses.")
    print("   Visit: https://ollama.ai")
    
    while True:
        print("\n" + "=" * 70)
        print("📋 MAIN MENU")
        print("=" * 70)
        print("\n📄 Document Management:")
        print("  1. Add single file to vector DB")
        print("  2. Add multiple files to vector DB")
        print("  3. Load existing index")
        print("\n💬 Query Operations:")
        print("  4. Query loaded index (single question)")
        print("  5. Interactive query mode (continuous)")
        print("\n📚 Index Management:")
        print("  6. List all indices")
        print("  7. Delete index")
        print("\n⚙️  Settings:")
        print("  8. Show current status")
        print("  9. Configure settings")
        print("  10. Exit")
        print("=" * 70)
        
        choice = input("\n👉 Enter your choice (1-10): ").strip()
        
        if choice == "1":
            file_path = input("\n📁 Enter file path: ").strip()
            if file_path:
                index_name = input("Enter index name (optional, press Enter to skip): ").strip() or None
                rag.add_file(file_path, index_name)
        
        elif choice == "2":
            print("\n📁 Enter file paths (one per line, empty line to finish):")
            file_paths = []
            while True:
                path = input().strip()
                if not path:
                    break
                file_paths.append(path)
            
            if file_paths:
                index_name = input("\nEnter index name (optional, press Enter to skip): ").strip() or None
                rag.add_multiple_files(file_paths, index_name)
            else:
                print("❌ No files provided")
        
        elif choice == "3":
            indices = rag.list_indices()
            if not indices:
                print("\n❌ No indices found")
                continue
            
            print("\n📚 Available Indices:")
            print("-" * 70)
            for i, idx in enumerate(indices, 1):
                print(f"  {i}. {idx['name']}")
                print(f"     File: {idx['filename']}")
                print(f"     Chunks: {idx['num_chunks']}, Created: {idx['created_at']}")
            print("-" * 70)
            
            try:
                idx_choice = int(input("\n👉 Enter index number: ")) - 1
                if 0 <= idx_choice < len(indices):
                    rag.load_index(indices[idx_choice]['name'])
                else:
                    print("❌ Invalid index number")
            except ValueError:
                print("❌ Please enter a valid number")
        
        elif choice == "4":
            if not rag.current_index:
                print("\n❌ No index loaded. Please load an index first (option 3).")
                continue
            
            question = input("\n❓ Enter your question: ").strip()
            if question:
                print("\n🤖 AI Response:")
                print("-" * 70)
                response = rag.query(question, stream=True)
                if not response:
                    print("(No response generated)")
                print("-" * 70)
        
        elif choice == "5":
            rag.interactive_query()
        
        elif choice == "6":
            indices = rag.list_indices()
            if indices:
                print("\n📚 Available Indices:")
                print("=" * 70)
                for i, idx in enumerate(indices, 1):
                    print(f"\n{i}. {idx['name']}")
                    print(f"   File: {idx['filename']}")
                    print(f"   Chunks: {idx['num_chunks']}")
                    print(f"   Created: {idx['created_at']}")
                print("=" * 70)
            else:
                print("\n❌ No indices found")
        
        elif choice == "7":
            indices = rag.list_indices()
            if not indices:
                print("\n❌ No indices found")
                continue
            
            print("\n📚 Available Indices:")
            print("-" * 70)
            for i, idx in enumerate(indices, 1):
                print(f"  {i}. {idx['name']} - {idx['filename']}")
            print("-" * 70)
            
            try:
                idx_choice = int(input("\n👉 Enter index number to delete: ")) - 1
                if 0 <= idx_choice < len(indices):
                    index_name = indices[idx_choice]['name']
                    confirm = input(f"\n⚠️  Are you sure you want to delete '{index_name}'? (y/N): ").strip().lower()
                    
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
            print("\n📊 Current Status")
            print("=" * 70)
            print(f"  Index loaded: {'✅ Yes' if rag.current_index else '❌ No'}")
            if rag.current_metadata:
                print(f"  Current file: {rag.current_metadata.get('original_filename', 'Unknown')}")
                print(f"  Chunks: {len(rag.current_chunks)}")
                print(f"  Index type: {rag.current_metadata.get('index_type', 'Unknown')}")
            print(f"  LLM: {'✅ Available' if rag.llm else '❌ Not available'}")
            print(f"  LLM Model: {rag.llm_model}")
            print(f"  Embedding Model: {rag.embedding_model_name}")
            print(f"  Chunk Size: {rag.chunk_size} chars")
            print(f"  Chunk Overlap: {rag.chunk_overlap} chars")
            print(f"  Default Index Type: {rag.index_type}")
            print(f"  Max Context Length: {rag.max_context_length} chars")
            print(f"  Streaming Generation: {'✅ Enabled' if rag.use_streaming else '❌ Disabled'}")
            print("=" * 70)
        
        elif choice == "9":
            print("\n⚙️  Configuration")
            print("=" * 70)
            print(f"Current settings:")
            print(f"  • Chunk size: {rag.chunk_size} characters")
            print(f"  • Chunk overlap: {rag.chunk_overlap} characters")
            print(f"  • LLM model: {rag.llm_model}")
            print(f"  • Index type: {rag.index_type}")
            print(f"  • Max context length: {rag.max_context_length} characters")
            print(f"  • Streaming generation: {'Enabled' if rag.use_streaming else 'Disabled'}")
            print("\n💡 Available LLM models:")
            print("  • llama3.2:1b (fastest)")
            print("  • llama3.2:3b (balanced)")
            print("  • llama3 (best quality)")
            print("\n💡 Available index types:")
            print("  • flat_l2 (exact search, best quality)")
            print("  • ivf_flat (approximate, balanced)")
            print("  • ivf_pq (compressed, fastest)")
            print("=" * 70)

            print("\nWhat would you like to change?")
            print("  1. Chunk size")
            print("  2. LLM model")
            print("  3. Index type")
            print("  4. Max context length")
            print("  5. Streaming generation")
            print("  6. Back to main menu")

            config_choice = input("\n👉 Enter choice: ").strip()

            if config_choice == "1":
                try:
                    new_size = int(input(f"\nEnter new chunk size (current: {rag.chunk_size}): "))
                    rag.configure(chunk_size=new_size)
                except ValueError:
                    print("❌ Invalid input")

            elif config_choice == "2":
                new_model = input(f"\nEnter new LLM model (current: {rag.llm_model}): ").strip()
                if new_model:
                    rag.configure(llm_model=new_model)

            elif config_choice == "3":
                new_index_type = input(f"\nEnter new index type (current: {rag.index_type}): ").strip()
                if new_index_type in ['flat_l2', 'ivf_flat', 'ivf_pq']:
                    rag.configure(index_type=new_index_type)
                else:
                    print("❌ Invalid index type. Choose from: flat_l2, ivf_flat, ivf_pq")

            elif config_choice == "4":
                try:
                    new_length = int(input(f"\nEnter new max context length (current: {rag.max_context_length}): "))
                    if new_length > 0:
                        rag.configure(max_context_length=new_length)
                    else:
                        print("❌ Context length must be positive")
                except ValueError:
                    print("❌ Invalid input")

            elif config_choice == "5":
                current = rag.use_streaming
                print(f"\nCurrent streaming setting: {'Enabled' if current else 'Disabled'}")
                new_streaming = input("Enable streaming generation? (y/n): ").strip().lower()
                if new_streaming in ['y', 'yes']:
                    rag.configure(use_streaming=True)
                elif new_streaming in ['n', 'no']:
                    rag.configure(use_streaming=False)
                else:
                    print("❌ Invalid input. Keeping current setting.")
        
        elif choice == "10":
            print("\n" + "=" * 70)
            print("👋 Thank you for using Optimized Enhanced RAG Manager!")
            print("💾 All indices saved in: faiss_index/")
            print("=" * 70 + "\n")
            break
        
        else:
            print("\n❌ Invalid choice. Please enter a number between 1-10.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
        sys.exit(0)