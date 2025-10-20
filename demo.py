#!/usr/bin/env python3
"""
Demo script to show how to use the RAG system
"""
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / "scripts"))

from rag_manager import EnhancedRAGManager

def demo():
    print("🤖 RAG System Demo")
    print("=" * 30)
    
    # Initialize the RAG manager
    rag = EnhancedRAGManager()
    
    if not rag.llm:
        print("❌ LLM not available. Please ensure Ollama is running.")
        return
    
    # Add the sample document
    print("\n1. Adding sample document...")
    if rag.add_file("sample_document.txt", "ai_docs"):
        print("✅ Document added successfully!")
        
        # Load the index
        print("\n2. Loading index...")
        if rag.load_index("ai_docs"):
            print("✅ Index loaded successfully!")
            
            # Ask some questions
            questions = [
                "What is artificial intelligence?",
                "What are the types of machine learning?",
                "What are some applications of AI?",
                "What is deep learning?"
            ]
            
            print("\n3. Asking questions...")
            for i, question in enumerate(questions, 1):
                print(f"\n❓ Question {i}: {question}")
                print("🤖 Answer:")
                print("-" * 40)
                response = rag.query(question)
                if response:
                    print(response)
                print("-" * 40)
        else:
            print("❌ Failed to load index")
    else:
        print("❌ Failed to add document")

if __name__ == "__main__":
    demo()
