#!/usr/bin/env python3
"""
Test script for Offline RAG CLI functionality
"""
import sys
import os
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from offline_rag_cli import OfflineRAG

def test_rag_functionality():
    """Test the core RAG functionality"""
    print("🧪 Testing Offline RAG Functionality")
    print("=" * 40)
    
    # Initialize RAG system
    rag = OfflineRAG()
    
    # Test 1: Check LLM initialization
    print("\n1. Testing LLM initialization...")
    if rag.llm:
        print("✅ LLM initialized successfully")
    else:
        print("❌ LLM initialization failed")
        return False
    
    # Test 2: Check FAISS manager
    print("\n2. Testing FAISS manager...")
    try:
        indices = rag.list_indices()
        print(f"✅ FAISS manager working - found {len(indices)} indices")
    except Exception as e:
        print(f"❌ FAISS manager error: {e}")
        return False
    
    # Test 3: Test PDF processing (if test PDF exists)
    print("\n3. Testing PDF processing...")
    test_pdf = "test_document.pdf"
    if os.path.exists(test_pdf):
        if rag.load_pdf(test_pdf):
            print("✅ PDF processing successful")
            
            # Test 4: Test question answering
            print("\n4. Testing question answering...")
            test_question = "What is this document about?"
            response = rag.ask_question(test_question)
            print(f"✅ Question answered: {response[:100]}...")
        else:
            print("❌ PDF processing failed")
            return False
    else:
        print("⚠️ No test PDF found - skipping PDF processing test")
        print("   To test PDF processing, place a PDF file named 'test_document.pdf' in the scripts directory")
    
    print("\n✅ All tests completed successfully!")
    return True

def test_with_sample_data():
    """Test with sample text data instead of PDF"""
    print("\n🧪 Testing with sample data...")
    
    rag = OfflineRAG()
    
    # Create sample chunks manually
    sample_chunks = [
        "This is a sample document about artificial intelligence and machine learning.",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to process data.",
        "Natural language processing helps computers understand human language."
    ]
    
    try:
        # Create FAISS index with sample data
        from offline_rag_cli import create_faiss_index
        rag.index, rag.emb_model = create_faiss_index(sample_chunks)
        rag.chunks = sample_chunks
        rag.loaded_file = "sample_data.txt"
        
        print("✅ Sample data loaded successfully")
        
        # Test question answering
        test_questions = [
            "What is machine learning?",
            "What is deep learning?",
            "What is natural language processing?"
        ]
        
        for question in test_questions:
            print(f"\n❓ Question: {question}")
            response = rag.ask_question(question)
            print(f"🤖 Answer: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample data test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Offline RAG Tests")
    print("=" * 50)
    
    # Test core functionality
    success1 = test_rag_functionality()
    
    # Test with sample data
    success2 = test_with_sample_data()
    
    if success1 and success2:
        print("\n🎉 All tests passed! The RAG system is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")
