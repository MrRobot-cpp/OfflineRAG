#!/usr/bin/env python3
"""
Quick file addition script
Usage: python quick_add.py <file_path> [index_name]
"""
import sys
import os
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from rag_manager import EnhancedRAGManager

def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_add.py <file_path> [index_name]")
        print("Example: python quick_add.py document.pdf my_docs")
        sys.exit(1)
    
    file_path = sys.argv[1]
    index_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"🟡 Adding file: {file_path}")
    
    rag = EnhancedRAGManager()
    
    if rag.add_file(file_path, index_name):
        print("✅ File added successfully!")
        
        # Ask if user wants to query immediately
        if input("\n❓ Do you want to query this file now? (y/N): ").strip().lower() == 'y':
            if rag.load_index(index_name or Path(file_path).stem.replace(' ', '_')):
                print("\n💬 Enter your questions (type 'exit' to quit):")
                while True:
                    question = input("\n❓ Question: ").strip()
                    if question.lower() == 'exit':
                        break
                    if question:
                        print("\n🤖 Answer:")
                        print("-" * 40)
                        response = rag.query(question)
                        if response:
                            print(response)
                        print("-" * 40)
    else:
        print("❌ Failed to add file")
        sys.exit(1)

if __name__ == "__main__":
    main()
