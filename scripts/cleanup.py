#!/usr/bin/env python3
"""
Cleanup script for OfflineRAG repository
"""
import shutil
import os
from pathlib import Path

def cleanup():
    """Clean up unnecessary files and create required directories"""
    
    # Files to remove
    files_to_remove = [
        "run_rag_cli.bat",
        "add_files_and_query.bat",
        "demo.py",
        "README_CLI.md",
        "scripts/offline_rag_cli.py",
        "repo_tree.txt",
        "TODO.md"
    ]
    
    # Clean cache directories
    cache_dirs = [
        "cache",
        "embedding_cache",
        "response_cache",
        "__pycache__",
        "scripts/__pycache__"
    ]
    
    # Create required directories with .gitkeep
    required_dirs = [
        "data",
        "faiss_index",
        "cache",
        "embedding_cache",
        "response_cache"
    ]
    
    print("🧹 Cleaning up repository...")
    
    # Remove unnecessary files
    for file in files_to_remove:
        try:
            if os.path.exists(file):
                os.remove(file)
                print(f"✅ Removed {file}")
        except Exception as e:
            print(f"❌ Failed to remove {file}: {e}")
    
    # Clean cache directories
    for dir_name in cache_dirs:
        try:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print(f"✅ Cleaned {dir_name}")
        except Exception as e:
            print(f"❌ Failed to clean {dir_name}: {e}")
    
    # Create required directories
    for dir_name in required_dirs:
        try:
            os.makedirs(dir_name, exist_ok=True)
            gitkeep = Path(dir_name) / ".gitkeep"
            gitkeep.touch()
            print(f"✅ Ensured {dir_name} exists")
        except Exception as e:
            print(f"❌ Failed to create {dir_name}: {e}")

if __name__ == "__main__":
    cleanup()