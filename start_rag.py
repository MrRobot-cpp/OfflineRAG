#!/usr/bin/env python3
"""
Start script for OfflineRAG - handles setup and launches the system
"""
import os
import sys
import subprocess
import venv
from pathlib import Path

def setup_venv():
    """Create and configure virtual environment"""
    venv_dir = Path("venv")
    if not venv_dir.exists():
        print("📦 Creating virtual environment...")
        venv.create(venv_dir, with_pip=True)
        
        # Get the pip path
        if sys.platform == "win32":
            pip_path = venv_dir / "Scripts" / "pip.exe"
        else:
            pip_path = venv_dir / "bin" / "pip"
            
        # Install requirements
        print("📥 Installing dependencies...")
        subprocess.run([str(pip_path), "install", "-r", "requirements.txt"])
    
    return venv_dir

def main():
    """Main setup and launch function"""
    print("🚀 Starting OfflineRAG System")
    print("=" * 50)
    
    # Setup virtual environment
    venv_dir = setup_venv()
    
    # Create necessary directories
    for dir_name in ["cache", "embedding_cache", "response_cache", "faiss_index", "data"]:
        os.makedirs(dir_name, exist_ok=True)
    
    # Get the python interpreter path
    if sys.platform == "win32":
        python_path = venv_dir / "Scripts" / "python.exe"
    else:
        python_path = venv_dir / "bin" / "python"
    
    # Launch the RAG system
    print("\n🔄 Launching RAG system...")
    try:
        subprocess.run([str(python_path), "scripts/optimized_enhanced_rag.py"])
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    
if __name__ == "__main__":
    main()