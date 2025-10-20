#!/usr/bin/env python3
"""
FAISS Index Management Utility
Command-line tool for managing FAISS indices
"""
import sys
import os
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from scripts.faiss_manager import FAISSManager

def main():
    manager = FAISSManager()
    
    print("🤖 FAISS Index Manager")
    print("=" * 40)
    
    while True:
        print("\n📋 Available Commands:")
        print("1. List indices")
        print("2. Show index details")
        print("3. Delete index")
        print("4. Cleanup old indices")
        print("5. Show directory info")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            list_indices(manager)
        elif choice == "2":
            show_index_details(manager)
        elif choice == "3":
            delete_index(manager)
        elif choice == "4":
            cleanup_indices(manager)
        elif choice == "5":
            show_directory_info(manager)
        elif choice == "6":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

def list_indices(manager):
    """List all available indices"""
    print("\n📚 Available Indices:")
    print("-" * 30)
    
    indices = manager.list_indices()
    if not indices:
        print("No indices found.")
        return
    
    for i, idx in enumerate(indices, 1):
        print(f"{i}. {idx['name']}")
        print(f"   File: {idx['filename']}")
        print(f"   Chunks: {idx['num_chunks']}")
        print(f"   Created: {idx['created_at']}")
        print()

def show_index_details(manager):
    """Show detailed information about a specific index"""
    indices = manager.list_indices()
    if not indices:
        print("❌ No indices found.")
        return
    
    print("\n📚 Available Indices:")
    for i, idx in enumerate(indices, 1):
        print(f"{i}. {idx['name']} - {idx['filename']}")
    
    try:
        choice = int(input("\nEnter index number: ")) - 1
        if 0 <= choice < len(indices):
            index_name = indices[choice]['name']
            details = manager.get_index_info(index_name)
            
            if details:
                print(f"\n📄 Details for: {index_name}")
                print("-" * 40)
                for key, value in details.items():
                    print(f"{key}: {value}")
            else:
                print("❌ Could not load index details.")
        else:
            print("❌ Invalid index number.")
    except ValueError:
        print("❌ Please enter a valid number.")

def delete_index(manager):
    """Delete a specific index"""
    indices = manager.list_indices()
    if not indices:
        print("❌ No indices found.")
        return
    
    print("\n📚 Available Indices:")
    for i, idx in enumerate(indices, 1):
        print(f"{i}. {idx['name']} - {idx['filename']}")
    
    try:
        choice = int(input("\nEnter index number to delete: ")) - 1
        if 0 <= choice < len(indices):
            index_name = indices[choice]['name']
            confirm = input(f"Are you sure you want to delete '{index_name}'? (y/N): ").strip().lower()
            
            if confirm == 'y':
                if manager.delete_index(index_name):
                    print(f"✅ Deleted index: {index_name}")
                else:
                    print(f"❌ Failed to delete index: {index_name}")
            else:
                print("❌ Deletion cancelled.")
        else:
            print("❌ Invalid index number.")
    except ValueError:
        print("❌ Please enter a valid number.")

def cleanup_indices(manager):
    """Clean up old indices"""
    try:
        days = int(input("Enter number of days (indices older than this will be deleted): "))
        if days < 1:
            print("❌ Please enter a positive number.")
            return
        
        confirm = input(f"Delete indices older than {days} days? (y/N): ").strip().lower()
        if confirm == 'y':
            deleted_count = manager.cleanup_old_indices(days)
            print(f"✅ Cleaned up {deleted_count} old indices.")
        else:
            print("❌ Cleanup cancelled.")
    except ValueError:
        print("❌ Please enter a valid number.")

def show_directory_info(manager):
    """Show directory information"""
    print(f"\n📂 Index Directory: {manager.index_dir.absolute()}")
    
    # Count files and directories
    total_files = 0
    total_dirs = 0
    total_size = 0
    
    for item in manager.index_dir.rglob('*'):
        if item.is_file():
            total_files += 1
            total_size += item.stat().st_size
        elif item.is_dir():
            total_dirs += 1
    
    print(f"📊 Statistics:")
    print(f"   Directories: {total_dirs}")
    print(f"   Files: {total_files}")
    print(f"   Total Size: {total_size / (1024*1024):.2f} MB")
    
    # Show disk usage
    import shutil
    try:
        total, used, free = shutil.disk_usage(manager.index_dir)
        print(f"   Disk Usage: {used / (1024*1024*1024):.2f} GB used, {free / (1024*1024*1024):.2f} GB free")
    except:
        pass

if __name__ == "__main__":
    main()
