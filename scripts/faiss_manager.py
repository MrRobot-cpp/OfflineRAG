# -- coding: utf-8 --
"""
Optimized FAISS Index Manager
Handles saving, loading, and managing FAISS indices with metadata.
Auto-adjusts cluster count to avoid training errors.
Supports IVF+PQ for faster retrieval and GPU acceleration.
"""
import os
import json
import pickle
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from datetime import datetime
import torch


class FAISSManager:
    def __init__(self, index_dir="faiss_index", use_gpu=True):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        self.metadata_file = self.index_dir / "metadata.json"

        # GPU detection
        self.gpu_available = use_gpu and self._check_gpu_availability()
        print("✅ GPU acceleration enabled for FAISS" if self.gpu_available else "ℹ️ Using CPU for FAISS operations")

    def _check_gpu_availability(self):
        """Check if GPU is available for FAISS"""
        try:
            if torch.cuda.is_available() and faiss.get_num_gpus() > 0:
                return True
            return False
        except Exception:
            return False

    def create_optimized_index(self, embeddings, index_type="ivf_pq", nlist=None, m=None, nbits=None):
        """
        Create an optimized FAISS index with automatic cluster adjustment.
        Prevents training errors when data points < clusters.
        """
        embeddings = np.array(embeddings).astype("float32")
        dim = embeddings.shape[1]
        n = embeddings.shape[0]

        if n == 0:
            raise ValueError("No embeddings provided to create index.")

        # --- Auto-adjust cluster count (nlist) ---
        if not nlist:
            nlist = min(256, max(1, n // 8))  # one cluster per 8 vectors, capped at 256

        # --- Create the base index ---
        if index_type == "flat_l2":
            print("⚙️ Using Flat L2 index (exact search)")
            index = faiss.IndexFlatL2(dim)

        elif index_type == "ivf_flat":
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            print(f"⚙️ Using IVF Flat index with nlist={nlist}")

        elif index_type == "ivf_pq":
            nlist = min(nlist, n)  # prevent overclustering
            m = m or min(dim // 4, 64)
            nbits = nbits or 8
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
            print(f"⚙️ Using IVF+PQ index with nlist={nlist}, m={m}, nbits={nbits}")

            # --- Prevent small dataset errors ---
            if n < nlist:
                print(f"⚠️ Not enough data points ({n}) for {nlist} clusters. Using Flat index instead.")
                index = faiss.IndexFlatL2(dim)

        else:
            raise ValueError(f"❌ Unknown index type: {index_type}")

        # --- Training (if required) ---
        if hasattr(index, "is_trained") and not index.is_trained:
            if n >= nlist:
                print(f"🟡 Training {index_type.upper()} index...")
                index.train(embeddings)
            else:
                print(f"⚠️ Skipping training: only {n} samples for {nlist} clusters. Using Flat index.")
                index = faiss.IndexFlatL2(dim)

        # --- Add embeddings ---
        print(f"🟡 Adding {n} vectors to index...")
        index.add(embeddings)

        # --- Move to GPU if available ---
        if self.gpu_available and index_type != "flat_l2":
            try:
                index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
                print("✅ Index moved to GPU")
            except Exception as e:
                print(f"⚠️ Could not move index to GPU: {e}")

        print("✅ FAISS index created successfully.")
        return index

    def save_index(self, index, model, chunks, filename, metadata=None):
        """Save FAISS index with metadata"""
        try:
            index_name = filename.replace(".pdf", "").replace(" ", "_")
            index_path = self.index_dir / index_name
            index_path.mkdir(exist_ok=True)

            # Save FAISS index
            faiss.write_index(index, str(index_path / "index.faiss"))

            # Save chunks
            with open(index_path / "chunks.pkl", "wb") as f:
                pickle.dump(chunks, f)

            # Save model info - properly extract model name
            model_name = getattr(model, "name_or_path", None)
            if model_name is None or "SentenceTransformer(" in str(model_name):
                # Fallback: try to get model name from the model object
                try:
                    model_name = model.model_name_or_path
                except AttributeError:
                    # Last resort: use the default model name
                    model_name = "all-MiniLM-L6-v2"
                    print(f"⚠️ Could not determine model name, using default: {model_name}")

            model_info = {"model_name": model_name, "dimension": index.d}
            with open(index_path / "model_info.json", "w") as f:
                json.dump(model_info, f)

            # Save metadata
            metadata = metadata or {}
            index_type = type(index).__name__
            metadata.update(
                {
                    "filename": filename,
                    "created_at": datetime.now().isoformat(),
                    "num_chunks": len(chunks),
                    "index_dimension": index.d,
                    "index_type": index_type,
                }
            )

            with open(index_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Update main metadata
            self._update_main_metadata(index_name, metadata)
            print(f"✅ Index saved for: {filename}")
            return True

        except Exception as e:
            print(f"❌ Error saving index: {e}")
            return False

    def load_index(self, index_name):
        """Load FAISS index by name"""
        try:
            index_path = self.index_dir / index_name
            if not index_path.exists():
                print(f"❌ Index not found: {index_name}")
                return None, None, None, None

            index = faiss.read_index(str(index_path / "index.faiss"))

            with open(index_path / "chunks.pkl", "rb") as f:
                chunks = pickle.load(f)

            with open(index_path / "model_info.json", "r") as f:
                model_info = json.load(f)

            model_name = model_info["model_name"]
            # Handle corrupted model names from previous saves
            if "SentenceTransformer(" in model_name or not model_name or model_name.startswith("sentence-transformers/"):
                print(f"⚠️ Invalid model name detected: {model_name[:50]}...")
                print("🔄 Using default model: all-MiniLM-L6-v2")
                model_name = "all-MiniLM-L6-v2"

            model = SentenceTransformer(model_name)

            with open(index_path / "metadata.json", "r") as f:
                metadata = json.load(f)

            print(f"✅ Index loaded: {metadata.get('filename', index_name)}")
            return index, model, chunks, metadata

        except Exception as e:
            print(f"❌ Error loading index: {e}")
            return None, None, None, None

    def list_indices(self):
        """List all available indices"""
        indices = []
        for item in self.index_dir.iterdir():
            if item.is_dir() and (item / "metadata.json").exists():
                try:
                    with open(item / "metadata.json", "r") as f:
                        metadata = json.load(f)
                    indices.append(
                        {
                            "name": item.name,
                            "filename": metadata.get("filename", "Unknown"),
                            "created_at": metadata.get("created_at", "Unknown"),
                            "num_chunks": metadata.get("num_chunks", 0),
                        }
                    )
                except Exception:
                    continue
        return indices

    def delete_index(self, index_name):
        """Delete an index"""
        try:
            import shutil

            index_path = self.index_dir / index_name
            if index_path.exists():
                shutil.rmtree(index_path)
                self._remove_from_main_metadata(index_name)
                print(f"✅ Index deleted: {index_name}")
                return True
            else:
                print(f"❌ Index not found: {index_name}")
                return False
        except Exception as e:
            print(f"❌ Error deleting index: {e}")
            return False

    def _update_main_metadata(self, index_name, metadata):
        """Update global metadata.json"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, "r") as f:
                    main_metadata = json.load(f)
            else:
                main_metadata = {}

            main_metadata[index_name] = metadata
            with open(self.metadata_file, "w") as f:
                json.dump(main_metadata, f, indent=2)
        except Exception as e:
            print(f"⚠️ Warning: Could not update main metadata: {e}")
