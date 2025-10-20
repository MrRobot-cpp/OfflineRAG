# OfflineRAG Optimization Implementation Plan

## 1. Update Dependencies
- [x] Add faiss-gpu, rank_bm25, diskcache, textblob, langchain-text-splitters, cross-encoder
- [x] Update requirements.txt with new packages

## 2. Enhance FAISS Manager (scripts/faiss_manager.py)
- [x] Add support for IVF+PQ index creation
- [ ] Add GPU acceleration detection and usage
- [x] Add index type configuration (FlatL2 vs IVF+PQ)
- [x] Maintain backward compatibility with existing indices

## 3. Optimize Embedding Generation
- [ ] Add option for smaller embedding models (BGE Small, MiniLM variants)
- [x] Implement batch processing with GPU acceleration
- [x] Add embedding caching with diskcache

## 4. Improve Text Chunking
- [x] Replace basic chunking with semantic chunking using langchain RecursiveCharacterTextSplitter
- [ ] Add support for Markdown and code-aware splitting
- [x] Optimize chunk size to 300-700 tokens

## 5. Enhance Retrieval System
- [ ] Implement hybrid retrieval (vector + BM25 keyword search)
- [ ] Add reranking with cross-encoder models
- [ ] Add top-k selection and context compression

## 6. Optimize Generation
- [x] Add response caching for repeated queries
- [x] Add configurable max_context_length
- [x] Add option to disable streaming for faster batch generation
- [x] Update status display to show new settings
- [x] Update configuration menu with new options
- [ ] Add support for quantized Ollama models (q4_K_M variants)
- [ ] Implement prompt compression

## 7. Add System Optimizations
- [ ] GPU acceleration for FAISS and embeddings
- [ ] Precompute and persist embeddings
- [ ] Add performance monitoring and profiling

## 8. Code Cleanup
- [ ] Refactor optimized_enhanced_rag.py for cleaner code
- [ ] Add proper error handling and logging
- [x] Improve CLI interface with progress bars

## 9. Testing and Validation
- [ ] Test backward compatibility with existing indices
- [ ] Benchmark performance improvements
- [ ] Validate accuracy with different configurations
