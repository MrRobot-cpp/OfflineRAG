FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p cache embedding_cache response_cache faiss_index

# Set environment variables
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/cache/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/app/cache/sentence-transformers

# Command to run the application
CMD ["python", "scripts/optimized_enhanced_rag.py"]