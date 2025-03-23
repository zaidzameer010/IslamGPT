# Advanced RAG System with Agno and Qdrant

This is an advanced RAG (Retrieval Augmented Generation) system built with Agno (formerly Phidata) and Qdrant.

## Features

- Text and PDF document support
- Agentic chunking for better document segmentation
- Efficient vector search with Qdrant
- Multiple knowledge bases combined into a unified search
- Reasoning capabilities for more comprehensive answers

## Setup

1. Install dependencies:
```bash
pip install agno python-dotenv qdrant-client
```

2. Configure environment variables in `.env`:
```
LLM_MODEL=your_model_name
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=your_openai_url
OLLAMA_URL=http://localhost:11434
EMBEDDING_MODEL=your_embedding_model
EMBEDDING_DIMENSIONS=1024
QDRANT_URL=http://localhost:6333
FORCE_REINDEX=false
```

3. Launch Qdrant:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

4. Run the application:
```bash
python advanced_rag.py
```

## Preventing Reindexing on Every Run

The system is configured to avoid reindexing files on every run by:

1. Using the `upsert=True` parameter for all knowledge bases
2. Creating collections automatically if they don't exist
3. Only indexing new or changed documents

To force a complete reindex, either:
- Set the `FORCE_REINDEX=true` environment variable
- Use the `recreate=True` parameter: `rag_app.load_knowledge(recreate=True)`

## Commands

- `!addtext filename:content` - Add a text document
- `!addpdf filepath` - Add a PDF document
- `!reasoning on/off` - Enable or disable reasoning mode
- `!reason question` - Run a query with reasoning mode and show full reasoning
- `!verify` - Verify agentic chunking implementation
- `!reload` - Reload environment variables and reconfigure agent
- `!config` - Show current configuration
- `!stats` - Show Qdrant collection statistics
- `exit/quit/q` - Exit the application

## Troubleshooting

If documents are still being reindexed on every run:

1. Check that Qdrant is running correctly: `curl http://localhost:6333/collections`
2. Look at collection status in the logs
3. Make sure FORCE_REINDEX=false in your .env file
4. Ensure you have proper read/write permissions for the data directory
5. Try running with `python -u advanced_rag.py` to see unbuffered output 