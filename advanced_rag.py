import os
from pathlib import Path
import dotenv
from typing import Dict, Any

from agno.agent import Agent
from agno.embedder.ollama import OllamaEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.knowledge.combined import CombinedKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.storage.sqlite import SqliteStorage
from agno.vectordb.qdrant import Qdrant
from agno.tools.calculator import CalculatorTools
from agno.document.chunking.agentic import AgenticChunking

# Function to load environment variables
def load_env_config():
    """Load or reload environment variables from .env file and return config values"""
    # Load environment variables from .env file
    dotenv.load_dotenv(override=True)  # Use override=True to force reload

    # Set OpenAI API key and base URL from environment variables
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
    os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

    # Return configuration dictionary
    return {
        "ollama_url": os.getenv("OLLAMA_URL"),
        "llm_model": os.getenv("LLM_MODEL"),
        "embedding_model": os.getenv("EMBEDDING_MODEL"),
        "embedding_dimensions": int(os.getenv("EMBEDDING_DIMENSIONS")),
        "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "force_reindex": os.getenv("FORCE_REINDEX", "false").lower() in ("true", "1", "yes"),
    }

# Load initial config
config = load_env_config()

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Initialize agent with the current configuration
def create_agent(config):
    """Create agent with given configuration"""
    # Configure embedder
    local_embedder = OllamaEmbedder(
        id=config["embedding_model"], 
        dimensions=config["embedding_dimensions"], 
        host=config["ollama_url"]
    )

    # Create agentic chunking strategy with proper LLM configuration
    agentic_cunking = AgenticChunking(
        model=OpenAIChat(id=config["llm_model"]),
        max_chunk_size=5000,
    )

    # Create knowledge base for text files
    text_knowledge = TextKnowledgeBase(
        path=Path("data"),  # Use the root data folder
        vector_db=Qdrant(
            collection="text_knowledge",
            url=config["qdrant_url"],
            embedder=local_embedder,
        ),
        chunking_strategy=agentic_cunking,  # Add agentic chunking
        file_pattern="**/*.txt",  # Only process text files
        upsert=True,  # Enable upsert to prevent reindexing on every run
    )

    # Create knowledge base for PDF files
    pdf_knowledge = PDFKnowledgeBase(
        path=Path("data"),  # Use the root data folder
        vector_db=Qdrant(
            collection="pdf_knowledge",
            url=config["qdrant_url"],
            embedder=local_embedder,
        ),
        chunking_strategy=agentic_cunking,  # Add agentic chunking
        file_pattern="**/*.pdf",  # Only process PDF files
        upsert=True,  # Enable upsert to prevent reindexing on every run
    )

    # Combine knowledge bases
    combined_knowledge = CombinedKnowledgeBase(
        sources=[text_knowledge, pdf_knowledge],
        vector_db=Qdrant(
            collection="combined_knowledge",
            url=config["qdrant_url"],
            embedder=local_embedder,
        ),
        upsert=True,  # Enable upsert to prevent reindexing on every run
    )

    # Create the advanced RAG agent
    return Agent(
        name="AdvancedRAG",
        model=OpenAIChat(id=config["llm_model"]),
        description="You are an advanced RAG assistant that searches through knowledge sources to provide accurate answers.",
        instructions=[
            "Search your knowledge base thoroughly before answering questions.",
            "Only use information from the provided knowledge base.",
            "Cite your sources clearly when providing information.",
            "If information is not available in your knowledge base, acknowledge the limitations of your knowledge.",
            "Provide detailed, well-structured responses with proper formatting.",
            "talk like a natural chatbot with natual tone",
            "you can give your own opinion and thoughts on the topic when asked, but always cite your sources",\
            "while giving teh answer don't say 'this is what i found from the knowledgebase', 'here is the answer from the knowledgebase, give the answer as it is part of your knowledgebase', just give the answer"
        ],
        knowledge=combined_knowledge,
        tools=[
            CalculatorTools(),
        ],
        storage=SqliteStorage(table_name="advanced_rag", db_file="advanced_rag.db"),
        add_history_to_messages=True,
        add_datetime_to_instructions=True,
        search_knowledge=True,
        markdown=True,
        reasoning=True,  # Enable reasoning by default
        reasoning_model=OpenAIChat(id=config["llm_model"]),  # Use the same model for reasoning
        structured_outputs=True,  # Enable structured outputs
        show_tool_calls=False,  # Disable showing tool calls in responses
    )

# Create the initial agent
advanced_rag = create_agent(config)


class AdvancedRAGApp:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.loaded = False
        self.config = config
        self.collections_metadata = self._get_collections_metadata()

    def _get_collections_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata about Qdrant collections to check if they exist"""
        collections = {}
        try:
            # Get collections info from each knowledge source
            for source in self.agent.knowledge.sources:
                vector_db = source.vector_db
                if hasattr(vector_db, 'qdrant_client'):
                    collection_name = vector_db.collection
                    try:
                        info = vector_db.qdrant_client.get_collection(collection_name=collection_name)
                        collections[collection_name] = {
                            "exists": True,
                            "vectors_count": info.vectors_count,
                            "status": "ready"
                        }
                    except Exception:
                        collections[collection_name] = {
                            "exists": False,
                            "vectors_count": 0,
                            "status": "not_found"
                        }
            
            # Get combined knowledge collection info
            if hasattr(self.agent.knowledge, 'vector_db') and hasattr(self.agent.knowledge.vector_db, 'qdrant_client'):
                collection_name = self.agent.knowledge.vector_db.collection
                try:
                    info = self.agent.knowledge.vector_db.qdrant_client.get_collection(collection_name=collection_name)
                    collections[collection_name] = {
                        "exists": True,
                        "vectors_count": info.vectors_count,
                        "status": "ready"
                    }
                except Exception:
                    collections[collection_name] = {
                        "exists": False,
                        "vectors_count": 0,
                        "status": "not_found"
                    }
        except Exception as e:
            print(f"⚠️ Error getting Qdrant collections metadata: {str(e)}")
        
        return collections
        
    def _ensure_collections_exist(self):
        """Ensure that all required Qdrant collections exist without recreating indexes"""
        try:
            # Get list of required collection names
            collection_names = set()
            for source in self.agent.knowledge.sources:
                if hasattr(source.vector_db, 'collection'):
                    collection_names.add(source.vector_db.collection)
            
            if hasattr(self.agent.knowledge, 'vector_db') and hasattr(self.agent.knowledge.vector_db, 'collection'):
                collection_names.add(self.agent.knowledge.vector_db.collection)
            
            # Check for any existing qdrant client
            qdrant_client = None
            for source in self.agent.knowledge.sources:
                if hasattr(source.vector_db, 'qdrant_client'):
                    qdrant_client = source.vector_db.qdrant_client
                    break
            
            if not qdrant_client and hasattr(self.agent.knowledge, 'vector_db') and hasattr(self.agent.knowledge.vector_db, 'qdrant_client'):
                qdrant_client = self.agent.knowledge.vector_db.qdrant_client
            
            if not qdrant_client:
                print("⚠️ Could not find a qdrant client instance")
                return
            
            # Get existing collections
            existing_collections = set()
            try:
                existing_collections = set(c.name for c in qdrant_client.get_collections().collections)
            except Exception as e:
                print(f"⚠️ Error getting existing collections: {str(e)}")
            
            # Create missing collections with default schema that matches the embedder dimensions
            for collection_name in collection_names:
                if collection_name not in existing_collections:
                    print(f"Creating missing collection: {collection_name}")
                    dimension = self.config.get("embedding_dimensions", 1024)
                    
                    try:
                        # Get any vector_db with this collection name to get embedder dimensions
                        for source in self.agent.knowledge.sources:
                            if hasattr(source.vector_db, 'collection') and source.vector_db.collection == collection_name and hasattr(source.vector_db, 'embedder'):
                                if hasattr(source.vector_db.embedder, 'dimensions'):
                                    dimension = source.vector_db.embedder.dimensions
                                    break
                        
                        # Create the collection with basic settings
                        from qdrant_client.models import VectorParams, Distance
                        qdrant_client.create_collection(
                            collection_name=collection_name,
                            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
                        )
                        print(f"✅ Created collection {collection_name} with dimension {dimension}")
                    except Exception as e:
                        print(f"⚠️ Error creating collection {collection_name}: {str(e)}")
        except Exception as e:
            print(f"⚠️ Error ensuring collections exist: {str(e)}")

    def reload_config(self):
        """Reload environment variables and recreate the agent with new config"""
        print("Reloading environment configuration...")
        self.config = load_env_config()
        self.agent = create_agent(self.config)
        self.loaded = False
        self.collections_metadata = self._get_collections_metadata()
        print("Environment configuration reloaded successfully!")
        print(f"Current config: LLM={self.config['llm_model']}, Embedder={self.config['embedding_model']}")
        
    def load_knowledge(self, recreate: bool = False):
        """Load all knowledge bases
        
        Args:
            recreate: If True, forces complete reindexing of all files.
                     If False (default), uses existing index if available.
        """
        print("\n📚 Loading knowledge base...")
        
        # Override recreate with environment variable if set
        if self.config.get("force_reindex", False):
            recreate = True
            print("⚠️ FORCE_REINDEX environment variable is set to true - forcing reindex")
        
        # Ensure collections exist first
        if not recreate:
            self._ensure_collections_exist()
            
        # Check if collections exist in Qdrant before deciding how to load
        self.collections_metadata = self._get_collections_metadata()  # Refresh metadata
        
        # Debug: print collection statuses
        print("📊 Qdrant collections status:")
        for name, metadata in self.collections_metadata.items():
            status = "✅ Ready" if metadata.get("exists", False) else "❌ Not Found"
            count = metadata.get("vectors_count", 0)
            print(f"  - {name}: {status}, {count} vectors")
            
        all_collections_exist = all(meta.get("exists", False) for meta in self.collections_metadata.values())
        collections_have_data = any(meta.get("vectors_count", 0) > 0 for meta in self.collections_metadata.values())
        
        # If requested to recreate or no collections exist, do a full reindex
        if recreate or not all_collections_exist:
            if recreate:
                reason = "Full reindex requested"
            else:
                reason = "Some Qdrant collections missing"
            
            print(f"🔄 {reason} - rebuilding entire knowledge base...")
            self.agent.knowledge.load(recreate=True, upsert=True)
            self.loaded = True
            
            # Refresh collection metadata after loading
            self.collections_metadata = self._get_collections_metadata()
            print("✅ Knowledge base completely rebuilt and loaded!")
            return
        
        # If collections have data, just load without recreating
        if collections_have_data:
            print("✨ Using existing Qdrant index...")
            try:
                # Just load the existing database without reprocessing
                self.agent.knowledge.load(recreate=False, upsert=True)
                self.loaded = True
                print("✅ Knowledge base loaded from existing Qdrant index!")
                return
            except Exception as e:
                print(f"⚠️ Error loading existing index: {str(e)}")
                print("🔄 Rebuilding index as fallback only because of error...")
                self.agent.knowledge.load(recreate=True, upsert=True)
                self.loaded = True
                print("✅ Knowledge base rebuilt completely due to loading error!")
                return
        else:
            # No data in collections - need to initialize
            print("🔄 Qdrant collections exist but are empty. Initializing knowledge base...")
            try:
                # Try to use upsert mode for the initial load
                self.agent.knowledge.load(recreate=False, upsert=True)
                self.loaded = True
                print("✅ Knowledge base initialized with upsert mode!")
            except Exception as e:
                print(f"⚠️ Upsert mode failed: {str(e)}")
                print("🔄 Falling back to recreate mode...")
                self.agent.knowledge.load(recreate=True, upsert=True)
                self.loaded = True
                print("✅ Knowledge base initialized with recreate mode!")

    def query(
        self, question: str, stream: bool = True, show_reasoning: bool = True
    ) -> str:
        """Query the RAG system with a question"""
        if not self.loaded:
            print("Knowledge base not loaded. Loading now...")
            self.load_knowledge(recreate=False)  # Explicitly set recreate=False here

        print(f"Querying: {question}")
        try:
            return self.agent.print_response(
                question, stream=stream, show_full_reasoning=show_reasoning
            )
        except Exception as e:
            print(f"Error during query: {str(e)}")
            print("Falling back to standard query without reasoning...")
            # Try again with reasoning disabled as fallback
            was_reasoning = self.agent.reasoning
            self.agent.reasoning = False
            try:
                result = self.agent.print_response(question, stream=stream)
                self.agent.reasoning = was_reasoning
                return result
            except Exception as e2:
                print(f"Fallback query also failed: {str(e2)}")
                self.agent.reasoning = was_reasoning
                return "Error processing your query. Please try again or check the logs for details."

    def add_text_document(self, content: str, filename: str):
        """Add a text document to the knowledge base"""
        if not filename.endswith('.txt'):
            filename += '.txt'
            
        filepath = Path(f"data/{filename}")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        # Process the new file
        print(f"Adding text document: {filename}")
        print("This will use agentic chunking to process the document...")
        
        try:
            # Add only the new document to the knowledge base
            self.agent.knowledge.sources[0].add_document(filepath)
            
            # Update the combined knowledge if needed
            if hasattr(self.agent.knowledge, 'update_from_sources'):
                self.agent.knowledge.update_from_sources()
                
            print(f"Added text document: {filename}")
        except Exception as e:
            print(f"Error adding document: {str(e)}")
            print("Falling back to full reload without reindexing...")
            self.agent.knowledge.load(recreate=False, upsert=True)  # Explicitly set recreate=False and upsert=True
            print(f"Added text document: {filename} (via full reload)")

    def add_pdf_document(self, filepath: str):
        """Add a PDF document to the knowledge base by copying it to the data directory"""
        source_path = Path(filepath)
        if not source_path.exists():
            print(f"Error: File {filepath} does not exist")
            return

        if not source_path.suffix.lower() == ".pdf":
            print(f"Error: File {filepath} is not a PDF")
            return

        dest_path = Path(f"data/{source_path.name}")

        # Copy the file
        import shutil
        shutil.copy2(source_path, dest_path)

        # Process the new file
        print(f"Adding PDF document: {source_path.name}")
        
        try:
            # Add only the new document to the knowledge base
            self.agent.knowledge.sources[1].add_document(dest_path)
            
            # Update the combined knowledge if needed
            if hasattr(self.agent.knowledge, 'update_from_sources'):
                self.agent.knowledge.update_from_sources()
                
            print(f"Added PDF document: {source_path.name}")
        except Exception as e:
            print(f"Error adding document: {str(e)}")
            print("Falling back to full reload without reindexing...")
            self.agent.knowledge.load(recreate=False, upsert=True)  # Explicitly set recreate=False and upsert=True
            print(f"Added PDF document: {source_path.name} (via full reload)")

    def verify_chunking(self):
        """Verify that agentic chunking is working correctly"""
        print("Verifying agentic chunking implementation...")
        try:
            # Access the chunking strategy from text knowledge base
            chunking_strategy = self.agent.knowledge.sources[0].chunking_strategy
            print(f"Chunking strategy: {type(chunking_strategy).__name__}")
            print(f"Model: {chunking_strategy.model.__class__.__name__}")
            print(f"Max chunk size: {chunking_strategy.max_chunk_size}")
            print("Agentic chunking is properly configured!")
            return True
        except Exception as e:
            print(f"Error verifying chunking: {str(e)}")
            return False

    def enable_reasoning(self, enable: bool = True):
        """Enable or disable reasoning mode for the agent"""
        self.agent.reasoning = enable
        status = "enabled" if enable else "disabled"
        print(f"Reasoning mode {status}")

    def reasoning_query(self, question: str, stream: bool = True):
        """Query with reasoning mode enabled and showing full reasoning"""
        # Temporarily enable reasoning if not already enabled
        was_enabled = self.agent.reasoning
        if not was_enabled:
            self.agent.reasoning = True

        # Run the query with full reasoning display
        try:
            result = self.query(question, stream=stream, show_reasoning=True)

            # Restore previous reasoning setting if it was changed
            if not was_enabled:
                self.agent.reasoning = False

            return result
        except Exception as e:
            print(f"Error in reasoning query: {str(e)}")
            # Restore previous reasoning setting if it was changed
            if not was_enabled:
                self.agent.reasoning = False
            return "Error processing your reasoning query. Please try again or check the logs for details."

    def show_collection_stats(self):
        """Show statistics about the Qdrant collections"""
        print("\n📊 Qdrant Collection Statistics:")
        self.collections_metadata = self._get_collections_metadata()
        
        if not self.collections_metadata:
            print("No Qdrant collections found or unable to retrieve statistics.")
            return
        
        for collection_name, metadata in self.collections_metadata.items():
            status = "✅ Ready" if metadata.get("exists", False) else "❌ Not Found"
            count = metadata.get("vectors_count", 0)
            print(f"  - {collection_name}: {status}, {count} vectors")


if __name__ == "__main__":
    # Initialize the RAG application
    rag_app = AdvancedRAGApp(advanced_rag)

    # Load knowledge base (set recreate=True to rebuild indexes)
    rag_app.load_knowledge(recreate=False)

    # Verify agentic chunking is working
    rag_app.verify_chunking()

    # Example usage
    print("\n=== Advanced RAG System Ready ===\n")
    print("Available commands:")
    print("  !addtext filename:content - Add a text document")
    print("  !addpdf filepath - Add a PDF document")
    print("  !reasoning on/off - Enable or disable reasoning mode")
    print("  !reason question - Run a query with reasoning mode and show full reasoning")
    print("  !verify - Verify agentic chunking implementation")
    print("  !reload - Reload environment variables and reconfigure agent")
    print("  !config - Show current configuration")
    print("  !stats - Show Qdrant collection statistics")
    print("  exit/quit/q - Exit the application")

    # Interactive query loop
    while True:
        user_input = input("\nEnter your question (or 'exit' to quit): ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break

        # Process special commands
        if user_input.startswith("!addtext "):
            # Format: !addtext filename:content
            parts = user_input[9:].split(":", 1)
            if len(parts) == 2:
                rag_app.add_text_document(parts[1], parts[0])
            else:
                print("Invalid format. Use: !addtext filename:content")
        elif user_input.startswith("!addpdf "):
            # Format: !addpdf filepath
            filepath = user_input[8:].strip()
            rag_app.add_pdf_document(filepath)
        elif user_input.startswith("!reasoning "):
            # Format: !reasoning on/off
            setting = user_input[11:].strip().lower()
            if setting in ["on", "true", "yes", "1"]:
                rag_app.enable_reasoning(True)
            elif setting in ["off", "false", "no", "0"]:
                rag_app.enable_reasoning(False)
            else:
                print("Invalid setting. Use: !reasoning on or !reasoning off")
        elif user_input.startswith("!reason "):
            # Format: !reason question
            question = user_input[8:].strip()
            if question:
                rag_app.reasoning_query(question)
            else:
                print("Please provide a question after !reason")
        elif user_input == "!verify":
            # Verify chunking implementation
            rag_app.verify_chunking()
        elif user_input == "!reload":
            # Reload environment configuration
            rag_app.reload_config()
            rag_app.load_knowledge(recreate=False)
        elif user_input == "!config":
            # Show current configuration
            print(f"Current configuration:")
            print(f"  LLM Model: {rag_app.config['llm_model']}")
            print(f"  Embedding Model: {rag_app.config['embedding_model']}")
            print(f"  Embedding Dimensions: {rag_app.config['embedding_dimensions']}")
            print(f"  Ollama URL: {rag_app.config['ollama_url']}")
            print(f"  Qdrant URL: {rag_app.config['qdrant_url']}")
        elif user_input == "!stats":
            # Show Qdrant collection statistics
            rag_app.show_collection_stats()
        else:
            # Regular query
            rag_app.query(user_input)
