import os
from pathlib import Path
import dotenv
from qdrant_client import QdrantClient
from datetime import datetime
import uuid

from agno.agent import Agent
from agno.embedder.ollama import OllamaEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.knowledge.combined import CombinedKnowledgeBase
from agno.models.openai import OpenAIChat 
from agno.storage.sqlite import SqliteStorage
from agno.document.chunking.agentic import AgenticChunking
from agno.vectordb.qdrant import Qdrant  # Fixed import name

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
        "IslamGPT": os.getenv("QDRANT_COLLECTION"),
    }

# Load initial config
config = load_env_config()

# Initialize Qdrant client
def get_qdrant_client(config):
    """Create and return Qdrant client instance"""
    return QdrantClient(url=config["qdrant_url"])

# Initialize agent with the current configuration
def create_agent(config):
    """Create agent with given configuration"""
    # Configure embedder
    local_embedder = OllamaEmbedder(
        id=config["embedding_model"], 
        dimensions=config["embedding_dimensions"], 
        host=config["ollama_url"]
    )

    # Initialize Qdrant client
    qdrant_client = get_qdrant_client(config)

    # Create unified knowledge base using only Qdrant
    IslamGPT = CombinedKnowledgeBase(
        sources=[],  # No local file sources
        vector_db=Qdrant(
            collection=os.getenv("QDRANT_COLLECTION"),
            url=config["qdrant_url"],
            embedder=local_embedder,
        ),
    )

    # Create the advanced RAG agent
    return Agent(
        name="AdvancedRAG",
        model=OpenAIChat(id=config["llm_model"]),
        description="you are an advanced RAG assistant that searches through the knowledge base to provide accurate and comprehensive answers.",
        instructions=[
            "Search your knowledge base thoroughly before answering questions.",
            "Only use information from the knowledge base.",
            "Cite your sources clearly when providing information.",
            "If information is not available in your knowledge base, clearly state that you don't know and don't use information outside of the knowledge base.",
            "Provide detailed, well-structured responses with proper formatting.",
            "behave like a normal ai assistant",
            "be friendly and engaging",
            "be concise and to the point",
            "stick to the facts and don't care about the user's opinion, be like 'it is what it is'",
        ],
        knowledge=IslamGPT,
        
        storage=SqliteStorage(table_name="advanced_rag", db_file="advanced_rag.db"),
        add_history_to_messages=True,
        add_datetime_to_instructions=True,
        search_knowledge=True,
        markdown=True,
        show_tool_calls=False,
    )

# Create the initial agent
advanced_rag = create_agent(config)


class AdvancedRAGApp:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.loaded = False
        self.config = config

    def reload_config(self):
        """Reload environment variables and recreate the agent with new config"""
        print("Reloading environment configuration...")
        self.config = load_env_config()
        self.agent = create_agent(self.config)
        self.loaded = False
        print("Environment configuration reloaded successfully!")
        print(f"Current config: LLM={self.config['llm_model']}, Embedder={self.config['embedding_model']}")

    def load_knowledge(self, recreate: bool = False):
        """Load knowledge base directly from Qdrant without any local files"""
        print("Loading knowledge base from Qdrant...")
        try:
            # Connect to existing Qdrant collection without any file operations
            self.agent.knowledge.load(recreate=False, upsert=False)
            self.loaded = True
            print("Knowledge base loaded successfully!")
        except Exception as e:
            print(f"Error loading knowledge base: {str(e)}")
            print("Please ensure the Qdrant collection exists and is properly configured.")

    def query(
        self, question: str, stream: bool = True
    ) -> str:
        """Query the RAG system with a question"""
        if not self.loaded:
            print("Knowledge base not loaded. Loading now...")
            self.load_knowledge()

        print(f"Querying: {question}")
        try:
            return self.agent.print_response(
                question, stream=stream
            )
        except Exception as e:
            print(f"Error querying knowledge base: {str(e)}")
            return "Error: Unable to query knowledge base"
        
    def addfiles(self, filepath: str, metadata: dict = None):
        """Add text or PDF files directly to Qdrant using Agno's document processing"""
        print(f"Adding file {filepath} to Qdrant...")
        try:
            # Determine file type based on extension
            file_extension = os.path.splitext(filepath)[1].lower()
            
            if file_extension == '.pdf':
                # Handle PDF file
                try:
                    with open(filepath, 'rb') as f:
                        pdf_content = f.read()
                    
                    # Create a temporary file to process the PDF
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
                        temp_pdf.write(pdf_content)
                        temp_pdf_path = temp_pdf.name
                    
                    try:
                        # Create a temporary PDFKnowledgeBase for processing this content
                        from agno.knowledge.pdf import PDFKnowledgeBase
                        from agno.document.chunking.agentic import AgenticChunking
                        
                        temp_kb = PDFKnowledgeBase(
                            files=[temp_pdf_path],
                            vector_db=self.agent.knowledge.vector_db,
                            chunking_strategy=AgenticChunking(),
                            metadata=[{
                                'type': 'pdf',
                                'filename': os.path.basename(filepath),
                                **(metadata or {})
                            }]
                        )
                    finally:
                        # Clean up temporary file
                        os.unlink(temp_pdf_path)
                        
                except Exception as e:
                    print(f"Error processing PDF file: {str(e)}")
                    return False
                    
            elif file_extension == '.txt':
                # Handle text file
                from agno.knowledge.text import TextKnowledgeBase
                from agno.document.chunking.agentic import AgenticChunking
                
                temp_kb = TextKnowledgeBase(
                    path=filepath,
                    vector_db=self.agent.knowledge.vector_db,
                    chunking_strategy=AgenticChunking(),
                    metadata=[{
                        'type': 'text',
                        'filename': os.path.basename(filepath),
                        **(metadata or {})
                    }]
                )
            else:
                print(f"Unsupported file type: {file_extension}")
                return False
            
            # Process and add to Qdrant through Agno's pipeline
            temp_kb.load(recreate=False, upsert=True)
            print(f"File {filepath} added successfully to Qdrant!")
            return True
            
        except FileNotFoundError:
            print(f"Error: File {filepath} not found")
            return False
        except Exception as e:
            print(f"Error adding file to Qdrant: {str(e)}")
            return False

if __name__ == "__main__":
    # Initialize the RAG application
    rag_app = AdvancedRAGApp(advanced_rag)

    # Load knowledge base without reindexing
    rag_app.load_knowledge()

    print("\n=== Advanced RAG System Ready ===\n")
    print("Available commands:")
    print("  !addfile filepath - Add text or PDF file directly to Qdrant")
    print("  !reload - Reload environment variables and reconfigure agent")
    print("  !config - Show current configuration")
    print("  exit/quit/q - Exit the application")

    # Interactive query loop
    while True:
        user_input = input("\nEnter your question (or 'exit' to quit): ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break

        # Process special commands
        if user_input.startswith("!addfile "):
            # Format: !addfile filepath
            filepath = user_input[9:].strip()
            metadata = {'source': 'user_upload', 'timestamp': str(datetime.now())}
            rag_app.addfiles(filepath, metadata)
        elif user_input == "!reload":
            # Reload environment configuration
            rag_app.reload_config()
            rag_app.load_knowledge()
        elif user_input == "!config":
            # Show current configuration
            print(f"Current configuration:")
            print(f"  LLM Model: {rag_app.config['llm_model']}")
            print(f"  Embedding Model: {rag_app.config['embedding_model']}")
            print(f"  Embedding Dimensions: {rag_app.config['embedding_dimensions']}")
            print(f"  Ollama URL: {rag_app.config['ollama_url']}")
            print(f"  Qdrant URL: {rag_app.config['qdrant_url']}")
        else:
            # Regular query
            rag_app.query(user_input)