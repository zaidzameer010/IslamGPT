import os
from pathlib import Path
import dotenv

from agno.agent import Agent
from agno.embedder.ollama import OllamaEmbedder
from agno.knowledge.text import TextKnowledgeBase
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.knowledge.combined import CombinedKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.storage.sqlite import SqliteStorage
from agno.vectordb.lancedb import LanceDb, SearchType
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
    }

# Load initial config
config = load_env_config()

# Create data directories if they don't exist
os.makedirs("data/text", exist_ok=True)
os.makedirs("data/pdf", exist_ok=True)
os.makedirs("lancedb", exist_ok=True)  # Create LanceDB directory if it doesn't exist

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
    agentic_chunking = AgenticChunking(
        model=OpenAIChat(id=config["llm_model"]),  # Use the same model for chunking
        max_chunk_size=5000,  # The maximum size of each chunk
    )

    # Create knowledge base for text files
    text_knowledge = TextKnowledgeBase(
        path=Path("data/text"),
        vector_db=LanceDb(
            uri="lancedb",
            table_name="text_knowledge",
            search_type=SearchType.hybrid,
            embedder=local_embedder,
        ),
        chunking_strategy=agentic_chunking,  # Add agentic chunking
    )

    # Create knowledge base for PDF files
    pdf_knowledge = PDFKnowledgeBase(
        path=Path("data/pdf"),
        vector_db=LanceDb(
            uri="lancedb",
            table_name="pdf_knowledge",
            search_type=SearchType.hybrid,
            embedder=local_embedder,
        ),
        chunking_strategy=agentic_chunking,  # Add agentic chunking
    )

    # Combine knowledge bases
    combined_knowledge = CombinedKnowledgeBase(
        sources=[text_knowledge, pdf_knowledge],
        vector_db=LanceDb(
            uri="lancedb",
            table_name="combined_knowledge",
            search_type=SearchType.hybrid,
            embedder=local_embedder,
        ),
    )

    # Create the advanced RAG agent
    return Agent(
        name="AdvancedRAG",
        model=OpenAIChat(id=config["llm_model"]),
        description="I am an advanced RAG assistant that searches through local knowledge sources to provide comprehensive answers.",
        instructions=[
            "Search your knowledge base thoroughly before answering questions.",
            "Only use information from the local knowledge base in the data folder.",
            "Do not use web search or external information sources.",
            "Cite your sources clearly when providing information.",
            "If information is not available in your knowledge base, acknowledge the limitations of your knowledge.",
            "Use the calculator for numerical computations when needed.",
            "Provide detailed, well-structured responses with proper formatting.",
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
        """Load all knowledge bases"""
        print("Loading knowledge base...")
        self.agent.knowledge.load(recreate=recreate)
        self.loaded = True
        print("Knowledge base loaded successfully!")

    def query(
        self, question: str, stream: bool = True, show_reasoning: bool = True
    ) -> str:
        """Query the RAG system with a question"""
        if not self.loaded:
            print("Knowledge base not loaded. Loading now...")
            self.load_knowledge()

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
        filepath = Path(f"data/text/{filename}")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        # Reload the knowledge base
        print(f"Adding text document: {filename}")
        print("This will use agentic chunking to process the document...")
        self.agent.knowledge.load(recreate=False)
        print(f"Added text document: {filename}")

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

    def add_pdf_document(self, filepath: str):
        """Add a PDF document to the knowledge base by copying it to the data/pdf directory"""
        source_path = Path(filepath)
        if not source_path.exists():
            print(f"Error: File {filepath} does not exist")
            return

        if not source_path.suffix.lower() == ".pdf":
            print(f"Error: File {filepath} is not a PDF")
            return

        dest_path = Path(f"data/pdf/{source_path.name}")

        # Copy the file
        import shutil

        shutil.copy2(source_path, dest_path)

        # Reload the knowledge base
        self.agent.knowledge.load(recreate=False)
        print(f"Added PDF document: {source_path.name}")

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
        else:
            # Regular query
            rag_app.query(user_input)
