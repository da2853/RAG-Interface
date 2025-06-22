"""
Document Search Engine: Advanced Document Indexing and Search

This application helps users quickly search and find information in their documents
using advanced indexing and retrieval methods with flexible search parameters
and the option to search general knowledge when documents don't contain the answer.

Main entry point for the application.
"""
import logging
import os
import sys
from pathlib import Path
import streamlit as st  # Import Streamlit at the file level

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import application modules
from config.config_manager import ConfigManager
from document_processing.text_extractor import TextExtractor
from document_processing.text_chunker import TextChunker
from document_processing.embedding_creator import EmbeddingCreator
from vector_db.qdrant_manager import QdrantManager
from answer_generation.answer_generator import AnswerGenerator
from ui.streamlit_components import UIComponents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("document_search.log")
    ]
)
logger = logging.getLogger(__name__)

def initialize_session_state():
    """
    Initialize Streamlit session state variables.
    This ensures all required session state variables exist.
    """
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
        
    if 'show_collection_manager' not in st.session_state:
        st.session_state.show_collection_manager = False

def main():
    """
    Main entry point for the Document Search Engine application.
    Initializes all components and starts the UI.
    """
    try:
        logger.info("Starting Document Search Engine application")

        # Initialize session state
        initialize_session_state()
        
        # Initialize configuration manager
        config_manager = ConfigManager()
        logger.info(f"Initialized configuration manager with collection: {config_manager.collection_name}")
        
        # Initialize document processing components
        text_extractor = TextExtractor()
        text_chunker = TextChunker()
        embedding_creator = EmbeddingCreator(api_key=config_manager.api_keys["OPENAI_API_KEY"])
        
        # Initialize vector database manager
        qdrant_manager = QdrantManager(
            url=config_manager.api_keys["QDRANT_URL"],
            api_key=config_manager.api_keys["QDRANT_API_KEY"]
        )
        
        # Initialize search engine
        answer_generator = AnswerGenerator(
            openrouter_api_key=config_manager.api_keys["OPENROUTER_API_KEY"]
        )
        
        # Create and display the UI
        UIComponents.create_main_page_layout(
            config_manager,
            text_extractor,
            text_chunker,
            embedding_creator,
            qdrant_manager,
            answer_generator
        )
        
    except Exception as e:
        logger.error(f"Error initializing application: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Display error in UI if possible
        st.error(f"Application initialization error: {str(e)}")

if __name__ == "__main__":
    main()