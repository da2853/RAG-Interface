"""
Streamlit UI components for ExamGPT application.
Provides reusable UI elements and layouts.
"""
import logging
import time
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple, Callable
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class UIComponents:
    """
    Provides reusable UI components and layouts for the Streamlit interface.
    """
    
    @staticmethod
    def create_sidebar(config_manager) -> None:
        """
        Create the sidebar for configuration settings.
        
        Args:
            config_manager: Configuration manager instance
        """
        with st.sidebar:
            st.title("⚙️ Configuration")
            
            # Collection management section
            UIComponents._create_collection_management_section(config_manager)
            
            # Prompt Templates section
            UIComponents._create_prompt_templates_section(config_manager)
            
            # API Key inputs section
            UIComponents._create_api_keys_section(config_manager)
            
            # Display processed files
            UIComponents._display_processed_files(config_manager)
    
    @staticmethod
    def _create_collection_management_section(config_manager) -> None:
        """
        Create the collection management section in the sidebar.
        
        Args:
            config_manager: Configuration manager instance
        """
        st.subheader("Collection Management")
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"Current: `{config_manager.collection_name}`")

        with col2:
            try:
                from vector_db.qdrant_manager import QdrantManager
                
                # Create QdrantManager with current settings
                qdrant_manager = QdrantManager(
                    url=config_manager.api_keys["QDRANT_URL"],
                    api_key=config_manager.api_keys["QDRANT_API_KEY"],
                    show_status=False  # Don't show status in sidebar
                )
                
                # Get list of collections
                collections = qdrant_manager.list_collections()
                
                if collections:
                    # Option to switch collections
                    if st.button("Manage Collections"):
                        st.session_state.show_collection_manager = True
                else:
                    st.write("No collections found")
                    
            except Exception as e:
                logger.error(f"Error connecting to Qdrant: {str(e)}")
                st.error(f"Error connecting to Qdrant: {str(e)}")

        # Add collection manager UI if requested
        if st.session_state.get("show_collection_manager", False):
            UIComponents._create_collection_manager_ui(config_manager, qdrant_manager)

        # Sync button to check what files are in Qdrant
        if st.button("Sync Files from Database"):
            UIComponents._sync_files_from_database(config_manager)
    
    @staticmethod
    def _create_collection_manager_ui(config_manager, qdrant_manager) -> None:
        """
        Create the collection manager UI.
        
        Args:
            config_manager: Configuration manager instance
            qdrant_manager: QdrantManager instance
        """
        st.subheader("Collection Manager")
        
        # Get list of collections
        try:
            collections = qdrant_manager.list_collections()
            
            if collections:
                # Display available collections
                st.write("Available Collections:")
                
                for collection in collections:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"`{collection}`")
                    
                    with col2:
                        if collection != config_manager.collection_name:
                            if st.button("Switch", key=f"switch_{collection}"):
                                config_manager.switch_collection(collection)
                                st.session_state.show_collection_manager = False
                                st.rerun()
                    
                    with col3:
                        if st.button("Delete", key=f"delete_{collection}"):
                            if collection == config_manager.collection_name:
                                st.warning("This is your current collection. Deleting it will create a new one.")
                                confirm_key = f"confirm_delete_{collection}"
                                if st.button("Confirm Delete", key=confirm_key):
                                    UIComponents._handle_collection_deletion(config_manager, qdrant_manager, collection)
                            else:
                                UIComponents._handle_collection_deletion(config_manager, qdrant_manager, collection)
            
            # Create new collection form
            st.subheader("Create New Collection")
            with st.form("new_collection_form"):
                collection_name = st.text_input("Collection Name (optional)")
                create_submit = st.form_submit_button("Create Collection")
                
                if create_submit:
                    new_name = config_manager.create_new_collection(collection_name)
                    if qdrant_manager.initialize_collection(new_name):
                        st.success(f"Created new collection: {new_name}")
                        st.session_state.show_collection_manager = False
                        st.rerun()
                    else:
                        st.error("Failed to initialize collection")
            
            # Close manager button
            if st.button("Close Manager"):
                st.session_state.show_collection_manager = False
                st.rerun()
            
        except Exception as e:
            logger.error(f"Error managing collections: {str(e)}")
            st.error(f"Error managing collections: {str(e)}")
    
    @staticmethod
    def _handle_collection_deletion(config_manager, qdrant_manager, collection) -> None:
        """
        Handle the deletion of a collection.
        
        Args:
            config_manager: Configuration manager instance
            qdrant_manager: QdrantManager instance
            collection: Name of the collection to delete
        """
        if qdrant_manager.delete_collection(collection):
            st.success(f"Deleted collection: {collection}")
            # If we deleted the current collection, create a new one
            if collection == config_manager.collection_name:
                new_name = config_manager.create_new_collection()
                st.info(f"Created new collection: {new_name}")
                st.rerun()
        else:
            st.error(f"Failed to delete collection: {collection}")
    
    @staticmethod
    def _sync_files_from_database(config_manager) -> None:
        """
        Sync files from the Qdrant database.
        
        Args:
            config_manager: Configuration manager instance
        """
        with st.spinner("Syncing files..."):
            try:
                from vector_db.qdrant_manager import QdrantManager
                
                # Create QdrantManager with current settings
                qdrant_manager = QdrantManager(
                    url=config_manager.api_keys["QDRANT_URL"],
                    api_key=config_manager.api_keys["QDRANT_API_KEY"]
                )
                
                # Get files from Qdrant
                files = qdrant_manager.get_processed_files(config_manager.collection_name)
                
                # Update config manager
                config_manager.processed_files = files
                config_manager.save_config()
                
                st.success(f"Synced {len(files)} files from database")
                
            except Exception as e:
                logger.error(f"Failed to sync files: {str(e)}")
                st.error(f"Failed to sync files: {str(e)}")
    
    @staticmethod
    def _create_prompt_templates_section(config_manager) -> None:
        """
        Create the prompt templates section in the sidebar.
        
        Args:
            config_manager: Configuration manager instance
        """
        st.subheader("Prompt Templates")
        
        # Get available templates
        template_options = list(config_manager.prompt_templates.keys())
        current_template_index = template_options.index(config_manager.selected_template)
        
        # Select template dropdown
        selected_template = st.selectbox(
            "Select Prompt Template", 
            options=template_options,
            index=current_template_index
        )
        
        # Show the selected template
        st.text_area(
            "Template Text",
            value=config_manager.prompt_templates[selected_template],
            height=150,
            disabled=True  # Make read-only to prevent accidental edits
        )
        
        # Set as active template button
        if selected_template != config_manager.selected_template:
            if st.button("Set as Active Template"):
                config_manager.set_selected_template(selected_template)
                config_manager.save_config()
                st.success(f"Set {selected_template} as active template")
        
        # Custom template editor
        with st.expander("Edit Custom Template"):
            custom_template = st.text_area(
                "Custom Template Text",
                value=config_manager.prompt_templates["Custom"],
                height=200
            )
            
            if st.button("Save Custom Template"):
                config_manager.add_prompt_template("Custom", custom_template)
                config_manager.save_config()
                st.success("Custom template saved")
    
    @staticmethod
    def _create_api_keys_section(config_manager) -> None:
        """
        Create the API keys section in the sidebar.
        
        Args:
            config_manager: Configuration manager instance
        """
        st.subheader("API Keys")
        
        # OpenAI API Key
        openai_key = st.text_input(
            "OpenAI API Key", 
            value=config_manager.api_keys["OPENAI_API_KEY"], 
            type="password",
            help="Required for embedding creation and document processing"
        )
        
        # OpenRouter API Key
        openrouter_key = st.text_input(
            "OpenRouter API Key", 
            value=config_manager.api_keys["OPENROUTER_API_KEY"], 
            type="password",
            help="Required for accessing various LLM models"
        )
        
        # Qdrant configuration
        st.subheader("Qdrant Configuration")
        use_cloud_qdrant = st.checkbox(
            "Use Cloud Qdrant", 
            value=bool(config_manager.api_keys["QDRANT_URL"]),
            help="Enable to use a cloud Qdrant instance instead of in-memory"
        )
        
        if use_cloud_qdrant:
            qdrant_url = st.text_input(
                "Qdrant URL", 
                value=config_manager.api_keys["QDRANT_URL"],
                help="The URL of your Qdrant cloud instance"
            )
            
            qdrant_api_key = st.text_input(
                "Qdrant API Key", 
                value=config_manager.api_keys["QDRANT_API_KEY"], 
                type="password",
                help="API key for your Qdrant cloud instance"
            )
        else:
            st.info("Using in-memory Qdrant instance (data will be lost when app restarts)")
            qdrant_url = ""
            qdrant_api_key = ""
        
        # Update API keys button
        if st.button("Update Configuration"):
            # Update API keys
            config_manager.update_api_key("OPENAI_API_KEY", openai_key)
            config_manager.update_api_key("OPENROUTER_API_KEY", openrouter_key)
            config_manager.update_api_key("QDRANT_URL", qdrant_url)
            config_manager.update_api_key("QDRANT_API_KEY", qdrant_api_key)
            
            # Save to config file
            if config_manager.save_config():
                st.success("Configuration updated and saved!")
            else:
                st.warning("Configuration updated but could not be saved to disk")
    
    @staticmethod
    def _display_processed_files(config_manager) -> None:
        """
        Display the processed files in the sidebar.
        
        Args:
            config_manager: Configuration manager instance
        """
        st.subheader("Processed Files")
        
        if config_manager.processed_files:
            with st.expander("View Processed Files"):
                for file in config_manager.processed_files:
                    st.write(f"- {file}")
        else:
            st.write("No files processed yet")
    
    @staticmethod
    def create_upload_tab(config_manager, 
                         text_extractor, 
                         text_chunker, 
                         embedding_creator, 
                         qdrant_manager) -> None:
        """
        Create the document upload tab.
        
        Args:
            config_manager: Configuration manager instance
            text_extractor: Text extractor instance
            text_chunker: Text chunker instance
            embedding_creator: Embedding creator instance
            qdrant_manager: Qdrant manager instance
        """
        st.header("Upload your PDF documents")
        
        # Check for required API keys early
        api_key_status = UIComponents._check_api_key_status(config_manager)
        if not api_key_status["openai"]:
            st.warning("⚠️ OpenAI API Key is required for processing documents. Please add it in the Configuration sidebar.")
        
        # File uploader
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        
        # Check for duplicates and provide feedback
        new_file_objects = []
        if uploaded_files:
            new_file_objects = UIComponents._check_duplicate_files(config_manager, uploaded_files)
        
        # Chunking options in expander
        chunk_settings = UIComponents._create_chunking_settings()
        chunk_size = chunk_settings["chunk_size"]
        chunk_overlap = chunk_settings["chunk_overlap"]
        
        # Process documents button
        if uploaded_files:
            if st.button("Process Documents", disabled=not api_key_status["openai"]):
                UIComponents._process_documents(
                    config_manager,
                    qdrant_manager,
                    text_extractor,
                    text_chunker,
                    embedding_creator,
                    new_file_objects,
                    chunk_size,
                    chunk_overlap
                )
    
    @staticmethod
    def _check_api_key_status(config_manager) -> Dict[str, bool]:
        """
        Check the status of required API keys.
        
        Args:
            config_manager: Configuration manager instance
            
        Returns:
            Dict with API key status
        """
        return {
            "openai": bool(config_manager.api_keys["OPENAI_API_KEY"]),
            "openrouter": bool(config_manager.api_keys["OPENROUTER_API_KEY"]),
            "qdrant": bool(config_manager.api_keys["QDRANT_URL"] and config_manager.api_keys["QDRANT_API_KEY"])
        }
    
    @staticmethod
    def _check_duplicate_files(config_manager, uploaded_files) -> List:
        """
        Check for duplicate files and provide feedback.
        
        Args:
            config_manager: Configuration manager instance
            uploaded_files: List of uploaded files
            
        Returns:
            List of new file objects
        """
        file_names = [file.name for file in uploaded_files]
        new_files, duplicates = config_manager.check_duplicate_files(file_names)
        
        # Map back to file objects
        new_file_objects = [f for f in uploaded_files if f.name in new_files]
        
        if duplicates:
            st.warning(f"The following files have already been processed: {', '.join(duplicates)}")
            
        if not new_file_objects and duplicates:
            st.info("All uploaded files have already been processed. You can proceed to the 'Ask Questions' tab.")
        
        # Display count of new files to be processed
        if new_file_objects:
            st.info(f"{len(new_file_objects)} new file(s) ready for processing.")
        
        return new_file_objects
    
    @staticmethod
    def _create_chunking_settings() -> Dict[str, int]:
        """
        Create the chunking settings UI.
        
        Returns:
            Dict with chunking settings
        """
        with st.expander("Advanced Options"):
            chunk_size = st.slider(
                "Chunk Size", 
                min_value=200, 
                max_value=2000, 
                value=700, 
                help="Number of characters in each chunk (700 recommended for academic material)"
            )
            
            chunk_overlap = st.slider(
                "Chunk Overlap", 
                min_value=50, 
                max_value=500, 
                value=175,
                help="Number of overlapping characters between chunks (175 recommended)"
            )
        
        # Add info about optimal settings
        with st.expander("Chunking Settings Guide"):
            st.markdown("""
            ### Recommended Settings for Exam Materials
            
            #### Chunk Size
            - **600-800 characters**: Best for most academic content
            - **400-600 characters**: Better for dense technical content with many formulas
            - **800-1200 characters**: Better for narrative content with fewer technical details
            
            #### Chunk Overlap
            - **25% of chunk size**: Ensures concepts don't get split between chunks
            
            #### Number of Chunks to Retrieve
            - **5-10 chunks**: Balanced context for most questions
            - **10-15 chunks**: For complex questions requiring more context
            """)
        
        return {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
    
    @staticmethod
    def _process_documents(config_manager, qdrant_manager, text_extractor, text_chunker, 
                          embedding_creator, new_file_objects, chunk_size, chunk_overlap) -> None:
        """
        Process documents and add them to the vector database.
        
        Args:
            config_manager: Configuration manager instance
            qdrant_manager: Qdrant manager instance
            text_extractor: Text extractor instance
            text_chunker: Text chunker instance
            embedding_creator: Embedding creator instance
            new_file_objects: List of new file objects
            chunk_size: Chunk size setting
            chunk_overlap: Chunk overlap setting
        """
        # Check API keys first
        if not config_manager.api_keys["OPENAI_API_KEY"]:
            st.error("OpenAI API Key is required for processing documents")
            return
            
        # Initialize Qdrant collection
        if not qdrant_manager.initialize_collection(config_manager.collection_name):
            st.error("Failed to initialize Qdrant collection")
            return
            
        if not new_file_objects:
            st.info("No new files to process. All uploaded files have already been processed.")
            return
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Set OpenAI API key for embedding creation
        embedding_creator.set_api_key(config_manager.api_keys["OPENAI_API_KEY"])
        
        successful_files = 0
        failed_files = []
        
        for i, uploaded_file in enumerate(new_file_objects):
            file_name = uploaded_file.name
            status_text.text(f"Processing {file_name}...")
            
            try:
                # Extract text from PDF
                text, error = text_extractor.extract_text_from_pdf(uploaded_file, file_name)
                
                if error:
                    logger.error(f"Error extracting text from {file_name}: {error}")
                    failed_files.append((file_name, f"Text extraction failed: {error}"))
                    progress_bar.progress((i + 1) / len(new_file_objects))
                    continue
                
                # Split text into chunks
                chunks = text_chunker.split_text_into_chunks(
                    text, 
                    chunk_size=chunk_size, 
                    overlap=chunk_overlap
                )
                
                if not chunks:
                    logger.warning(f"No text chunks extracted from {file_name}")
                    failed_files.append((file_name, "No text chunks extracted"))
                    progress_bar.progress((i + 1) / len(new_file_objects))
                    continue
                
                # Create embeddings
                status_text.text(f"Creating embeddings for {file_name}...")
                embeddings = embedding_creator.create_embeddings(chunks)
                
                if not embeddings:
                    logger.error(f"Failed to create embeddings for {file_name}")
                    failed_files.append((file_name, "Failed to create embeddings"))
                    progress_bar.progress((i + 1) / len(new_file_objects))
                    continue
                
                # Store in Qdrant
                status_text.text(f"Storing embeddings for {file_name}...")
                if qdrant_manager.store_embeddings(
                    config_manager.collection_name, 
                    embeddings, 
                    {"filename": file_name}
                ):
                    config_manager.add_processed_file(file_name)
                    successful_files += 1
                else:
                    failed_files.append((file_name, "Failed to store embeddings"))
                
                progress_bar.progress((i + 1) / len(new_file_objects))
                
            except Exception as e:
                logger.error(f"Error processing {file_name}: {str(e)}")
                failed_files.append((file_name, str(e)))
                progress_bar.progress((i + 1) / len(new_file_objects))
        
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        # Show summary
        if successful_files > 0:
            st.success(f"Successfully processed {successful_files} out of {len(new_file_objects)} documents")
            
            # Save updated file list to config
            config_manager.save_config()
        else:
            st.error("Failed to process any documents")
        
        # Show details for failed files
        if failed_files:
            with st.expander(f"Details for {len(failed_files)} failed files"):
                for file_name, error in failed_files:
                    st.error(f"{file_name}: {error}")
    
    @staticmethod
    def create_question_tab(config_manager, 
                          embedding_creator, 
                          qdrant_manager, 
                          answer_generator) -> None:
        """
        Create the question answering tab.
        
        Args:
            config_manager: Configuration manager instance
            embedding_creator: Embedding creator instance
            qdrant_manager: Qdrant manager instance
            answer_generator: Answer generator instance
        """
        st.header("Ask questions about your documents")
        
        # Check for required API keys early
        api_key_status = UIComponents._check_api_key_status(config_manager)
        
        # Set system prompt based on selected template
        current_template = config_manager.selected_template
        template_text = config_manager.get_prompt_template(current_template)
        answer_generator.set_system_prompt(template_text)
        
        # Display current assistant type with better formatting
        st.info(f"Current assistant: **{current_template}**")
        
        # Query input with placeholder text
        query = st.text_area(
            "Enter your question:", 
            height=100,
            placeholder="Example: Explain the key concepts from chapter 3..."
        )
        
        # Answer mode selection
        answer_mode = UIComponents._create_answer_mode_selection(config_manager)
        use_context = (answer_mode == "With document context (RAG)")
        
        # Model selection and parameters
        col1, col2 = st.columns([1, 1])
        
        with col1:
            selected_models = UIComponents._create_model_selection()
        
        with col2:
            ui_settings = UIComponents._create_additional_settings(use_context)
            search_limit = ui_settings.get("search_limit", 10)
            enable_streaming = ui_settings.get("enable_streaming", True)
        
        # Get answer button
        if UIComponents._create_get_answer_button(
            query, 
            selected_models, 
            use_context, 
            api_key_status, 
            config_manager
        ):
            UIComponents._process_question(
                query,
                selected_models,
                use_context,
                search_limit,
                enable_streaming,
                config_manager,
                answer_generator,
                embedding_creator,
                qdrant_manager
            )
    
    @staticmethod
    def _create_answer_mode_selection(config_manager) -> str:
        """
        Create the answer mode selection UI.
        
        Args:
            config_manager: Configuration manager instance
            
        Returns:
            Selected answer mode
        """
        answer_mode = st.radio(
            "Answer mode:",
            ["With document context (RAG)", "Without document context (Direct)"],
            index=0,
            help="Choose whether to use your uploaded documents as context for answering."
        )
        
        # Show appropriate explanation based on selected mode
        if answer_mode == "With document context (RAG)":
            st.info("Answers will be based on your uploaded documents. This is useful for questions related to your specific materials.")
            if not config_manager.processed_files:
                st.warning("No documents have been processed yet. Please upload and process documents first.")
        else:
            st.warning("Answers will come from the model's general knowledge, not your documents. Use this for general questions not covered in your materials.")
        
        return answer_mode
    
    @staticmethod
    def _create_model_selection() -> List[str]:
        """
        Create the model selection UI.
        
        Returns:
            List of selected models
        """
        available_models = [
            "qwen/qwq-32b",
            "qwen/qwen-max",
            "google/gemini-2.0-flash-001",
            "google/gemini-pro",
            "google/gemini-2.0-flash-thinking-exp:free",
            "deepseek/deepseek-r1-distill-llama-70b",
            "anthropic/claude-3.7-sonnet:thinking",
            "openai/o3-mini-high",
            "openai/chatgpt-4o-latest",
            "openai/o1"
        ]
        
        selected_models = st.multiselect(
            "Select models (max 5 recommended):",
            available_models,
            default=[available_models[0]],
            help="Select the AI models to use for answering your question"
        )
        
        if not selected_models:
            st.warning("Please select at least one model")
        
        return selected_models
    
    @staticmethod
    def _create_additional_settings(use_context) -> Dict[str, Any]:
        """
        Create the additional settings UI.
        
        Args:
            use_context: Whether to use document context
            
        Returns:
            Dict with additional settings
        """
        settings = {}
        
        # Only show RAG options if using context
        if use_context:
            settings["search_limit"] = st.slider(
                "Number of chunks to retrieve:", 
                min_value=3, 
                max_value=25, 
                value=10,
                help="More chunks provide more context but may slow down processing"
            )
        else:
            settings["search_limit"] = 0  # Not used in direct mode
        
        # Option for streaming responses
        settings["enable_streaming"] = st.checkbox(
            "Enable streaming responses", 
            value=True,
            help="See answers as they are generated (only works with single model selection)"
        )
        
        return settings
    
    @staticmethod
    def _create_get_answer_button(query, selected_models, use_context, api_key_status, config_manager) -> bool:
        """
        Create the get answer button and check if it's clicked.
        
        Args:
            query: User query
            selected_models: List of selected models
            use_context: Whether to use document context
            api_key_status: API key status dict
            config_manager: Configuration manager instance
            
        Returns:
            True if the button is clicked and all checks pass, False otherwise
        """
        # Get appropriate button label based on context mode
        button_label = "Get Answer with Context" if use_context else "Get Direct Answer"
        
        # Check if the button should be disabled
        button_disabled = not query or not selected_models
        
        # Additional check for RAG mode
        if use_context and not config_manager.processed_files:
            button_disabled = True
        
        # Create the button
        button_clicked = st.button(button_label, disabled=button_disabled)
        
        # Check if the button is clicked
        if button_clicked:
            # Check API keys
            if use_context and (not api_key_status["openai"] or not api_key_status["openrouter"]):
                st.error("Both OpenAI and OpenRouter API keys are required for context-based answers")
                return False
            elif not use_context and not api_key_status["openrouter"]:
                st.error("OpenRouter API key is required")
                return False
            
            # All checks pass
            return True
        
        return False
    
    @staticmethod
    def _process_question(query, selected_models, use_context, search_limit, enable_streaming,
                        config_manager, answer_generator, embedding_creator, qdrant_manager) -> None:
        """
        Process a question and display the answer.
        
        Args:
            query: User query
            selected_models: List of selected models
            use_context: Whether to use document context
            search_limit: Number of chunks to retrieve
            enable_streaming: Whether to enable streaming responses
            config_manager: Configuration manager instance
            answer_generator: Answer generator instance
            embedding_creator: Embedding creator instance
            qdrant_manager: Qdrant manager instance
        """
        with st.spinner("Processing query..."):
            try:
                # Set API keys
                answer_generator.set_api_key(config_manager.api_keys["OPENROUTER_API_KEY"])
                
                # Create tabs for each model
                model_tabs = st.tabs(selected_models)
                
                # Create placeholders inside each tab
                model_placeholders = {}
                result_placeholders = {}
                
                for i, model in enumerate(selected_models):
                    with model_tabs[i]:
                        model_placeholders[model] = st.empty()
                        model_placeholders[model].info(f"Starting request for {model}...")
                        result_placeholders[model] = st.empty()
                
                # Different processing paths based on whether we're using context
                similar_chunks = None
                all_sources = None
                
                if use_context:
                    # Get similar chunks for RAG
                    similar_chunks = UIComponents._get_similar_chunks(
                        query,
                        config_manager,
                        embedding_creator,
                        qdrant_manager,
                        search_limit
                    )
                    
                    if not similar_chunks:
                        st.error("No relevant information found in the documents.")
                        return
                
                # Now handle either streaming or concurrent generation
                if enable_streaming and len(selected_models) == 1:
                    # Single model with streaming
                    UIComponents._handle_streaming_response(
                        query,
                        similar_chunks,
                        selected_models[0],
                        model_tabs[0],
                        model_placeholders,
                        result_placeholders,
                        answer_generator,
                        use_context
                    )
                    
                    # Get sources for context-based query (only if using context)
                    if use_context:
                        _, all_sources = answer_generator.generate_answer(
                            query, 
                            similar_chunks, 
                            model=selected_models[0],
                            use_context=True
                        )
                else:
                    # Multiple models or non-streaming - use concurrent processing
                    all_sources = UIComponents._handle_concurrent_responses(
                        query,
                        similar_chunks,
                        selected_models,
                        model_placeholders,
                        result_placeholders,
                        answer_generator,
                        use_context
                    )
                
                # Display sources in an expander outside the tabs (only if using context)
                if use_context and all_sources:
                    UIComponents._display_source_documents(all_sources)
                
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                st.error(f"Error processing your query: {str(e)}")
    
    @staticmethod
    def _get_similar_chunks(query, config_manager, embedding_creator, qdrant_manager, search_limit):
        """
        Get similar chunks for RAG.
        
        Args:
            query: User query
            config_manager: Configuration manager instance
            embedding_creator: Embedding creator instance
            qdrant_manager: Qdrant manager instance
            search_limit: Number of chunks to retrieve
            
        Returns:
            List of similar chunks or None if not found
        """
        # Set OpenAI API key for embedding creation
        embedding_creator.set_api_key(config_manager.api_keys["OPENAI_API_KEY"])
        
        # Create embedding for query
        query_embedding = embedding_creator.create_query_embedding(query)
        
        # Search for similar chunks
        return qdrant_manager.search_similar_chunks(
            config_manager.collection_name,
            query_embedding,
            limit=search_limit
        )
    
    @staticmethod
    def _handle_streaming_response(query, similar_chunks, model, model_tab, 
                                 model_placeholders, result_placeholders,
                                 answer_generator, use_context):
        """
        Handle streaming response for a single model.
        
        Args:
            query: User query
            similar_chunks: List of similar chunks
            model: Model name
            model_tab: Model tab
            model_placeholders: Dict of model placeholders
            result_placeholders: Dict of result placeholders
            answer_generator: Answer generator instance
            use_context: Whether to use document context
        """
        with model_tab:
            answer_container = result_placeholders[model]
            status_container = model_placeholders[model]
            status_container.info(f"Streaming response from {model}...")
            
            full_answer = ""
            for text_chunk in answer_generator.generate_answer_streaming(
                query, 
                similar_chunks, 
                model=model,
                use_context=use_context
            ):
                full_answer += text_chunk
                answer_container.markdown(full_answer)
            
            status_container.success(f"Completed streaming from {model}")
    
    @staticmethod
    def _handle_concurrent_responses(query, similar_chunks, selected_models,
                                   model_placeholders, result_placeholders,
                                   answer_generator, use_context):
        """
        Handle concurrent responses for multiple models.
        
        Args:
            query: User query
            similar_chunks: List of similar chunks
            selected_models: List of selected models
            model_placeholders: Dict of model placeholders
            result_placeholders: Dict of result placeholders
            answer_generator: Answer generator instance
            use_context: Whether to use document context
            
        Returns:
            List of sources or None
        """
        # Submit all tasks concurrently and get futures
        futures = answer_generator.generate_answers_concurrently(
            query, 
            similar_chunks, 
            selected_models,
            use_context=use_context
        )
        
        all_sources = None
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                # Get the result
                result = future.result()
                model = result["model"]
                
                # Update UI with the result
                if result["status"] == "Completed":
                    model_placeholders[model].success(f"Completed in {result['time_taken']:.2f}s")
                    result_placeholders[model].markdown(result["answer"])
                    
                    # Store sources from first completed model (only if using context)
                    if use_context and all_sources is None and "sources" in result:
                        all_sources = result["sources"]
                else:
                    model_placeholders[model].error(f"Error with {model}")
                    result_placeholders[model].error(result["answer"])
                    
            except Exception as e:
                logger.error(f"Error processing model result: {str(e)}")
                st.error(f"Error processing a model: {str(e)}")
        
        return all_sources
    
    @staticmethod
    def _display_source_documents(sources):
        """
        Display source documents in an expander.
        
        Args:
            sources: List of sources
        """
        with st.expander("View Source Documents"):
            # Group sources by document
            doc_sources = {}
            for source in sources:
                doc = source['document']
                if doc not in doc_sources:
                    doc_sources[doc] = []
                doc_sources[doc].append(source)
            
            # Display sources by document
            for doc, chunks in doc_sources.items():
                st.markdown(f"### Document: {doc}")
                # Sort chunks by relevance score
                sorted_chunks = sorted(chunks, key=lambda x: x['score'], reverse=True)
                for i, source in enumerate(sorted_chunks):
                    st.markdown(f"**Excerpt {i+1} (Relevance: {source['score']:.2f})**")
                    st.text_area(
                        f"Excerpt {i+1}", 
                        value=source['text'], 
                        height=150, 
                        disabled=True,
                        key=f"source_{doc}_{i}"
                    )

    @staticmethod
    def create_main_page_layout(config_manager, 
                               text_extractor, 
                               text_chunker, 
                               embedding_creator, 
                               qdrant_manager, 
                               answer_generator) -> None:
        """
        Create the main page layout with tabs.
        
        Args:
            config_manager: Configuration manager instance
            text_extractor: Text extractor instance
            text_chunker: Text chunker instance
            embedding_creator: Embedding creator instance
            qdrant_manager: Qdrant manager instance
            answer_generator: Answer generator instance
        """
        # Set page configuration
        st.set_page_config(
            page_title="DocumentGPT: Graph RAG",
            page_icon="📚",
            layout="wide"
        )
        
        # Initialize session state values if not present
        if "show_collection_manager" not in st.session_state:
            st.session_state.show_collection_manager = False
        
        # Create sidebar
        UIComponents.create_sidebar(config_manager)
        
        # Main content
        st.title("📚 DocumentGPT: Advanced RAG Assistant")
        st.markdown("""
        This application helps you quickly search and ask questions about your documents using a Graph RAG approach.
        
        ### Features:
        1. Upload and process PDF documents in the **Upload Documents** tab
        2. Ask questions about your documents in the **Ask Questions** tab
        3. Multiple assistant modes for different use cases (academic, legal, technical, research)
        4. Option to ask direct questions without document context
        5. Compare answers from multiple LLM models
        """)
        
        # Create tabs
        tab1, tab2 = st.tabs(["📄 Upload Documents", "❓ Ask Questions"])
        
        # Upload Documents Tab
        with tab1:
            UIComponents.create_upload_tab(
                config_manager,
                text_extractor,
                text_chunker,
                embedding_creator,
                qdrant_manager
            )
        
        # Ask Questions Tab
        with tab2:
            UIComponents.create_question_tab(
                config_manager,
                embedding_creator,
                qdrant_manager,
                answer_generator
            )
        
        # Footer
        st.markdown("---")
        st.markdown("DocumentGPT: Advanced RAG Assistant © 2025")