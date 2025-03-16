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
            
            # Collection information
            st.subheader("Collection Info")
            st.write(f"Current collection: `{config_manager.collection_name}`")
            
            # Option to reset collection
            if st.button("Create New Collection"):
                config_manager.reset_collection()
                st.success(f"Created new collection: {config_manager.collection_name}")
                st.rerun()
            
            # Sync button to check what files are in Qdrant
            if st.button("Sync Files from Database"):
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
                        st.error(f"Failed to sync files: {str(e)}")
            
            # Prompt Templates
            st.subheader("Prompt Templates")
            selected_template = st.selectbox(
                "Select Prompt Template", 
                options=list(config_manager.prompt_templates.keys()),
                index=list(config_manager.prompt_templates.keys()).index(config_manager.selected_template)
            )
            
            # Show the selected template
            st.text_area(
                "Template Text",
                value=config_manager.prompt_templates[selected_template],
                height=150
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
            
            # API Key inputs
            st.subheader("API Keys")
            openai_key = st.text_input(
                "OpenAI API Key", 
                value=config_manager.api_keys["OPENAI_API_KEY"], 
                type="password"
            )
            
            openrouter_key = st.text_input(
                "OpenRouter API Key", 
                value=config_manager.api_keys["OPENROUTER_API_KEY"], 
                type="password"
            )
            
            # Qdrant configuration
            st.subheader("Qdrant Configuration")
            use_cloud_qdrant = st.checkbox(
                "Use Cloud Qdrant", 
                value=bool(config_manager.api_keys["QDRANT_URL"])
            )
            
            if use_cloud_qdrant:
                qdrant_url = st.text_input(
                    "Qdrant URL", 
                    value=config_manager.api_keys["QDRANT_URL"]
                )
                
                qdrant_api_key = st.text_input(
                    "Qdrant API Key", 
                    value=config_manager.api_keys["QDRANT_API_KEY"], 
                    type="password"
                )
            else:
                st.info("Using in-memory Qdrant instance (data will be lost when app restarts)")
                qdrant_url = ""
                qdrant_api_key = ""
            
            # Update API keys
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
            
            # Display processed files
            st.subheader("Processed Files")
            if config_manager.processed_files:
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
        
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        
        # Check for duplicates and provide feedback
        if uploaded_files:
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
        
        # Chunking options
        with st.expander("Advanced Options"):
            chunk_size = st.slider("Chunk Size", min_value=200, max_value=2000, value=700, 
                                  help="Number of characters in each chunk (700 recommended for academic material)")
            chunk_overlap = st.slider("Chunk Overlap", min_value=50, max_value=500, value=175,
                                     help="Number of overlapping characters between chunks (175 recommended)")
        
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
        
        if uploaded_files:
            if st.button("Process Documents"):
                # Check API keys first
                if not config_manager.api_keys["OPENAI_API_KEY"]:
                    st.error("OpenAI API Key is required for processing documents")
                elif not qdrant_manager.initialize_collection(config_manager.collection_name):
                    st.error("Failed to initialize Qdrant collection")
                else:
                    # Use only new files for processing
                    file_names = [file.name for file in uploaded_files]
                    new_files, _ = config_manager.check_duplicate_files(file_names)
                    
                    # Map back to file objects
                    new_file_objects = [f for f in uploaded_files if f.name in new_files]
                    
                    if not new_file_objects:
                        st.info("No new files to process. All uploaded files have already been processed.")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        successful_files = 0
                        
                        # Set OpenAI API key for embedding creation
                        embedding_creator.set_api_key(config_manager.api_keys["OPENAI_API_KEY"])
                        
                        for i, uploaded_file in enumerate(new_file_objects):
                            file_name = uploaded_file.name
                            status_text.text(f"Processing {file_name}...")
                            
                            try:
                                # Extract text from PDF
                                text, error = text_extractor.extract_text_from_pdf(uploaded_file, file_name)
                                
                                if error:
                                    st.error(f"Error processing {file_name}: {error}")
                                    progress_bar.progress((i + 1) / len(new_file_objects))
                                    continue
                                
                                # Split text into chunks
                                chunks = text_chunker.split_text_into_chunks(
                                    text, 
                                    chunk_size=chunk_size, 
                                    overlap=chunk_overlap
                                )
                                
                                if not chunks:
                                    st.warning(f"No text chunks extracted from {file_name}")
                                    progress_bar.progress((i + 1) / len(new_file_objects))
                                    continue
                                
                                # Create embeddings
                                status_text.text(f"Creating embeddings for {file_name}...")
                                embeddings = embedding_creator.create_embeddings(chunks)
                                
                                if not embeddings:
                                    st.error(f"Failed to create embeddings for {file_name}")
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
                                
                                progress_bar.progress((i + 1) / len(new_file_objects))
                                
                            except Exception as e:
                                st.error(f"Error processing {file_name}: {str(e)}")
                                logger.error(f"Error processing {file_name}: {str(e)}")
                                progress_bar.progress((i + 1) / len(new_file_objects))
                        
                        progress_bar.progress(1.0)
                        status_text.text("Processing complete!")
                        
                        if successful_files > 0:
                            st.success(f"Successfully processed {successful_files} out of {len(new_file_objects)} documents")
                            
                            # Save updated file list to config
                            config_manager.save_config()
                        else:
                            st.error("Failed to process any documents")
    
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
        
        # Set system prompt based on selected template
        current_template = config_manager.selected_template
        template_text = config_manager.get_prompt_template(current_template)
        answer_generator.set_system_prompt(template_text)
        
        # Display current assistant type
        st.info(f"Current assistant: **{current_template}**")
        
        query = st.text_area("Enter your question:", height=100)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
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
                default=[available_models[0]]  # Default to first model
            )
            
            if not selected_models:
                st.warning("Please select at least one model")
        
        with col2:
            # Option to enable/disable RAG context
            use_context = st.checkbox("Include document context", value=True)
            if not use_context:
                st.warning("Context is disabled. Answers will not be based on your documents.")
                search_limit = 0
            else:
                search_limit = st.slider("Number of chunks to retrieve:", min_value=3, max_value=25, value=10)
            
            # New option for streaming responses
            enable_streaming = st.checkbox("Enable streaming responses", value=True)
        
        # Show a "Direct Query" button if not using context
        if not use_context and st.button("Ask Directly") and query and selected_models:
            # Check API keys
            if not config_manager.api_keys["OPENROUTER_API_KEY"]:
                st.error("OpenRouter API key is required")
            else:
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
                        
                        # Process with streaming or concurrent generation based on user choice
                        if enable_streaming and len(selected_models) == 1:
                            # Single model with streaming
                            model = selected_models[0]
                            with model_tabs[0]:
                                answer_container = result_placeholders[model]
                                status_container = model_placeholders[model]
                                status_container.info(f"Streaming response from {model}...")
                                
                                full_answer = ""
                                for text_chunk in answer_generator.generate_answer_streaming(
                                    query, 
                                    use_context=False,
                                    model=model
                                ):
                                    full_answer += text_chunk
                                    answer_container.markdown(full_answer)
                                
                                status_container.success(f"Completed streaming from {model}")
                        else:
                            # Submit all tasks concurrently and get futures
                            futures = answer_generator.generate_answers_concurrently(
                                query, 
                                use_context=False,
                                models=selected_models
                            )
                            
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
                                    else:
                                        model_placeholders[model].error(f"Error with {model}")
                                        result_placeholders[model].error(result["answer"])
                                        
                                except Exception as e:
                                    st.error(f"Error processing a model: {str(e)}")
                    
                    except Exception as e:
                        st.error(f"Error processing your query: {str(e)}")
                        logger.error(f"Error processing query: {str(e)}")
        
        # Show "Get Answer with Context" button if using context
        elif use_context and st.button("Get Answer with Context") and query and selected_models:
            # Check API keys
            if not config_manager.api_keys["OPENAI_API_KEY"] or not config_manager.api_keys["OPENROUTER_API_KEY"]:
                st.error("Both OpenAI and OpenRouter API keys are required")
            elif not config_manager.processed_files:
                st.warning("No documents have been processed. Please upload and process documents first.")
            else:
                with st.spinner("Processing query..."):
                    try:
                        # Set API keys
                        embedding_creator.set_api_key(config_manager.api_keys["OPENAI_API_KEY"])
                        answer_generator.set_api_key(config_manager.api_keys["OPENROUTER_API_KEY"])
                        
                        # Create embedding for query
                        query_embedding = embedding_creator.create_query_embedding(query)
                        
                        # Search for similar chunks
                        similar_chunks = qdrant_manager.search_similar_chunks(
                            config_manager.collection_name,
                            query_embedding,
                            limit=search_limit
                        )
                        
                        if similar_chunks:
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
                            
                            # Process with streaming or concurrent generation based on user choice
                            if enable_streaming and len(selected_models) == 1:
                                # Single model with streaming
                                model = selected_models[0]
                                with model_tabs[0]:
                                    answer_container = result_placeholders[model]
                                    status_container = model_placeholders[model]
                                    status_container.info(f"Streaming response from {model}...")
                                    
                                    full_answer = ""
                                    for text_chunk in answer_generator.generate_answer_streaming(
                                        query, 
                                        similar_chunks, 
                                        model=model,
                                        use_context=True
                                    ):
                                        full_answer += text_chunk
                                        answer_container.markdown(full_answer)
                                    
                                    status_container.success(f"Completed streaming from {model}")
                                    
                                    # Get sources for the single model (we need to run the non-streaming version to get sources)
                                    _, all_sources = answer_generator.generate_answer(
                                        query, 
                                        similar_chunks, 
                                        model=model,
                                        use_context=True
                                    )
                            else:
                                # Submit all tasks concurrently and get futures
                                futures = answer_generator.generate_answers_concurrently(
                                    query, 
                                    similar_chunks, 
                                    selected_models,
                                    use_context=True
                                )
                                
                                # Initialize sources
                                all_sources = None
                                
                                # Process results as they complete (in the main thread)
                                for future in concurrent.futures.as_completed(futures):
                                    try:
                                        # Get the result
                                        result = future.result()
                                        model = result["model"]
                                        
                                        # Update UI with the result (safe in main thread)
                                        if result["status"] == "Completed":
                                            model_placeholders[model].success(f"Completed in {result['time_taken']:.2f}s")
                                            result_placeholders[model].markdown(result["answer"])
                                            
                                            # Store sources from first completed model
                                            if all_sources is None:
                                                all_sources = result["sources"]
                                        else:
                                            model_placeholders[model].error(f"Error with {model}")
                                            result_placeholders[model].error(result["answer"])
                                            
                                    except Exception as e:
                                        # If we can't determine which model had an error, show a general error
                                        st.error(f"Error processing a model: {str(e)}")
                                                            
                            # Display sources in an expander outside the tabs
                            if all_sources:
                                with st.expander("View Source Documents"):
                                    # Group sources by document
                                    doc_sources = {}
                                    for source in all_sources:
                                        doc = source['document']
                                        if doc not in doc_sources:
                                            doc_sources[doc] = []
                                        doc_sources[doc].append(source)
                                    
                                    # Display sources by document
                                    for doc, chunks in doc_sources.items():
                                        st.markdown(f"### Document: {doc}")
                                        for i, source in enumerate(chunks):
                                            st.markdown(f"**Excerpt {i+1} (Relevance: {source['score']:.2f})**")
                                            st.text_area(f"Excerpt {i+1}", value=source['text'], height=150, disabled=True)
                        else:
                            st.error("No relevant information found in the documents.")
                            
                    except Exception as e:
                        st.error(f"Error processing your query: {str(e)}")
                        logger.error(f"Error processing query: {str(e)}")
    
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
        st.markdown("DocumentGPT: Advanced RAG Assistant | Made with ❤️ and 📚")