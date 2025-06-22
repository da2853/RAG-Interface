"""
Streamlit UI components for Document Search Engine application.
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

ICON_CONFIG = "⚙️"
ICON_COLLECTION = "🗂️"
ICON_SEARCH_PARAMS = "📋"
ICON_KEY = "🔑"
ICON_FILE = "📄"
ICON_UPLOAD = "☁️"
ICON_SEARCH = "🔍"
ICON_PROCESS = "⏳"
ICON_SYNC = "🔄"
ICON_DELETE = "🗑️"
ICON_SWITCH = "↔️"
ICON_SAVE = "💾"
ICON_INFO = "ℹ️"
ICON_WARNING = "⚠️"
ICON_ERROR = "❌"
ICON_SUCCESS = "✅"
ICON_ENGINE = "🔧"
ICON_SOURCE = ""
ICON_REALTIME = "⚡"

class UIComponents:
    """
    Provides reusable UI components and layouts for the Streamlit interface.
    """

    @staticmethod
    def initialize_session_state():
        """Initialize session state variables."""
        defaults = {
            "show_collection_manager": False,
            "query_results": {},
            "query_in_progress": False,
            "current_query": None,
            "current_models": [],
            "processed_files_status": {}, # Store status per file during upload
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def create_sidebar(config_manager) -> None:
        """ Create the sidebar for configuration settings. """
        with st.sidebar:
            st.title(f"{ICON_CONFIG} Configuration")

            # API Key inputs section
            UIComponents._create_api_keys_section(config_manager)

            st.divider()

            # Collection management section
            UIComponents._create_collection_management_section(config_manager)

            st.divider()

            # Prompt Templates section
            UIComponents._create_prompt_templates_section(config_manager)

            st.divider()

            # Display processed files
            UIComponents._display_processed_files(config_manager)

    @staticmethod
    def _create_collection_management_section(config_manager) -> None:
        """ Create the collection management section in the sidebar. """
        st.subheader(f"{ICON_COLLECTION} Collection Management")
        
        current_collection = config_manager.collection_name
        st.write(f"Current Collection: `{current_collection}`")

        col1, col2 = st.columns([1,1])
        with col1:
            if st.button(f"{ICON_SYNC} Sync Files", help="Fetch the list of processed files from the current vector database collection."):
                UIComponents._sync_files_from_database(config_manager)

        # Defer Qdrant connection until needed
        qdrant_manager = None
        try:
            from vector_db.qdrant_manager import QdrantManager
            qdrant_url = config_manager.api_keys.get("QDRANT_URL")
            qdrant_api_key = config_manager.api_keys.get("QDRANT_API_KEY")

            # Only try to connect if Qdrant is configured or not using cloud (in-memory)
            if qdrant_url or not qdrant_url: # Checks if cloud is configured or if using in-memory
                 qdrant_manager = QdrantManager(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    show_status=False
                )
                 collections = qdrant_manager.list_collections()
                 if collections is not None: # Check if connection was successful
                     with col2:
                         if st.button(f"{ICON_COLLECTION} Manage"):
                             st.session_state.show_collection_manager = not st.session_state.get("show_collection_manager", False)
                 else:
                     st.warning(f"{ICON_WARNING} Could not list collections. Check Qdrant connection.", icon="⚠️")

        except ImportError:
             st.warning("Qdrant library not found.", icon=ICON_WARNING)
        except Exception as e:
            logger.error(f"Error interacting with Qdrant for collection management: {str(e)}")
            st.error(f"{ICON_ERROR} Error connecting to Qdrant: Check URL/Key.")


        # Add collection manager UI if requested and manager is available
        if st.session_state.get("show_collection_manager", False) and qdrant_manager:
            UIComponents._create_collection_manager_ui(config_manager, qdrant_manager)


    @staticmethod
    def _create_collection_manager_ui(config_manager, qdrant_manager) -> None:
        """ Create the collection manager UI within an expander. """
        with st.expander("Collection Manager", expanded=True):
            st.subheader("Available Collections")

            try:
                collections = qdrant_manager.list_collections()
                if collections is None:
                    st.error(f"{ICON_ERROR} Failed to retrieve collections.")
                    return

                if not collections:
                    st.info("No existing collections found.")

                current_collection = config_manager.collection_name
                for collection in collections:
                    cols = st.columns([3, 1, 1])
                    with cols[0]:
                        st.markdown(f"`{collection}` {'**(Current)**' if collection == current_collection else ''}")
                    with cols[1]:
                        if collection != current_collection:
                            if st.button(ICON_SWITCH, key=f"switch_{collection}", help=f"Switch to '{collection}'"):
                                config_manager.switch_collection(collection)
                                st.session_state.show_collection_manager = False
                                st.success(f"Switched to collection: {collection}")
                                st.rerun() # Rerun to update UI immediately
                    with cols[2]:
                         # Add confirmation for deletion
                        delete_key = f"delete_{collection}"
                        confirm_key = f"confirm_delete_{collection}"

                        if st.button(ICON_DELETE, key=delete_key, help=f"Delete '{collection}'"):
                             st.session_state[confirm_key] = True # Set flag to show confirmation

                        if st.session_state.get(confirm_key, False):
                            st.warning(f"Really delete `{collection}`? This cannot be undone.")
                            if st.button("Confirm Deletion", key=f"confirm_btn_{collection}"):
                                UIComponents._handle_collection_deletion(config_manager, qdrant_manager, collection)
                                del st.session_state[confirm_key] # Clear confirmation flag
                                st.rerun() # Rerun to update list
                            if st.button("Cancel", key=f"cancel_btn_{collection}"):
                                del st.session_state[confirm_key] # Clear confirmation flag
                                st.rerun()


                st.divider()
                st.subheader("Create New Collection")
                with st.form("new_collection_form"):
                    new_collection_name = st.text_input("New Collection Name (leave blank for default)")
                    create_submit = st.form_submit_button(f"{ICON_SAVE} Create Collection")

                    if create_submit:
                        # Use default name generation if input is empty
                        name_to_create = new_collection_name if new_collection_name else None
                        created_name = config_manager.create_new_collection(name_to_create)

                        if qdrant_manager.initialize_collection(created_name):
                            st.success(f"{ICON_SUCCESS} Created and switched to collection: `{created_name}`")
                            st.session_state.show_collection_manager = False
                            st.rerun()
                        else:
                            st.error(f"{ICON_ERROR} Failed to initialize collection in Qdrant.")
                            # Attempt to revert config change if Qdrant failed
                            try:
                                # This assumes create_new_collection switched the config
                                config_manager.switch_collection(config_manager.collection_name) # Switch back? Risky if prev name wasn't stored
                                logger.warning(f"Attempted to revert config after failed Qdrant init for {created_name}")
                            except Exception as revert_e:
                                logger.error(f"Failed to revert config after Qdrant error: {revert_e}")


            except Exception as e:
                logger.error(f"Error in collection manager UI: {str(e)}")
                st.error(f"{ICON_ERROR} Error managing collections: {str(e)}")

    @staticmethod
    def _handle_collection_deletion(config_manager, qdrant_manager, collection) -> None:
        """ Handle the deletion of a collection with feedback. """
        is_current = (collection == config_manager.collection_name)
        if is_current:
            st.warning("Deleting the current collection. A new default collection will be created.", icon=ICON_WARNING)

        with st.spinner(f"Deleting collection '{collection}'..."):
            deleted = qdrant_manager.delete_collection(collection)

        if deleted:
            st.success(f"{ICON_SUCCESS} Deleted collection: `{collection}`")
            # If we deleted the current collection, create and switch to a new default one
            if is_current:
                try:
                    new_name = config_manager.create_new_collection() # Creates default name
                    if qdrant_manager.initialize_collection(new_name):
                         st.info(f"Created and switched to new default collection: `{new_name}`", icon=ICON_INFO)
                    else:
                         st.error(f"{ICON_ERROR} Deleted current collection but failed to initialize a new one in Qdrant!")
                         # Config might be in an inconsistent state here
                except Exception as e:
                     st.error(f"{ICON_ERROR} Error creating new default collection: {e}")
            # Refresh processed files list as it might change if collection changed
            UIComponents._sync_files_from_database(config_manager)
        else:
            st.error(f"{ICON_ERROR} Failed to delete collection: `{collection}`")


    @staticmethod
    def _sync_files_from_database(config_manager) -> None:
        """ Sync files from the Qdrant database with status indicator. """
        status = st.status(f"{ICON_PROCESS} Syncing files from database...", expanded=False)
        try:
            from vector_db.qdrant_manager import QdrantManager
            qdrant_manager = QdrantManager(
                url=config_manager.api_keys.get("QDRANT_URL"),
                api_key=config_manager.api_keys.get("QDRANT_API_KEY"),
                show_status=False
            )

            files = qdrant_manager.get_processed_files(config_manager.collection_name)
            if files is not None: # Check for connection/retrieval errors
                config_manager.processed_files = files
                config_manager.save_config()
                status.update(label=f"{ICON_SUCCESS} Synced {len(files)} files from '{config_manager.collection_name}'.", state="complete")
                st.rerun() # Rerun to update the displayed list
            else:
                 status.update(label=f"{ICON_ERROR} Failed to sync: Could not retrieve file list.", state="error")

        except ImportError:
            status.update(label=f"{ICON_ERROR} Qdrant library not found.", state="error")
        except Exception as e:
            logger.error(f"Failed to sync files: {str(e)}")
            status.update(label=f"{ICON_ERROR} Failed to sync files: {str(e)}", state="error")

    @staticmethod
    def _create_prompt_templates_section(config_manager) -> None:
        """ Create the search parameters section in the sidebar. """
        st.subheader(f"{ICON_SEARCH_PARAMS} Search Parameters")

        template_options = list(config_manager.prompt_templates.keys())
        try:
            current_template_index = template_options.index(config_manager.selected_template)
        except ValueError:
            # Handle case where selected template is no longer valid (e.g., removed)
            current_template_index = 0
            config_manager.set_selected_template(template_options[0])
            logger.warning(f"Selected template '{config_manager.selected_template}' not found. Resetting to '{template_options[0]}'.")


        selected_template = st.selectbox(
            "Active Prompt Template:",
            options=template_options,
            index=current_template_index,
            help="Select the system prompt used by the AI for answering questions."
        )

        # Update selection if changed
        if selected_template != config_manager.selected_template:
            config_manager.set_selected_template(selected_template)
            config_manager.save_config()
            st.success(f"Active template set to: {selected_template}")
            # No rerun needed, selectbox handles state

        st.text_area(
            "Template Preview:",
            value=config_manager.prompt_templates[selected_template],
            height=150,
            disabled=True,
            key=f"template_preview_{selected_template}" # Ensure key changes on selection
        )

        with st.expander("Edit Custom Template"):
            custom_template_text = st.text_area(
                "Custom Template Content:",
                value=config_manager.prompt_templates.get("Custom", ""), # Use .get for safety
                height=200,
                key="custom_template_editor"
            )

            if st.button(f"{ICON_SAVE} Save Custom Template"):
                if "Custom" not in template_options and not custom_template_text:
                     st.warning("Cannot save an empty initial Custom template.", icon=ICON_WARNING)
                else:
                    config_manager.add_prompt_template("Custom", custom_template_text)
                    config_manager.save_config()
                    st.success("Custom template saved.")
                    # If 'Custom' was just selected, refresh the preview area implicitly via rerun
                    if config_manager.selected_template == "Custom":
                        st.rerun()


    @staticmethod
    def _create_api_keys_section(config_manager) -> None:
        """ Create the service configuration section in the sidebar with better layout. """
        st.subheader(f"{ICON_KEY} Service Configuration")

        with st.container(border=True):
             st.markdown("**Search Services (Required)**")
             openai_key = st.text_input(
                "Indexing Service Key",
                value=config_manager.api_keys.get("OPENAI_API_KEY", ""),
                type="password",
                help="Required for creating document embeddings during document indexing."
            )
             openrouter_key = st.text_input(
                "Search Service Key",
                value=config_manager.api_keys.get("OPENROUTER_API_KEY", ""),
                type="password",
                help="Required for performing document searches using various search engines."
            )
             # Visual indicators for missing keys
             if not openai_key: st.warning("Indexing service key needed for document processing.", icon=ICON_WARNING)
             if not openrouter_key: st.warning("Search service key needed for document search.", icon=ICON_WARNING)


        with st.container(border=True):
            st.markdown("**Vector Database (Qdrant)**")
            use_cloud_qdrant = st.toggle(
                "Use Cloud Qdrant Instance",
                value=bool(config_manager.api_keys.get("QDRANT_URL")), # Base toggle on URL presence
                help="Enable to connect to a hosted Qdrant service. If disabled, an in-memory database (lost on restart) is used."
            )

            qdrant_url = ""
            qdrant_api_key = ""

            if use_cloud_qdrant:
                qdrant_url = st.text_input(
                    "Qdrant URL",
                    value=config_manager.api_keys.get("QDRANT_URL", ""),
                    help="The full URL of your Qdrant cloud instance (e.g., https://your-instance.qdrant.cloud:6333)."
                )
                qdrant_api_key = st.text_input(
                    "Qdrant API Key",
                    value=config_manager.api_keys.get("QDRANT_API_KEY", ""),
                    type="password",
                    help="API key for your Qdrant cloud instance (if required)."
                )
                if not qdrant_url: st.warning("Qdrant URL is required for cloud mode.", icon=ICON_WARNING)
            else:
                st.info("Using temporary in-memory Qdrant. Data will be lost when the app stops.", icon=ICON_INFO)


        if st.button(f"{ICON_SAVE} Update Configuration"):
            # Update keys based on inputs
            config_manager.update_api_key("OPENAI_API_KEY", openai_key)
            config_manager.update_api_key("OPENROUTER_API_KEY", openrouter_key)
            # Only update Qdrant keys if cloud mode is enabled
            config_manager.update_api_key("QDRANT_URL", qdrant_url if use_cloud_qdrant else "")
            config_manager.update_api_key("QDRANT_API_KEY", qdrant_api_key if use_cloud_qdrant else "")

            if config_manager.save_config():
                st.success("Configuration updated and saved!")
                # Optionally trigger connection checks or updates elsewhere if needed
            else:
                st.warning("Configuration updated in memory but failed to save to disk.", icon=ICON_WARNING)

    @staticmethod
    def _display_processed_files(config_manager) -> None:
        """ Display the processed files in the sidebar expander. """
        st.subheader(f"{ICON_FILE} Processed Files in `{config_manager.collection_name}`")

        if config_manager.processed_files:
            with st.expander(f"View {len(config_manager.processed_files)} Files", expanded=False):
                # Use a scrollable container for potentially long lists
                with st.container(height=200):
                    for file in sorted(config_manager.processed_files): # Sort for consistency
                        st.caption(f"- {file}") # Use caption for less emphasis
        else:
            st.caption("No files processed in this collection yet. Use the 'Sync Files' button or upload new documents.")

    @staticmethod
    def create_upload_tab(config_manager,
                         text_extractor,
                         text_chunker,
                         embedding_creator,
                         qdrant_manager) -> None:
        """ Create the document upload tab UI. """
        st.header(f"{ICON_UPLOAD} Upload & Process Documents")
        st.markdown("Upload PDF documents here to make them searchable.")

        # Check for required service keys early
        api_key_status = UIComponents._check_api_key_status(config_manager)
        if not api_key_status["openai"]:
            st.warning(f"{ICON_WARNING} Indexing service key is required for processing documents. Please add it in the Configuration sidebar.", icon="⚠️")
        if not api_key_status["qdrant_ready"]:
             st.warning(f"{ICON_WARNING} Database service is not configured correctly (URL/Key missing for cloud mode?). Document processing might fail.", icon="⚠️")

        # File uploader
        uploaded_files = st.file_uploader(
            "Choose PDF files to upload:",
            type="pdf",
            accept_multiple_files=True,
            help="You can upload multiple PDF files at once."
            )

        new_file_objects = []
        if uploaded_files:
            new_file_objects = UIComponents._check_duplicate_files(config_manager, uploaded_files)

        col1, col2 = st.columns(2)
        with col1:
            chunk_settings = UIComponents._create_chunking_settings()
            chunk_size = chunk_settings["chunk_size"]
            chunk_overlap = chunk_settings["chunk_overlap"]
        with col2:
             # Add a placeholder or other controls if needed
             st.markdown(" ") # Placeholder for alignment
             st.markdown(" ")
             # Process documents button - enabled only if keys are present and new files exist
             can_process = api_key_status["openai"] and api_key_status["qdrant_ready"] and new_file_objects
             if st.button(f"{ICON_PROCESS} Process {len(new_file_objects)} New Documents", disabled=not can_process):
                 # Reset processing status for new batch
                 st.session_state.processed_files_status = {f.name: "queued" for f in new_file_objects}

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

        # Display detailed status during/after processing
        if st.session_state.get("processed_files_status"):
             st.divider()
             st.subheader("Processing Status")
             for filename, status_info in st.session_state.processed_files_status.items():
                 if isinstance(status_info, dict): # Detailed status
                      if status_info["state"] == "processing":
                           st.info(f"{ICON_PROCESS} {filename}: {status_info['message']}", icon="⏳")
                      elif status_info["state"] == "success":
                           st.success(f"{ICON_SUCCESS} {filename}: {status_info['message']}", icon="✅")
                      elif status_info["state"] == "error":
                           st.error(f"{ICON_ERROR} {filename}: {status_info['message']}", icon="❌")
                 elif status_info == "queued": # Initial state
                      st.info(f"{ICON_FILE} {filename}: Queued...", icon="📄")


    @staticmethod
    def _check_api_key_status(config_manager) -> Dict[str, bool]:
        """ Check the status of required API keys and Qdrant readiness. """
        openai_ok = bool(config_manager.api_keys.get("OPENAI_API_KEY"))
        openrouter_ok = bool(config_manager.api_keys.get("OPENROUTER_API_KEY"))

        # Qdrant readiness check
        qdrant_url = config_manager.api_keys.get("QDRANT_URL")
        qdrant_ready = True # Assume in-memory is always ready
        if qdrant_url: # If cloud mode is intended
             # Basic check: URL must be present for cloud mode
             if not qdrant_url:
                  qdrant_ready = False
             # Could add a quick connection test here if needed, but might slow down UI
             # For now, just check if URL is set when cloud is expected

        return {
            "openai": openai_ok,
            "openrouter": openrouter_ok,
            "qdrant_ready": qdrant_ready, # True if in-memory or cloud URL is set
        }


    @staticmethod
    def _check_duplicate_files(config_manager, uploaded_files) -> List:
        """ Check for duplicate files and provide user feedback. """
        file_names = [file.name for file in uploaded_files]
        # Ensure processed_files is treated as a set for efficient checking
        processed_set = set(config_manager.processed_files)
        new_files = [name for name in file_names if name not in processed_set]
        duplicates = [name for name in file_names if name in processed_set]

        new_file_objects = [f for f in uploaded_files if f.name in new_files]

        if duplicates:
            st.warning(f"{ICON_WARNING} Already processed: {', '.join(duplicates)}", icon="⚠️")

        if not new_file_objects and duplicates:
            st.info("All uploaded files have already been processed in the current collection.", icon=ICON_INFO)
        elif new_file_objects:
            st.info(f"{ICON_SUCCESS} {len(new_file_objects)} new file(s) ready for processing.", icon="📄")

        return new_file_objects

    @staticmethod
    def _create_chunking_settings() -> Dict[str, int]:
        """ Create the chunking settings UI in an expander. """
        with st.expander("Advanced Chunking Options"):
            cols = st.columns(2)
            with cols[0]:
                chunk_size = st.slider(
                    "Chunk Size (characters)",
                    min_value=200,
                    max_value=2000,
                    value=700,
                    step=50,
                    help="Size of text chunks. Smaller chunks capture detail, larger ones capture broader context. (Default: 700)"
                )
            with cols[1]:
                # Calculate max overlap based on chunk size to prevent issues
                max_overlap = max(50, chunk_size // 2) # Ensure overlap is not too large
                chunk_overlap = st.slider(
                    "Chunk Overlap (characters)",
                    min_value=50,
                    max_value=max_overlap,
                    value=min(175, max_overlap), # Adjust default if max_overlap is smaller
                    step=25,
                    help="Characters shared between consecutive chunks. Helps maintain context. (Default: 175 or 50% of size)"
                )

            # Guide moved inside expander
            st.markdown("""
            **Chunking Guide:**
            *   **Academic/General:** Size 600-800, Overlap ~25%
            *   **Dense Technical:** Size 400-600, Overlap ~25-30%
            *   **Narrative/Broad:** Size 800-1200, Overlap ~20-25%
            Adjust based on your specific document content.
            """)

        return {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}


    @staticmethod
    def _process_documents(config_manager, qdrant_manager, text_extractor, text_chunker,
                          embedding_creator, new_file_objects, chunk_size, chunk_overlap) -> None:
        """ Process documents with improved status feedback using st.status. """
        if not new_file_objects:
            st.info("No new files selected for processing.", icon=ICON_INFO)
            return

        # Check prerequisites again before starting
        api_key_status = UIComponents._check_api_key_status(config_manager)
        if not api_key_status["openai"]:
            st.error("OpenAI API Key is missing. Cannot process documents.", icon=ICON_ERROR)
            return
        if not api_key_status["qdrant_ready"]:
             st.error("Qdrant not configured. Cannot store document data.", icon=ICON_ERROR)
             return

        # Initialize Qdrant collection (should happen before processing files)
        try:
             if not qdrant_manager.initialize_collection(config_manager.collection_name):
                 st.error(f"Failed to initialize Qdrant collection '{config_manager.collection_name}'. Aborting.", icon=ICON_ERROR)
                 return
             logger.info(f"Ensured Qdrant collection '{config_manager.collection_name}' exists.")
        except Exception as q_init_e:
             st.error(f"Error initializing Qdrant collection: {q_init_e}", icon=ICON_ERROR)
             logger.error(f"Qdrant init error: {q_init_e}")
             return


        # Set OpenAI API key for embedding creator
        embedding_creator.set_api_key(config_manager.api_keys["OPENAI_API_KEY"])

        total_files = len(new_file_objects)
        processed_count = 0
        failed_files_summary = {} # Store filename: error message

        overall_status = st.status(f"{ICON_PROCESS} Starting document processing...", expanded=True)

        for i, uploaded_file in enumerate(new_file_objects):
            file_name = uploaded_file.name
            file_status_key = f"status_{file_name}"
            st.session_state.processed_files_status[file_name] = {"state": "processing", "message": "Starting..."}


            overall_status.update(label=f"{ICON_PROCESS} Processing file {i+1}/{total_files}: **{file_name}**")

            try:
                 # --- Text Extraction ---
                 st.session_state.processed_files_status[file_name]["message"] = "Extracting text..."
                 overall_status.write(f"📄 Extracting text from `{file_name}`...") # Update within status
                 text, error = text_extractor.extract_text_from_pdf(uploaded_file, file_name)
                 if error:
                      raise ValueError(f"Text extraction failed: {error}")
                 if not text:
                     raise ValueError("No text extracted from the PDF.")
                 overall_status.write(f"   {ICON_SUCCESS} Text extracted ({len(text)} chars).")

                 # --- Text Chunking ---
                 st.session_state.processed_files_status[file_name]["message"] = "Chunking text..."
                 overall_status.write(f"🔪 Chunking text...")
                 chunks = text_chunker.split_text_into_chunks(
                     text,
                     chunk_size=chunk_size,
                     overlap=chunk_overlap
                 )
                 if not chunks:
                     raise ValueError("Text chunking resulted in zero chunks.")
                 overall_status.write(f"   {ICON_SUCCESS} Created {len(chunks)} chunks.")

                 # --- Embedding Creation ---
                 st.session_state.processed_files_status[file_name]["message"] = "Creating embeddings..."
                 overall_status.write(f"🧠 Creating embeddings...")
                 # chunks is a list of strings, create_embeddings returns list of dicts with 'text' and 'embedding'
                 embeddings_data = embedding_creator.create_embeddings(chunks)
                 if not embeddings_data or len(embeddings_data) != len(chunks):
                      raise ValueError("Failed to create embeddings or mismatch in count.")
                 overall_status.write(f"   {ICON_SUCCESS} Embeddings created.")

                 # --- Storing in Qdrant ---
                 st.session_state.processed_files_status[file_name]["message"] = "Storing in database..."
                 overall_status.write(f"💾 Storing in Qdrant collection '{config_manager.collection_name}'...")
                 # Pass embeddings_data directly (already has correct structure with 'text' and 'embedding' keys)
                 if not qdrant_manager.store_embeddings(
                     config_manager.collection_name,
                     embeddings_data, # Use the embeddings_data returned by create_embeddings
                     metadata={"filename": file_name} # Common metadata (correct parameter name)
                 ):
                      raise RuntimeError("Failed to store embeddings in Qdrant.")
                 overall_status.write(f"   {ICON_SUCCESS} Stored successfully.")

                 # --- Finalize Success ---
                 config_manager.add_processed_file(file_name)
                 processed_count += 1
                 st.session_state.processed_files_status[file_name] = {"state": "success", "message": "Processed successfully"}

            except Exception as e:
                 logger.error(f"Error processing {file_name}: {str(e)}", exc_info=True)
                 error_message = f"Failed: {str(e)}"
                 failed_files_summary[file_name] = str(e)
                 st.session_state.processed_files_status[file_name] = {"state": "error", "message": error_message}
                 overall_status.write(f"   {ICON_ERROR} Error processing {file_name}: {str(e)}")


        # --- Processing Complete ---
        config_manager.save_config() # Save the updated list of processed files

        final_message = f"Processed {processed_count}/{total_files} documents."
        if failed_files_summary:
             final_message += f" ({len(failed_files_summary)} failed)"
             overall_status.update(label=final_message, state="error", expanded=True)
             st.error(f"Processing finished with {len(failed_files_summary)} errors.", icon=ICON_ERROR)
             with st.expander("Failed File Details"):
                  for fname, err in failed_files_summary.items():
                       st.error(f"**{fname}:** {err}", icon=ICON_ERROR)
        elif processed_count == 0 and total_files > 0:
             final_message = f"Failed to process any of the {total_files} documents."
             overall_status.update(label=final_message, state="error", expanded=True)
             st.error("Processing completed, but no documents were successfully processed.", icon=ICON_ERROR)
        else:
             overall_status.update(label=final_message, state="complete", expanded=False)
             st.success(f"Successfully processed {processed_count} documents.", icon=ICON_SUCCESS)

        # Optional: Clear the status dict after completion or keep for display
        # st.session_state.processed_files_status = {}


    @staticmethod
    def create_question_tab(config_manager,
                          embedding_creator,
                          qdrant_manager,
                          answer_generator) -> None:
        """ Create the document search tab UI. """
        st.header(f"{ICON_SEARCH} Search Documents")
        st.markdown("Search for information in your processed documents or access general knowledge when documents don't contain the answer.")

        # Check API keys
        api_key_status = UIComponents._check_api_key_status(config_manager)
        if not api_key_status["openrouter"]:
            st.warning(f"{ICON_WARNING} Search service API Key is required for document search. Please add it in the Configuration sidebar.", icon="⚠️")
            # Optionally disable the rest of the tab if key is missing
            # return

        # Set search parameters based on selected template
        current_template = config_manager.selected_template
        template_text = config_manager.get_prompt_template(current_template)
        answer_generator.set_system_prompt(template_text)
        st.info(f"Using Search Mode: **{current_template}**", icon=ICON_SEARCH_PARAMS)


        # Search Input Area
        query = st.text_area(
            "Enter your search query:",
            height=120,
            placeholder="Example: Find information about topic X from the uploaded documents."
        )

        # Settings Columns
        settings_col1, settings_col2 = st.columns(2)

        with settings_col1:
            st.subheader("Search Mode & Context")
            answer_mode = UIComponents._create_answer_mode_selection(config_manager)
            use_context = (answer_mode == "Search uploaded documents")

            # Context related settings only if document search mode is selected
            search_limit = 0
            if use_context:
                search_limit = st.slider(
                    f"{ICON_SOURCE} Max Document Excerpts:",
                    min_value=3,
                    max_value=25,
                    value=10,
                    help="How many relevant document excerpts to retrieve for context. More excerpts = more comprehensive results but potentially slower."
                )
                # Check if context can actually be used
                if not config_manager.processed_files:
                    st.warning(f"{ICON_WARNING} No documents processed in '{config_manager.collection_name}'. Document search cannot be performed.", icon="⚠️")
                if not api_key_status["openai"]:
                    st.warning(f"{ICON_WARNING} Indexing service key needed to search document embeddings.", icon="⚠️")


        with settings_col2:
            st.subheader("Search Engine Selection")
            selected_models = UIComponents._create_model_selection()
            enable_streaming = st.toggle(
                f"{ICON_REALTIME} Real-time Results",
                value=True,
                help="Show search results as they are generated. Only effective when a single search engine is selected."
            )
            # Warn if streaming is on but multiple models selected
            if enable_streaming and len(selected_models) > 1:
                 st.caption(f"{ICON_INFO} Real-time results work best with a single selected search engine.")


        st.divider()

        # Search Button - more checks for enabled state
        can_ask = bool(query) and bool(selected_models) and api_key_status["openrouter"]
        button_label = f"{ICON_SEARCH} Search Documents{' with Context' if use_context else ' (General)'}"

        # Additional checks for document search mode
        if use_context:
             rag_ready = api_key_status["openai"] and config_manager.processed_files and api_key_status["qdrant_ready"]
             if not rag_ready:
                  can_ask = False
                  st.error("Cannot search documents. Check: indexing service key, processed files, and database config.", icon=ICON_ERROR)

        # Store button click state
        ask_button_clicked = st.button(button_label, disabled=not can_ask, type="primary")


        # --- Answer Generation Logic ---
        if ask_button_clicked:
             # Clear previous results and set processing flag
             st.session_state.query_results = {model: {"status": "pending"} for model in selected_models}
             st.session_state.query_in_progress = True
             st.session_state.current_query = query
             st.session_state.current_models = selected_models
             st.rerun() # Rerun to start processing immediately

        # --- Display Results Area (persistent across reruns using session state) ---
        if st.session_state.get("query_in_progress") or st.session_state.get("query_results"):
             st.subheader(f"{ICON_ENGINE} Search Results")
             query_to_display = st.session_state.get("current_query", "N/A")
             st.markdown(f"**Search Query:** *{query_to_display}*")


             # Check if processing is still needed (might have been interrupted)
             if st.session_state.get("query_in_progress"):
                  # Check if all results are in, if so, flip the flag
                  all_done = all(res.get("status") not in ["pending", "running"]
                                 for res in st.session_state.query_results.values())
                  if all_done:
                      st.session_state.query_in_progress = False
                  else:
                       UIComponents._process_question(
                           st.session_state.current_query,
                           st.session_state.current_models,
                           use_context, # Use context setting from current UI state
                           search_limit, # Use setting from current UI state
                           enable_streaming, # Use setting from current UI state
                           config_manager,
                           answer_generator,
                           embedding_creator,
                           qdrant_manager
                       )

             UIComponents._display_query_results(use_context) 


    @staticmethod
    def _create_answer_mode_selection(config_manager) -> str:
        """ Create the search mode radio buttons with explanations. """
        modes = ["Search uploaded documents", "General knowledge search"]
        captions = [
            "Find information from your uploaded document collection. Requires processed files and necessary service keys.",
            "Search general knowledge base. Does not use your uploaded documents."
        ]

        answer_mode = st.radio(
            "Search Mode:",
            options=modes,
            captions=captions,
            index=0, # Default to document search
            horizontal=True,
        )

        # Add warnings based on mode and state
        if answer_mode == modes[0] and not config_manager.processed_files:
            st.caption(f"{ICON_WARNING} Warning: No documents found in the current collection for searching.")
        # (Other warnings about keys are handled elsewhere)

        return answer_mode

    @staticmethod
    def _create_model_selection() -> List[str]:
        """ Create the multi-select box for choosing search engines. """
        # Consider fetching this list dynamically if possible/needed
        available_models = [
            "google/gemini-2.5-flash",
            "google/gemini-2.5-pro",
            "anthropic/claude-sonnet-4",
            "x-ai/grok-3",
            "openai/gpt-4.1",
            "qwen/qwen3-235b-a22b",
            "openai/o4-mini-high",
        ]
        # Filter models based on availability or tier if needed

        # Sensible default model
        default_model = "openai/gpt-4o-mini"
        if default_model not in available_models:
             default_model = available_models[0] if available_models else None

        selected_models = st.multiselect(
            "Select search engine(s):",
            available_models,
            default=[default_model] if default_model else [],
            help="Choose one or more search engines to find information. Results will appear in tabs."
        )

        if not selected_models:
            st.warning("Please select at least one search engine.", icon=ICON_WARNING)
        # Limit selection?
        # max_models = 5
        # if len(selected_models) > max_models:
        #     st.warning(f"Selecting more than {max_models} models may slow down response time significantly.", icon=ICON_WARNING)

        return selected_models


    @staticmethod
    def _process_question(query, selected_models, use_context, search_limit, enable_streaming,
                        config_manager, answer_generator, embedding_creator, qdrant_manager) -> None:
        """
        Process a question: fetch context (if needed) and generate answers using threads,
        updating st.session_state. This function should ideally be called only once per query,
        even if Streamlit reruns.
        """

        # Check if results for this query/model combo are already being processed or done
        # This check helps prevent redundant thread creation on UI reruns
        is_processing_or_done = all(
            st.session_state.query_results.get(model, {}).get("status") in ["running", "completed", "error"]
            for model in selected_models
        )
        if is_processing_or_done:
            logger.debug("Query processing already started or completed for these models.")
            return # Avoid starting new threads if already handled

        logger.info(f"Starting processing for query: '{query[:50]}...' using models: {selected_models}")

        # Update status to 'running' in session state
        for model in selected_models:
            if st.session_state.query_results.get(model, {}).get("status") == "pending":
                st.session_state.query_results[model]["status"] = "running"
                st.session_state.query_results[model]["message"] = "Fetching context / Sending request..."

        # --- Get Context (RAG only) ---
        similar_chunks = None
        ordered_sources = None
        if use_context:
            with st.spinner(f"{ICON_PROCESS} Retrieving relevant document context..."):
                try:
                    similar_chunks = UIComponents._get_similar_chunks(
                        query,
                        config_manager,
                        embedding_creator,
                        qdrant_manager,
                        search_limit
                    )
                    if similar_chunks:
                        # Prepare context string and keep ordered sources
                        context_str, ordered_sources = answer_generator._prepare_context(similar_chunks) # Use generator's method
                        logger.info(f"Retrieved {len(similar_chunks)} context chunks.")
                    else:
                        logger.warning("No relevant context chunks found for the query.")
                        # Update status for all models if no context found
                        for model in selected_models:
                            st.session_state.query_results[model] = {
                                "status": "error",
                                "answer": "No relevant information found in the documents to answer this question.",
                                "sources": [],
                                "time_taken": 0
                            }
                        st.session_state.query_in_progress = False # Stop processing
                        st.rerun() # Update UI immediately
                        return

                except Exception as e:
                    logger.error(f"Error retrieving context: {str(e)}", exc_info=True)
                    st.error(f"{ICON_ERROR} Error retrieving document context: {str(e)}")
                    # Update status for all models on context error
                    for model in selected_models:
                         st.session_state.query_results[model] = {
                              "status": "error",
                              "answer": f"Failed to retrieve document context: {str(e)}",
                              "sources": [],
                              "time_taken": 0
                         }
                    st.session_state.query_in_progress = False # Stop processing
                    st.rerun() # Update UI immediately
                    return

        # --- Generate Answers ---
        try:
            answer_generator.set_api_key(config_manager.api_keys["OPENROUTER_API_KEY"])

            # Handle single model streaming case separately
            if enable_streaming and len(selected_models) == 1:
                model = selected_models[0]
                st.session_state.query_results[model]["message"] = "Streaming response..."
                st.session_state.query_results[model]["answer"] = ""
                st.session_state.query_results[model]["streaming"] = True
                
                # Start streaming in real-time
                UIComponents._handle_streaming_response(
                    query, similar_chunks, model, use_context, ordered_sources,
                    answer_generator
                )
            else:
                # Handle multiple models or non-streaming single model
                UIComponents._generate_answers_concurrently(
                    query, similar_chunks, selected_models, answer_generator, use_context, ordered_sources
                )

        except Exception as e:
            logger.error(f"Error during answer generation dispatch: {str(e)}", exc_info=True)
            st.error(f"{ICON_ERROR} An unexpected error occurred while preparing to get answers: {str(e)}")
            # Mark all pending/running models as error
            for model in selected_models:
                if st.session_state.query_results.get(model, {}).get("status") in ["pending", "running"]:
                     st.session_state.query_results[model] = {
                         "status": "error",
                         "answer": f"Failed before generation: {str(e)}",
                         "sources": [],
                         "time_taken": 0
                     }
            st.session_state.query_in_progress = False
            st.rerun()

    @staticmethod
    def _handle_streaming_response(query, similar_chunks, model, use_context, ordered_sources, answer_generator):
        """Handle real-time streaming response for a single model."""
        start_time = time.time()
        
        try:
            # Initialize streaming result
            st.session_state.query_results[model] = {
                "status": "streaming",
                "answer": "",
                "sources": ordered_sources if use_context else [],
                "time_taken": 0,
                "streaming": True
            }
            
            # Get the streaming generator
            stream_generator = answer_generator.generate_answer_streaming(
                query, similar_chunks, model=model, use_context=use_context
            )
            
            full_answer = ""
            chunk_count = 0
            last_update_time = time.time()
            update_interval = 0.5  # Update UI every 0.5 seconds max
            
            for chunk in stream_generator:
                chunk_count += 1
                full_answer += chunk
                current_time = time.time()
                
                # Update session state
                st.session_state.query_results[model]["answer"] = full_answer
                st.session_state.query_results[model]["time_taken"] = current_time - start_time
                
                # Update UI periodically based on time elapsed since last update
                if current_time - last_update_time >= update_interval:
                    st.rerun()
                    last_update_time = current_time
            
            # Final update when streaming is complete
            end_time = time.time()
            st.session_state.query_results[model] = {
                "status": "completed",
                "answer": full_answer,
                "sources": ordered_sources if use_context else [],
                "time_taken": end_time - start_time,
                "streaming": False
            }
            st.session_state.query_in_progress = False
            st.rerun()
            
            logger.info(f"Streaming completed for {model} in {end_time - start_time:.2f}s with {chunk_count} chunks")
            
        except Exception as e:
            error_msg = f"Error during streaming: {str(e)}"
            logger.error(f"Streaming error for {model}: {error_msg}", exc_info=True)
            
            st.session_state.query_results[model] = {
                "status": "error",
                "answer": error_msg,
                "sources": [],
                "time_taken": time.time() - start_time,
                "streaming": False
            }
            st.session_state.query_in_progress = False
            st.rerun()

    @staticmethod
    def _generate_answers_concurrently(query, similar_chunks, selected_models,
                                     answer_generator, use_context, ordered_sources):
        """Runs answer generation in threads and updates session state."""

        def _update_state_callback(future):
            """Callback function to update session state when a future completes."""
            try:
                result = future.result() # Get result from the future
                model = result['model']
                # Update the session state for this model
                st.session_state.query_results[model] = {
                    "status": result['status'].lower(), # completed or error
                    "answer": result['answer'],
                    # Use the pre-fetched ordered_sources if RAG, empty list otherwise
                    "sources": ordered_sources if use_context else [],
                    "time_taken": result.get('time_taken', 0)
                }
                logger.info(f"Received result for model {model}: Status {result['status']}")

                # Check if all models are done after this update
                all_done = all(res.get("status") not in ["pending", "running"]
                               for res in st.session_state.query_results.values())
                if all_done:
                    st.session_state.query_in_progress = False
                    logger.info("All concurrent tasks completed.")
                
                # Trigger a rerun to update the UI **after** state is updated
                # Be cautious with frequent reruns inside callbacks
                # st.rerun() # Maybe not needed if rendering loop handles it

            except Exception as e:
                # Find which model this future was for (tricky without passing model name)
                # This approach relies on the future object potentially having context,
                # or we need to map futures back to models.
                # For now, log the error generically.
                logger.error(f"Error in future callback: {e}", exc_info=True)
                # How to mark the correct model as error? Requires mapping futures.
                # A simpler way is to let the main loop check futures, not use callbacks.


        # --- Using ThreadPoolExecutor WITHOUT callbacks, process in main thread ---
        max_workers = min(len(selected_models), 5) # Limit concurrent requests
        futures_map = {} # Map future to model name

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            logger.info(f"Submitting {len(selected_models)} tasks to ThreadPoolExecutor...")
            for model in selected_models:
                 if st.session_state.query_results.get(model, {}).get("status") == "running":
                      st.session_state.query_results[model]["message"] = f"Sending request to {model}..."
                      future = executor.submit(
                           answer_generator.generate_answer_wrapper, # Use the new wrapper
                           query,
                           similar_chunks,
                           model=model,
                           use_context=use_context
                      )
                      futures_map[future] = model
                      # future.add_done_callback(_update_state_callback) # Using callback approach

            # --- Alternative: Process results in the main thread using as_completed ---
            # This is generally safer with Streamlit's execution model than callbacks
            results_processed = 0
            for future in concurrent.futures.as_completed(futures_map):
                 model = futures_map[future]
                 try:
                      result = future.result()
                      st.session_state.query_results[model] = {
                           "status": result['status'].lower(),
                           "answer": result['answer'],
                           "sources": ordered_sources if use_context else [], # Attach context sources here
                           "time_taken": result.get('time_taken', 0)
                      }
                      logger.info(f"Processed result for {model}. Status: {result['status']}")
                      results_processed += 1

                 except Exception as exc:
                      logger.error(f"Model {model} generated an exception: {exc}", exc_info=True)
                      st.session_state.query_results[model] = {
                           "status": "error",
                           "answer": f"Error during generation: {exc}",
                           "sources": [],
                           "time_taken": 0
                      }
                      results_processed += 1

                 # Optional: Trigger rerun after each result for faster UI update,
                 # but might cause flickering or performance issues.
                 # st.rerun()


        all_submitted_models_done = all(
             st.session_state.query_results.get(m, {}).get("status") in ["completed", "error"]
             for m in selected_models
         )

        if all_submitted_models_done:
             st.session_state.query_in_progress = False
             logger.info("All concurrent tasks completed (checked via as_completed loop).")
             # st.rerun() # Potentially needed if updates didn't trigger redraw properly
        else:
             logger.warning("as_completed finished, but not all models have final status. Might be ongoing.")


    @staticmethod
    def _display_query_results(use_context):
        """ Renders the results stored in st.session_state.query_results """
        results = st.session_state.get("query_results", {})
        if not results:
             return # Nothing to display yet

        model_names = list(results.keys())
        if not model_names:
            return

        tabs = st.tabs([f"{ICON_ENGINE} {name}" for name in model_names])
        displayed_sources = False # Flag to display sources only once

        for i, model in enumerate(model_names):
            with tabs[i]:
                result_data = results[model]
                status = result_data.get("status", "unknown")
                message = result_data.get("message", "")
                answer = result_data.get("answer", "")
                sources = result_data.get("sources", [])
                time_taken = result_data.get("time_taken")
                is_streaming = result_data.get("streaming", False)

                if status == "pending" or status == "running":
                     st.info(f"{ICON_PROCESS} {message or status.capitalize()}...", icon="⏳")
                elif status == "streaming":
                     # Show streaming indicator and partial content
                     if time_taken is not None:
                          st.info(f"{ICON_REALTIME} Streaming response... ({time_taken:.1f}s)", icon="⚡")
                     else:
                          st.info(f"{ICON_REALTIME} Streaming response...", icon="⚡")
                     
                     # Display partial answer if available
                     if answer:
                          st.markdown(answer)
                     
                     # Show sources if available and first tab
                     if use_context and sources and not displayed_sources:
                          UIComponents._display_source_documents(sources)
                          displayed_sources = True
                          
                elif status == "completed":
                     if time_taken is not None:
                          st.caption(f"Search completed in {time_taken:.2f} seconds.")
                     st.markdown(answer) # Display the search results
                     if use_context and sources and not displayed_sources:
                          UIComponents._display_source_documents(sources)
                          displayed_sources = True # Show sources only once below the first completed tab
                elif status == "error":
                     st.error(f"{ICON_ERROR} Error: {answer or 'An unknown error occurred.'}", icon="❌")


    @staticmethod
    def _get_similar_chunks(query, config_manager, embedding_creator, qdrant_manager, search_limit):
        """ Retrieve similar chunks from Qdrant for RAG. """
        if not config_manager.api_keys.get("OPENAI_API_KEY"):
            raise ValueError("OpenAI API Key needed for query embedding.")
        if not config_manager.collection_name:
             raise ValueError("No active collection selected.")

        embedding_creator.set_api_key(config_manager.api_keys["OPENAI_API_KEY"])

        logger.info(f"Creating embedding for query: '{query[:50]}...'")
        query_embedding = embedding_creator.create_query_embedding(query)
        if query_embedding is None:
             raise ValueError("Failed to create query embedding.")
        logger.info(f"Embedding created. Searching in '{config_manager.collection_name}' with limit {search_limit}.")

        similar_chunks = qdrant_manager.search_similar_chunks(
            collection_name=config_manager.collection_name,
            query_embedding=query_embedding,
            limit=search_limit
        )

        if similar_chunks is None:
             logger.warning(f"Search returned None, possibly connection error or empty collection.")
             return []

        logger.info(f"Found {len(similar_chunks)} similar chunks.")
        return similar_chunks


    @staticmethod
    def _display_source_documents(sources):
        """ Display source documents cleanly in an expander without nesting. """
        st.divider()
        with st.expander(f"{ICON_SOURCE} View Source Document Excerpts", expanded=False): # Outer expander is allowed
            if not sources:
                st.caption("No source documents were retrieved or provided.")
                return

            doc_sources = {}
            for source in sources:
                 doc = source.get('document', 'Unknown Document')
                 score = source.get('score', 0.0)
                 text = source.get('text', 'Source text not available.')

                 if doc not in doc_sources:
                      doc_sources[doc] = []
                 doc_sources[doc].append({'text': text, 'score': score})

            for doc in sorted(doc_sources.keys()):
                st.markdown(f"**📄 Document: `{doc}`**")
                sorted_chunks = sorted(doc_sources[doc], key=lambda x: x['score'], reverse=True)

                for i, source_chunk in enumerate(sorted_chunks):
                     st.markdown(f"**Excerpt {i+1} (Relevance: {source_chunk['score']:.3f})**")
                     st.markdown(f"> {source_chunk['text']}")
                     if i < len(sorted_chunks) - 1:
                          st.markdown("---") 

                st.divider()

    @staticmethod
    def create_main_page_layout(config_manager,
                               text_extractor,
                               text_chunker,
                               embedding_creator,
                               qdrant_manager,
                               answer_generator) -> None:
        """ Create the main page layout with tabs and initialize state. """
        st.set_page_config(
            page_title="Document Search Engine",
            page_icon=ICON_SOURCE, # Use a relevant icon
            layout="wide"
        )

        # Initialize session state MUST be called early
        UIComponents.initialize_session_state()

        # Create sidebar (populates config based on state)
        UIComponents.create_sidebar(config_manager)

        # Main content area
        st.title(f"{ICON_SOURCE} Document Search Engine")
        st.markdown("""
        Search and find information in your documents using advanced indexing methods. Upload PDFs, then search to get relevant information from your document collections.
        Configure search parameters and manage document collections in the sidebar.
        """)

        # Create tabs
        tab1, tab2 = st.tabs([f"{ICON_UPLOAD} Process Documents", f"{ICON_SEARCH} Search Documents"])

        with tab1:
            UIComponents.create_upload_tab(
                config_manager,
                text_extractor,
                text_chunker,
                embedding_creator,
                qdrant_manager
            )

        with tab2:
            UIComponents.create_question_tab(
                config_manager,
                embedding_creator,
                qdrant_manager,
                answer_generator
            )

        # Footer
        st.divider()
        st.caption("Document Search Engine")