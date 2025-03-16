"""
Configuration manager for ExamGPT application.
Handles loading, saving, and updating configuration settings.
"""
import json
import os
from pathlib import Path
import uuid
import logging
from typing import Dict, List, Optional, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages application configuration including API keys, collection names,
    and processed files list with persistence to disk.
    """

    def __init__(self, config_path: Union[str, Path] = "rag_assistant_config.json"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path)
        
        # Default API keys (empty)
        self.api_keys = {
            "OPENAI_API_KEY": "",
            "OPENROUTER_API_KEY": "",
            "QDRANT_URL": "",
            "QDRANT_API_KEY": ""
        }
        
        # Default processed files list
        self.processed_files = []
        
        # Default collection name (will be generated if not loaded)
        self._collection_name = None
        
        # Default prompt templates
        self.prompt_templates = {
            "Academic Assistant": (
                "You are an AI assistant helping with academic questions. "
                "Answer based on the provided context and your understanding. "
                "If the information is not in the context, say 'Based on the provided documents, "
                "I don't have enough information to answer this question.' "
                "Be precise and clear. When quoting from the documents, indicate which document "
                "the information comes from."
            ),
            "Legal Assistant": (
                "You are an AI assistant helping with legal questions. "
                "Answer based on the provided legal documents and your understanding of legal principles. "
                "If the information is not in the provided documents, say 'Based on the provided legal documents, "
                "I don't have enough information to answer this question. Please consult a qualified legal professional.' "
                "Be precise and reference specific sections of the documents when possible."
            ),
            "Technical Documentation Helper": (
                "You are an AI assistant helping with technical questions. "
                "Answer based on the provided technical documentation and your understanding. "
                "If the information is not in the context, say 'Based on the provided technical documents, "
                "I don't have enough information to answer this question.' "
                "Be precise and reference specific parts of the documentation when possible."
            ),
            "Research Assistant": (
                "You are an AI research assistant. "
                "Answer based on the provided research papers and your understanding of the scientific literature. "
                "If the information is not in the context, say 'Based on the provided research papers, "
                "I don't have enough information to answer this question.' "
                "Be precise and cite specific papers and sections when possible."
            ),
            "Custom": (
                "You are a helpful AI assistant. "
                "Answer based on the provided documents and your understanding. "
                "If the information is not in the context, acknowledge the limits of the provided information. "
                "Be clear and helpful in your responses."
            )
        }
        
        # Default selected prompt template
        self.selected_template = "Academic Assistant"
        
        # Load existing configuration or create default
        self._load_config()
    
    @property
    def collection_name(self) -> str:
        """
        Get the current collection name, generating a new one if needed.
        
        Returns:
            str: The collection name
        """
        if not self._collection_name:
            self._collection_name = f"exam_docs_{uuid.uuid4().hex[:8]}"
            self.save_config()
            
        return self._collection_name
    
    @collection_name.setter
    def collection_name(self, name: str):
        """
        Set the collection name.
        
        Args:
            name: The new collection name
        """
        self._collection_name = name
    
    def _load_config(self) -> None:
        """
        Load configuration from disk, with error handling.
        Falls back to defaults if loading fails.
        """
        try:
            if not self.config_path.exists():
                logger.info(f"Config file {self.config_path} not found. Using defaults.")
                return
                
            with open(self.config_path, "r") as f:
                config = json.load(f)
                
                # Load collection name
                if 'collection_name' in config:
                    self._collection_name = config['collection_name']
                
                # Load processed files
                if 'processed_files' in config and isinstance(config['processed_files'], list):
                    self.processed_files = config['processed_files']
                
                # Load API keys
                if 'api_keys' in config and isinstance(config['api_keys'], dict):
                    # Only update keys that are in our template
                    for key in self.api_keys:
                        if key in config['api_keys']:
                            self.api_keys[key] = config['api_keys'][key]
                
                # Load prompt templates
                if 'prompt_templates' in config and isinstance(config['prompt_templates'], dict):
                    # Merge with default templates (keeping custom ones)
                    for key, value in config['prompt_templates'].items():
                        self.prompt_templates[key] = value
                
                # Load selected template
                if 'selected_template' in config and config['selected_template'] in self.prompt_templates:
                    self.selected_template = config['selected_template']
                
                logger.info("Configuration loaded successfully")
                
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding config file: {str(e)}")
        except PermissionError as e:
            logger.error(f"Permission error accessing config file: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {str(e)}")
    
    def save_config(self) -> bool:
        """
        Save the current configuration to disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure the parent directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, "w") as f:
                json.dump({
                    "collection_name": self._collection_name,
                    "processed_files": self.processed_files,
                    "api_keys": self.api_keys,
                    "prompt_templates": self.prompt_templates,
                    "selected_template": self.selected_template
                }, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except PermissionError as e:
            logger.error(f"Permission error saving config file: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False
    
    def update_api_key(self, key_name: str, value: str) -> None:
        """
        Update a specific API key.
        
        Args:
            key_name: The name of the API key to update
            value: The new value for the API key
        
        Raises:
            KeyError: If the key name is not valid
        """
        if key_name not in self.api_keys:
            raise KeyError(f"Invalid API key name: {key_name}")
        
        self.api_keys[key_name] = value
    
    def get_api_key(self, key_name: str) -> str:
        """
        Get a specific API key.
        
        Args:
            key_name: The name of the API key to get
            
        Returns:
            str: The API key value
            
        Raises:
            KeyError: If the key name is not valid
        """
        if key_name not in self.api_keys:
            raise KeyError(f"Invalid API key name: {key_name}")
        
        return self.api_keys[key_name]
    
    def add_processed_file(self, filename: str) -> None:
        """
        Add a file to the processed files list if not already present.
        
        Args:
            filename: The name of the processed file
        """
        if filename not in self.processed_files:
            self.processed_files.append(filename)
    
    def remove_processed_file(self, filename: str) -> bool:
        """
        Remove a file from the processed files list.
        
        Args:
            filename: The name of the file to remove
            
        Returns:
            bool: True if the file was removed, False if it wasn't in the list
        """
        if filename in self.processed_files:
            self.processed_files.remove(filename)
            return True
        return False
    
    def check_duplicate_files(self, filenames: List[str]) -> tuple[List[str], List[str]]:
        """
        Check for duplicate files compared to processed files.
        
        Args:
            filenames: List of filenames to check
            
        Returns:
            tuple: (new_files, duplicates)
        """
        duplicates = []
        new_files = []
        
        for filename in filenames:
            if filename in self.processed_files:
                duplicates.append(filename)
            else:
                new_files.append(filename)
        
        return new_files, duplicates
    
    def reset_collection(self) -> None:
        """
        Create a new collection and reset the processed files list.
        """
        self._collection_name = f"rag_docs_{uuid.uuid4().hex[:8]}"
        self.processed_files = []
        self.save_config()
        logger.info(f"Collection reset to {self._collection_name}")
        
    def get_prompt_template(self, template_name: Optional[str] = None) -> str:
        """
        Get a prompt template by name.
        
        Args:
            template_name: Name of the template (uses selected template if None)
            
        Returns:
            str: The prompt template text
        """
        name = template_name or self.selected_template
        return self.prompt_templates.get(name, self.prompt_templates["Custom"])
    
    def add_prompt_template(self, name: str, template: str) -> None:
        """
        Add or update a prompt template.
        
        Args:
            name: Name of the template
            template: The prompt template text
        """
        self.prompt_templates[name] = template
        logger.info(f"Added prompt template: {name}")
    
    def delete_prompt_template(self, name: str) -> bool:
        """
        Delete a prompt template.
        
        Args:
            name: Name of the template to delete
            
        Returns:
            bool: True if the template was deleted, False if it didn't exist or is a default template
        """
        # Don't allow deleting the default templates
        default_templates = ["Academic Assistant", "Legal Assistant", "Technical Documentation Helper", 
                            "Research Assistant", "Custom"]
        
        if name in default_templates:
            logger.warning(f"Cannot delete default template: {name}")
            return False
            
        if name in self.prompt_templates:
            del self.prompt_templates[name]
            
            # If the deleted template was selected, fall back to "Custom"
            if self.selected_template == name:
                self.selected_template = "Custom"
                
            logger.info(f"Deleted prompt template: {name}")
            return True
            
        return False
    
    def set_selected_template(self, template_name: str) -> bool:
        """
        Set the selected prompt template.
        
        Args:
            template_name: Name of the template to select
            
        Returns:
            bool: True if successful, False if template doesn't exist
        """
        if template_name in self.prompt_templates:
            self.selected_template = template_name
            logger.info(f"Selected prompt template: {template_name}")
            return True
        
        logger.warning(f"Cannot select non-existent template: {template_name}")
        return False