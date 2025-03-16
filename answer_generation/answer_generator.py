"""
Answer generator module for ExamGPT application.
Handles generating answers to user queries using various LLM providers.
"""
import logging
import time
import json
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple, Iterator, Union
import requests
import streamlit as st  # For status updates

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class AnswerGenerator:
    """
    Generates answers to user queries based on retrieved context chunks
    using various language model providers.
    """
    
    def __init__(self, 
                 openrouter_api_key: Optional[str] = None,
                 max_retries: int = 3,
                 timeout: float = 90.0,
                 max_workers: int = 5,
                 system_prompt: Optional[str] = None,
                 show_status: bool = True):
        """
        Initialize the answer generator.
        
        Args:
            openrouter_api_key: OpenRouter API key
            max_retries: Maximum retry attempts for API calls
            timeout: Timeout for API calls in seconds
            max_workers: Maximum concurrent workers for parallel processing
            system_prompt: Custom system prompt (uses default if None)
            show_status: Whether to show generation status in the Streamlit UI
        """
        self.openrouter_api_key = openrouter_api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_workers = max_workers
        self.show_status = show_status
        
        # Default system prompt
        self.system_prompt = system_prompt or (
            "You are an AI assistant helping answer questions. "
            "Answer based on the provided context and your understanding. "
            "If the information is not in the context, say 'Based on the provided documents, "
            "I don't have enough information to answer this question.' "
            "Be precise and clear. When quoting from the documents, indicate which document "
            "the information comes from."
        )
    
    def set_api_key(self, api_key: str) -> None:
        """
        Set the OpenRouter API key.
        
        Args:
            api_key: OpenRouter API key
        """
        self.openrouter_api_key = api_key
        
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set the system prompt.
        
        Args:
            prompt: System prompt text
        """
        if not prompt:
            logger.warning("Empty system prompt provided, using default")
            return
            
        self.system_prompt = prompt
        logger.info("System prompt updated")
    
    def _prepare_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Prepare context string from context chunks.
        
        Args:
            context_chunks: List of context chunks
            
        Returns:
            str: Formatted context string
        """
        # Order chunks by document and position
        ordered_chunks = []
        doc_chunks = {}
        
        for chunk in context_chunks:
            doc = chunk['document']
            if doc not in doc_chunks:
                doc_chunks[doc] = []
            doc_chunks[doc].append(chunk)
        
        # Sort chunks within each document by index
        for doc, chunks in doc_chunks.items():
            ordered_chunks.extend(sorted(chunks, key=lambda x: x.get('chunk_index', 0)))
        
        # Prepare context
        context_parts = []
        for chunk in ordered_chunks:
            doc_name = chunk['document']
            context_parts.append(f"[Document: {doc_name}]\n{chunk['text']}\n")
        
        return "\n".join(context_parts), ordered_chunks
    
    def generate_answer(self, 
                    query: str, 
                    context_chunks: Optional[List[Dict[str, Any]]] = None,
                    model: str = "qwen/qwq-32b",
                    use_context: bool = True) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate an answer using OpenRouter API without streaming.
        
        Args:
            query: User query
            context_chunks: List of context chunks (can be None if use_context is False)
            model: Model to use for generation
            use_context: Whether to include context in the prompt
            
        Returns:
            tuple: (answer, ordered_chunks)
            
        Raises:
            ValueError: If no API key is set
        """
        if not self.openrouter_api_key:
            error_msg = "OpenRouter API key not provided"
            logger.error(error_msg)
            if self.show_status:
                st.error(error_msg)
            return error_msg, []
        
        # Prepare context if needed
        ordered_chunks = []
        if use_context and context_chunks:
            context, ordered_chunks = self._prepare_context(context_chunks)
        
        # Prepare the request
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        
        # Select appropriate system prompt based on context mode
        if use_context:
            # Use the standard system prompt for document context
            system_prompt = self.system_prompt
            # Prepare user message with context
            if context_chunks:
                user_message = f"Here are relevant excerpts from documents:\n\n{context}\n\nBased on these materials, please answer this question: {query}"
            else:
                user_message = query
        else:
            # For direct queries, use a modified system prompt without document references
            system_prompt = (
                "You are an AI assistant helping answer questions. "
                "Answer based on your general knowledge and training. "
                "Be precise and clear in your responses."
            )
            # Use query directly without document context
            user_message = query
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "stream": False  # Ensure streaming is disabled
        }
        
        # Retry logic
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                logger.info(f"Sending request to OpenRouter using model: {model}")
                
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result['choices'][0]['message']['content']
                    logger.info(f"Successfully generated answer using {model}")
                    return answer, ordered_chunks
                else:
                    retry_count += 1
                    logger.warning(f"Error from OpenRouter: {response.status_code} - {response.text} (attempt {retry_count})")
                    
                    if retry_count >= self.max_retries:
                        error_msg = f"Error: {response.status_code} - {response.text}"
                        if self.show_status:
                            st.error(error_msg)
                        return error_msg, []
                    
                    # Exponential backoff
                    time.sleep(2 ** retry_count)
                    
            except requests.exceptions.Timeout:
                retry_count += 1
                logger.warning(f"Request to OpenRouter timed out (attempt {retry_count})")
                
                if retry_count >= self.max_retries:
                    error_msg = "Error: Request to OpenRouter timed out after multiple attempts"
                    if self.show_status:
                        st.error(error_msg)
                    return error_msg, []
                
                # Wait longer for timeout issues
                time.sleep(3 ** retry_count)
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"Error generating answer (attempt {retry_count}): {str(e)}")
                
                if retry_count >= self.max_retries:
                    error_msg = f"Error generating answer after {self.max_retries} attempts: {str(e)}"
                    if self.show_status:
                        st.error(error_msg)
                    return error_msg, []
                
                time.sleep(2 ** retry_count)  # Exponential backoff
        
        return "An unexpected error occurred", []
    
    def generate_answer_streaming(self, 
                                query: str, 
                                context_chunks: Optional[List[Dict[str, Any]]] = None, 
                                model: str = "qwen/qwq-32b",
                                use_context: bool = True) -> Iterator[str]:
        """
        Generate an answer using OpenRouter API with streaming support.
        
        Args:
            query: User query
            context_chunks: List of context chunks (can be None if use_context is False)
            model: Model to use for generation
            use_context: Whether to include context in the prompt
            
        Yields:
            str: Chunks of the generated answer
            
        Raises:
            ValueError: If no API key is set
        """
        if not self.openrouter_api_key:
            yield "Error: OpenRouter API key not provided"
            return
        
        # Select appropriate system prompt based on context mode
        if use_context:
            # Use the standard system prompt for document context
            system_prompt = self.system_prompt
            
            # Prepare context if needed
            if context_chunks:
                context, _ = self._prepare_context(context_chunks)
                # Prepare user message with context
                user_message = f"Here are relevant excerpts from documents:\n\n{context}\n\nBased on these materials, please answer this question: {query}"
            else:
                # Use query directly if no context chunks available
                user_message = query
        else:
            # For direct queries, use a modified system prompt without document references
            system_prompt = (
                "You are an AI assistant helping answer questions. "
                "Answer based on your general knowledge and training. "
                "Be precise and clear in your responses."
            )
            # Use query directly without document context
            user_message = query
        
        # Prepare the request
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"  # Explicitly request SSE format
        }
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "stream": True
        }
        
        # Make streaming request with retry logic
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                logger.info(f"Sending streaming request to OpenRouter using model: {model}")
                
                with requests.post(url, headers=headers, json=payload, stream=True, timeout=self.timeout) as response:
                    if response.status_code != 200:
                        retry_count += 1
                        logger.warning(f"Error from OpenRouter: {response.status_code} - {response.text} (attempt {retry_count})")
                        
                        if retry_count >= self.max_retries:
                            yield f"Error: {response.status_code} - {response.text}"
                            return
                        
                        time.sleep(2 ** retry_count)  # Exponential backoff
                        continue
                    
                    # Process streaming response line by line
                    for line in response.iter_lines(decode_unicode=True):
                        if not line or not line.strip():
                            continue
                        
                        if line.startswith('data: '):
                            data = line[6:].strip()
                            if data == '[DONE]':
                                logger.info(f"Streaming response completed for {model}")
                                return
                            
                            try:
                                data_obj = json.loads(data)
                                # Handle different response formats
                                if "choices" in data_obj and len(data_obj["choices"]) > 0:
                                    choice = data_obj["choices"][0]
                                    # Check delta (streaming) or message (non-streaming) format
                                    if "delta" in choice:
                                        content = choice["delta"].get("content", "")
                                    elif "message" in choice:
                                        content = choice["message"].get("content", "")
                                    else:
                                        content = ""
                                    
                                    if content:
                                        yield content
                            except json.JSONDecodeError as e:
                                # Log error but continue processing
                                logger.warning(f"Error parsing streaming data: {e}")
                                continue
                
                # If we get here, streaming completed successfully
                return
                
            except requests.exceptions.Timeout:
                retry_count += 1
                logger.warning(f"Request to OpenRouter timed out (attempt {retry_count})")
                
                if retry_count >= self.max_retries:
                    yield "Error: Request to OpenRouter timed out after multiple attempts"
                    return
                
                time.sleep(3 ** retry_count)  # Wait longer for timeout issues
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"Error in streaming response (attempt {retry_count}): {str(e)}")
                
                if retry_count >= self.max_retries:
                    yield f"Error generating answer after {self.max_retries} attempts: {str(e)}"
                    return
                
                time.sleep(2 ** retry_count)  # Exponential backoff
        
    def generate_answers_concurrently(self, 
                                     query: str, 
                                     context_chunks: Optional[List[Dict[str, Any]]] = None, 
                                     models: List[str] = [],
                                     use_context: bool = True) -> List[concurrent.futures.Future]:
        """
        Generate answers using multiple models concurrently.
        
        Args:
            query: User query
            context_chunks: List of context chunks (can be None if use_context is False)
            models: List of models to use
            use_context: Whether to include context in the prompt
            
        Returns:
            list: List of Future objects that will contain results
        """
        if not models:
            logger.warning("No models provided for concurrent generation")
            return []
        
        # Limit the number of models to process concurrently
        models_to_process = models[:self.max_workers]
        
        if len(models_to_process) < len(models):
            logger.warning(f"Limiting concurrent processing to {self.max_workers} models")
        
        # Define the worker function
        def generate_for_model(model):
            """Worker function that processes a single model request"""
            try:
                start_time = time.time()
                
                # Call API to generate answer (this happens in parallel)
                answer, sources = self.generate_answer(
                    query, 
                    context_chunks, 
                    model=model,
                    use_context=use_context
                )
                
                process_time = time.time() - start_time
                
                return {
                    "model": model,
                    "answer": answer,
                    "sources": sources,
                    "time_taken": process_time,
                    "status": "Completed"
                }
                    
            except Exception as e:
                error_msg = f"Error with {model}: {str(e)}"
                logger.error(error_msg)
                
                return {
                    "model": model,
                    "answer": error_msg,
                    "sources": [],
                    "status": "Error",
                    "error": str(e)
                }
        
        # Use ThreadPoolExecutor for concurrent execution
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(models), self.max_workers)) as executor:
            # Submit all tasks at once for true parallelism
            futures = [executor.submit(generate_for_model, model) for model in models]
        
        logger.info(f"Submitted {len(futures)} concurrent model requests")
        return futures