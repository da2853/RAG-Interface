"""
Answer generator module for ExamGPT application.
Handles generating answers to user queries using various LLM providers via OpenRouter.
Includes fixes for concurrent execution stability.
"""
import logging
import time
import json
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple, Iterator, Union
import requests
import streamlit as st # Used only for optional status updates, keep usage minimal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class AnswerGenerator:
    """
    Generates answers using OpenRouter, supports context, streaming, and concurrent requests.
    """

    def __init__(self,
                 openrouter_api_key: Optional[str] = None,
                 max_retries: int = 2, # Reduced default retries
                 timeout: float = 120.0, # Increased default timeout
                 max_workers: int = 5,
                 system_prompt: Optional[str] = None,
                 show_status: bool = False): # Default show_status to False to avoid UI clutter
        """ Initialize the answer generator. """
        self.openrouter_api_key = openrouter_api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_workers = max_workers # Used by UI component, not directly here now
        self.show_status = show_status # Controls optional st.error calls

        self.system_prompt_rag = system_prompt or (
            "You are an AI assistant specialized in analyzing provided documents. "
            "Answer the user's question based *only* on the given context excerpts. "
            "If the answer is not found in the context, state that clearly. "
            "Cite the source document for information used (e.g., '[Document: report.pdf]'). "
            "Be concise and accurate."
        )
        self.system_prompt_direct = ( # Separate prompt for non-RAG
             "You are a helpful AI assistant. Answer the user's question based on your general knowledge. "
             "Be informative and clear in your responses."
        )
        self.current_system_prompt = self.system_prompt_rag # Default

    def set_api_key(self, api_key: str) -> None:
        """ Set or update the OpenRouter API key. """
        self.openrouter_api_key = api_key
        logger.debug("OpenRouter API key updated.")

    def set_system_prompt(self, prompt: str) -> None:
        """
        Set the system prompt *used for RAG*. The direct prompt remains fixed.
        If prompt is empty, resets RAG prompt to default.
        """
        if not prompt:
            logger.warning("Empty system prompt provided, resetting RAG prompt to default.")
            self.system_prompt_rag = self.__init__.__defaults__[3] # Get default from init signature
        else:
            self.system_prompt_rag = prompt
        logger.info("RAG System prompt updated.")
        # Keep current_system_prompt updated based on expected next call type? Risky.
        # Better to select prompt within generate methods based on use_context flag.

    def _prepare_context(self, context_chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Prepare context string from chunks and return ordered chunks (for sourcing).
        Handles potential missing keys gracefully.
        """
        if not context_chunks:
            return "", []

        # Add safety checks for chunk structure
        valid_chunks = []
        for i, chunk in enumerate(context_chunks):
            if isinstance(chunk, dict) and 'document' in chunk and 'text' in chunk:
                 # Assign index if missing, helps ordering robustness
                 chunk['chunk_index'] = chunk.get('chunk_index', i)
                 valid_chunks.append(chunk)
            else:
                 logger.warning(f"Skipping invalid chunk format: {chunk}")

        if not valid_chunks:
             return "", []


        # Order chunks by document name, then by original index/position
        doc_chunks = {}
        for chunk in valid_chunks:
            doc = chunk['document']
            if doc not in doc_chunks:
                doc_chunks[doc] = []
            doc_chunks[doc].append(chunk)

        ordered_chunks = []
        for doc in sorted(doc_chunks.keys()): # Sort by document name
            # Sort chunks within the document by index
            sorted_doc_chunks = sorted(doc_chunks[doc], key=lambda x: x['chunk_index'])
            ordered_chunks.extend(sorted_doc_chunks)


        context_parts = []
        for chunk in ordered_chunks:
            doc_name = chunk['document']
            # Ensure text is string
            text = str(chunk['text']) if chunk.get('text') is not None else "[Content missing]"
            context_parts.append(f"[Document: {doc_name}]\n{text}\n---") # Use separator

        context_str = "\n".join(context_parts)
        return context_str, ordered_chunks


    def _make_api_request(self, payload: Dict[str, Any], stream: bool = False) -> Union[requests.Response, None]:
        """ Makes the API request to OpenRouter with retry logic. """
        if not self.openrouter_api_key:
            logger.error("OpenRouter API key not set.")
            if self.show_status: st.error("OpenRouter API key not set.")
            return None

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        if stream:
            headers["Accept"] = "text/event-stream"

        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    stream=stream # Pass stream argument to requests
                )
                # Check for common non-200 errors that indicate immediate failure
                if response.status_code in [401, 403]: # Unauthorized / Forbidden
                     logger.error(f"OpenRouter Auth Error ({response.status_code}): {response.text}")
                     if self.show_status: st.error(f"OpenRouter Auth Error ({response.status_code}). Check API Key.")
                     return None # Don't retry auth errors
                if response.status_code == 429: # Rate limit
                     logger.warning(f"Rate limit hit ({response.status_code}). Retrying after delay...")
                     # Implement backoff based on headers if available, otherwise fixed delay
                     time.sleep(5 * (retry_count + 1)) # Simple exponential backoff
                     retry_count += 1
                     continue # Retry the request
                # Allow other non-200 potentially retryable errors or success (200)
                return response

            except requests.exceptions.Timeout:
                logger.warning(f"Request timed out (attempt {retry_count + 1}/{self.max_retries + 1}).")
                if retry_count == self.max_retries: return None # Failed after retries
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {retry_count + 1}/{self.max_retries + 1}): {e}")
                if retry_count == self.max_retries: return None # Failed after retries
            except Exception as e: # Catch unexpected errors during request setup/execution
                 logger.error(f"Unexpected error during API request: {e}", exc_info=True)
                 return None # Fail immediately on unexpected errors

            # Wait before retrying non-timeout errors
            time.sleep(2 ** retry_count)
            retry_count += 1

        logger.error(f"API request failed after {self.max_retries} retries.")
        return None


    def generate_answer(self,
                    query: str,
                    context_chunks: Optional[List[Dict[str, Any]]] = None,
                    model: str = "openai/gpt-4o-mini", # Updated default
                    use_context: bool = True) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """ Generate a complete answer (non-streaming). """

        # Prepare context and select system prompt
        ordered_chunks = []
        user_message = query
        system_prompt = self.system_prompt_direct

        if use_context:
            system_prompt = self.system_prompt_rag
            if context_chunks:
                 context_str, ordered_chunks = self._prepare_context(context_chunks)
                 if context_str: # Only add context if it was prepared successfully
                      user_message = f"CONTEXT:\n---\n{context_str}\n---\n\nQUESTION: {query}"
                 else:
                      logger.warning("use_context=True but failed to prepare context string.")
                      # Fallback: proceed without context string but use RAG prompt
                      user_message = f"CONTEXT: [No relevant context found in documents]\n\nQUESTION: {query}"

            else: # use_context is True, but no chunks provided
                 logger.warning("use_context=True but context_chunks is None or empty.")
                 user_message = f"CONTEXT: [No documents provided for context]\n\nQUESTION: {query}"


        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "stream": False
        }

        logger.info(f"Generating non-streaming answer with model: {model}")
        response = self._make_api_request(payload, stream=False)

        if response is None:
             return "Error: API request failed after retries.", ordered_chunks # Return error message
        if response.status_code == 200:
            try:
                result = response.json()
                answer = result['choices'][0]['message']['content']
                logger.info(f"Successfully generated answer using {model}.")
                return answer, ordered_chunks
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                 logger.error(f"Error parsing successful response from {model}: {e} - Response: {response.text}")
                 return f"Error: Could not parse response from {model}.", ordered_chunks
        else:
            error_msg = f"Error: API returned status {response.status_code} - {response.text}"
            logger.error(error_msg)
            if self.show_status: st.error(error_msg)
            return error_msg, ordered_chunks


    def generate_answer_streaming(self,
                                query: str,
                                context_chunks: Optional[List[Dict[str, Any]]] = None,
                                model: str = "openai/gpt-4o-mini",
                                use_context: bool = True) -> Iterator[str]:
        """ Generate an answer with streaming. """

         # Prepare context and select system prompt (same logic as non-streaming)
        ordered_chunks = [] # Not returned by generator, but context is prepared
        user_message = query
        system_prompt = self.system_prompt_direct

        if use_context:
            system_prompt = self.system_prompt_rag
            if context_chunks:
                 context_str, ordered_chunks = self._prepare_context(context_chunks)
                 if context_str:
                      user_message = f"CONTEXT:\n---\n{context_str}\n---\n\nQUESTION: {query}"
                 else:
                      user_message = f"CONTEXT: [No relevant context found in documents]\n\nQUESTION: {query}"
            else:
                 user_message = f"CONTEXT: [No documents provided for context]\n\nQUESTION: {query}"


        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "stream": True
        }

        logger.info(f"Generating streaming answer with model: {model}")
        response = self._make_api_request(payload, stream=True)

        if response is None:
            yield "Error: API request failed after retries."
            return
        if response.status_code != 200:
            yield f"Error: API returned status {response.status_code} - {response.text}"
            return

        # Process the streaming response
        try:
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith('data: '):
                    data_str = line[len('data: '):].strip()
                    if data_str == '[DONE]':
                        logger.info(f"Streaming finished for {model}.")
                        break # Stream finished successfully
                    if not data_str:
                         continue # Skip empty data lines

                    try:
                        data_obj = json.loads(data_str)
                        # Check structure for content delta
                        if isinstance(data_obj, dict) and 'choices' in data_obj:
                             if data_obj['choices'] and isinstance(data_obj['choices'][0], dict):
                                 delta = data_obj['choices'][0].get('delta', {})
                                 content = delta.get('content')
                                 if content: # Yield only if content exists
                                     yield content

                    except json.JSONDecodeError:
                        logger.warning(f"Could not decode JSON from stream line: {data_str}")
                        continue # Skip malformed lines
        except Exception as e:
            error_msg = f"Error processing stream: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield f"\n\n[Stream interrupted: {error_msg}]" # Yield error within stream
        finally:
            # Ensure response is closed (important for streaming)
            if response:
                 response.close()


    def generate_answer_wrapper(self, query, context_chunks, model, use_context) -> Dict[str, Any]:
        """
        Wrapper for generate_answer used by concurrent executor.
        Returns a dictionary including status and timing.
        """
        start_time = time.time()
        try:
            answer, sources = self.generate_answer(
                query,
                context_chunks,
                model=model,
                use_context=use_context
            )
            time_taken = time.time() - start_time

            # Check if answer indicates an error occurred within generate_answer
            if answer is None or answer.startswith("Error:"):
                status = "Error"
                error_message = answer if answer else "Unknown error during generation."
                logger.error(f"Error generating answer for {model}: {error_message}")
                return {
                    "model": model, "answer": error_message, "status": status,
                    "sources": sources, "time_taken": time_taken
                }
            else:
                status = "Completed"
                logger.info(f"Successfully generated answer for {model} in {time_taken:.2f}s")
                return {
                    "model": model, "answer": answer, "status": status,
                    "sources": sources, "time_taken": time_taken
                }

        except Exception as e:
            time_taken = time.time() - start_time
            error_msg = f"Exception in generate_answer_wrapper for {model}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "model": model, "answer": error_msg, "status": "Error",
                "sources": [], "time_taken": time_taken
            }

    # generate_answers_concurrently is removed as the logic is now handled
    # by the UI component (_generate_answers_concurrently) using ThreadPoolExecutor
    # and the generate_answer_wrapper method.