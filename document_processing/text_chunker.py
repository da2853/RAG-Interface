"""
Text chunking module for ExamGPT application.
Handles dividing large texts into smaller, semantically meaningful chunks.
"""
import logging
from typing import List, Optional
import streamlit as st  # For status updates

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class TextChunker:
    """
    Splits text into overlapping chunks with intelligent boundary detection
    for optimal semantic coherence.
    """
    
    def __init__(self, 
                 default_chunk_size: int = 1000, 
                 default_overlap: int = 200,
                 show_status: bool = True):
        """
        Initialize the text chunker.
        
        Args:
            default_chunk_size: Default size of each chunk in characters
            default_overlap: Default overlap between chunks in characters
            show_status: Whether to show chunking status in the Streamlit UI
        """
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap
        self.show_status = show_status
    
    def split_text_into_chunks(self, 
                              text: str, 
                              chunk_size: Optional[int] = None, 
                              overlap: Optional[int] = None) -> List[str]:
        """
        Split text into overlapping chunks with improved boundary detection.
        
        Args:
            text: The text to split into chunks
            chunk_size: Size of each chunk in characters (uses default if None)
            overlap: Overlap between chunks in characters (uses default if None)
            
        Returns:
            list: List of text chunks
        """
        if not text:
            logger.warning("Empty text provided for chunking")
            return []
        
        # Use default values if not provided
        chunk_size = chunk_size or self.default_chunk_size
        overlap = overlap or self.default_overlap
        
        # Validate parameters
        if chunk_size <= 0:
            logger.error(f"Invalid chunk size: {chunk_size}")
            raise ValueError("Chunk size must be positive")
        
        if overlap < 0 or overlap >= chunk_size:
            logger.error(f"Invalid overlap: {overlap}")
            raise ValueError("Overlap must be non-negative and less than chunk size")
        
        chunks = []
        start = 0
        text_length = len(text)
        
        # Reasonable max chunks for a document to prevent infinite loops
        max_chunks = max(1, text_length // 100)  # 1 chunk per 100 chars would be extreme
        chunk_count = 0
        
        # For debugging
        if self.show_status:
            st.write(f"Text length: {text_length} characters")
        
        logger.info(f"Chunking text of length {text_length} with chunk_size={chunk_size}, overlap={overlap}")
        
        while start < text_length and chunk_count < max_chunks:
            # Ensure minimum progress in each iteration
            min_end = start + max(50, chunk_size // 10)  # At least move forward by 10% of chunk_size
            end = min(start + chunk_size, text_length)
            
            # Try to find a good boundary
            if end < text_length:
                # Look for paragraph breaks first (double newline)
                paragraph_break = text.find('\n\n', max(end - 100, 0), min(end + 100, text_length))
                
                if paragraph_break != -1 and paragraph_break > min_end:
                    end = paragraph_break + 2
                    logger.debug(f"Found paragraph break at position {paragraph_break}")
                else:
                    # Look for sentence end (period followed by space or newline)
                    sentence_end = -1
                    for i in range(end, min(end + 100, text_length)):
                        if i < text_length - 1 and text[i] == '.' and (text[i+1] == ' ' or text[i+1] == '\n'):
                            sentence_end = i + 1
                            break
                    
                    if sentence_end != -1 and sentence_end > min_end:
                        end = sentence_end
                        logger.debug(f"Found sentence end at position {sentence_end}")
                    else:
                        # Fall back to space
                        space = text.find(' ', end)
                        if space != -1 and space < end + 50 and space > min_end:
                            end = space + 1
                            logger.debug(f"Found space at position {space}")
            
            # Create the chunk
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
                chunk_count += 1
                
                # Debug output for the first few chunks
                if chunk_count <= 3 and self.show_status:
                    st.write(f"Chunk {chunk_count}: {len(chunk)} chars, start={start}, end={end}")
            
            # Always advance start by at least 10% of chunk_size
            next_start = end - overlap
            if next_start <= start + (chunk_size // 10):
                next_start = start + (chunk_size // 10)
            
            start = next_start
        
        # Add a warning if we hit the max chunks limit
        if chunk_count >= max_chunks:
            warning_msg = f"Document chunking stopped at {chunk_count} chunks to prevent processing issues"
            logger.warning(warning_msg)
            if self.show_status:
                st.warning(warning_msg)
        
        if self.show_status:
            st.write(f"Created {len(chunks)} chunks from {text_length} characters")
        
        logger.info(f"Created {len(chunks)} chunks from text of length {text_length}")
        return chunks