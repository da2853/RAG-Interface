"""
Text extraction module for ExamGPT application.
Handles extracting text from PDF files.
"""
import os
import tempfile
import logging
from typing import Tuple, Optional, BinaryIO, Union
from pypdf import PdfReader
import streamlit as st  # For status updates

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class TextExtractor:
    """
    Extracts text from uploaded PDF files with robust error handling
    and debugging capabilities.
    """
    
    def __init__(self, show_status: bool = True):
        """
        Initialize the text extractor.
        
        Args:
            show_status: Whether to show extraction status in the Streamlit UI
        """
        self.show_status = show_status
    
    def extract_text_from_pdf(self, pdf_file: BinaryIO, filename: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """
        Extract text from a PDF file with error handling and debugging.
        
        Args:
            pdf_file: The PDF file object to extract text from
            filename: Optional name of the file for logging purposes
        
        Returns:
            tuple: (extracted_text, error_message)
                If successful, error_message will be None.
                If unsuccessful, extracted_text will be an empty string.
        """
        temp_path = None
        file_display_name = filename or "unnamed PDF"
        
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(pdf_file.getvalue())
                temp_path = temp_file.name
            
            # Open the PDF file
            reader = PdfReader(temp_path)
            
            # Check if the PDF has pages
            if len(reader.pages) == 0:
                logger.warning(f"{file_display_name} has no pages")
                return "", "PDF has no pages"
            
            # Extract text from each page
            text = ""
            page_count = len(reader.pages)
            
            if self.show_status:
                st.write(f"Extracting text from {file_display_name} ({page_count} pages)")
            
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                
                if page_text:
                    text += page_text + "\n"
                    
                    # Show preview of first page for debugging
                    if i == 0 and self.show_status:
                        preview = page_text[:100] + "..." if len(page_text) > 100 else page_text
                        st.write(f"First page preview: {preview}")
                else:
                    logger.warning(f"Could not extract text from page {i+1} in {file_display_name}")
                    if self.show_status:
                        st.warning(f"Could not extract text from page {i+1} in {file_display_name}")
            
            # Show text statistics
            text = text.strip()
            word_count = len(text.split())
            
            if self.show_status:
                st.write(f"Extracted {len(text)} characters, approximately {word_count} words")
            
            # Check if any text was extracted
            if not text:
                error_msg = "Could not extract any text from PDF"
                logger.warning(f"{error_msg}: {file_display_name}")
                return "", error_msg
            
            logger.info(f"Successfully extracted {word_count} words from {file_display_name}")
            return text, None
            
        except Exception as e:
            error_msg = f"Error extracting text: {str(e)}"
            logger.error(f"{error_msg} from {file_display_name}")
            return "", error_msg
            
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"Error removing temporary file {temp_path}: {str(e)}")