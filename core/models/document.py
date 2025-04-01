# core/models/document.py
import os
import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

# Configure logging
logger = logging.getLogger(__name__)

class Document:
    """Robust document representation with metadata and encoding handling."""
    
    def __init__(self, text: str = None, filename: Optional[str] = None, bytes_data: bytes = None):
        """
        Initialize document from text or bytes.
        
        Args:
            text: Text content of the document (if already decoded)
            filename: Optional filename
            bytes_data: Raw bytes of the document (if not yet decoded)
        """
        self.filename = filename or "unnamed_document.txt"
        self.created_at = datetime.now()
        self.metadata = {}
        
        # Properties to be calculated
        self.text = ""
        self.character_count = 0
        self.word_count = 0
        self.line_count = 0
        self.estimated_tokens = 0
        
        # Load content from either text or bytes
        if text is not None:
            self._process_text(text)
        elif bytes_data is not None:
            self._load_from_bytes(bytes_data)
        else:
            raise ValueError("Either text or bytes_data must be provided")
    
    def _process_text(self, text: str) -> None:
        """Process text content and compute metadata."""
        if not isinstance(text, str):
            raise TypeError(f"Expected string for text, got {type(text)}")
            
        self.text = text
        self.character_count = len(text)
        self.word_count = len(text.split())
        self.line_count = len(text.splitlines())
        
        # Simple token estimation (approximation)
        # Average English word is about 4-5 characters + space
        # OpenAI tokenizer typically produces more tokens than words
        self.estimated_tokens = int(self.word_count * 1.3)
        
        # Extract metadata based on filename
        if self.filename:
            self._extract_file_metadata()
    
    def _load_from_bytes(self, bytes_data: bytes) -> None:
        """
        Load document from bytes with multiple encoding attempts.
        Tries various encodings and uses the first one that succeeds.
        """
        if not isinstance(bytes_data, bytes):
            raise TypeError(f"Expected bytes for bytes_data, got {type(bytes_data)}")
            
        # List of encodings to try, in order of preference
        encodings = ["utf-8", "latin-1", "windows-1252", "iso-8859-1", "cp1252", "utf-16"]
        
        # Try each encoding
        successful_decode = False
        decode_errors = []
        
        for encoding in encodings:
            try:
                text = bytes_data.decode(encoding)
                self._process_text(text)
                self.metadata["encoding"] = encoding
                successful_decode = True
                logger.info(f"Successfully decoded file '{self.filename}' using {encoding} encoding")
                break
            except UnicodeDecodeError as e:
                decode_errors.append(f"{encoding}: {str(e)}")
                continue
        
        # If all decodings failed, try with error handling
        if not successful_decode:
            try:
                # Use 'replace' error handler to replace invalid bytes with a replacement character
                text = bytes_data.decode("utf-8", errors="replace")
                self._process_text(text)
                self.metadata["encoding"] = "utf-8-replaced"
                self.metadata["encoding_errors"] = True
                logger.warning(f"Decoded file '{self.filename}' with replacement characters due to encoding issues")
                
                # Also log the specific errors for debugging
                logger.debug(f"Encoding errors encountered: {decode_errors}")
            except Exception as e:
                # Final fallback - if even error handling fails
                error_msg = f"Could not decode file '{self.filename}' with any common encoding. " \
                           f"Tried: {', '.join(encodings)}. Errors: {decode_errors}"
                logger.error(error_msg)
                raise ValueError(error_msg) from e
    
    def _extract_file_metadata(self) -> None:
        """Extract metadata based on filename."""
        if not self.filename:
            return
            
        self.metadata["extension"] = os.path.splitext(self.filename)[1].lower()
        self.metadata["file_size_bytes"] = len(self.text.encode('utf-8'))
        
        # Try to extract date from filename
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(\d{2}-\d{2}-\d{4})',  # DD-MM-YYYY
            r'(\d{8})'               # YYYYMMDD
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, self.filename)
            if match:
                self.metadata["date_in_filename"] = match.group(1)
                break
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of document properties."""
        return {
            "filename": self.filename,
            "word_count": self.word_count,
            "estimated_tokens": self.estimated_tokens,
            "character_count": self.character_count,
            "line_count": self.line_count,
            "created_at": self.created_at.isoformat(),
            **self.metadata
        }
    
    def get_chunk_text(self, start: int, end: int) -> str:
        """Get a specific chunk of text by word indices."""
        words = self.text.split()
        chunk_words = words[start:end]
        return " ".join(chunk_words)
    
    @classmethod
    def from_file(cls, file_path: str) -> 'Document':
        """Create a Document from a file path."""
        filename = os.path.basename(file_path)
        
        try:
            with open(file_path, 'rb') as f:
                bytes_data = f.read()
                
            return cls(filename=filename, bytes_data=bytes_data)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
    
    @classmethod
    def from_uploaded_file(cls, file_obj) -> 'Document':
        """Create a Document from a Streamlit uploaded file."""
        try:
            filename = file_obj.name
            bytes_data = file_obj.read()
            
            return cls(filename=filename, bytes_data=bytes_data)
        except Exception as e:
            logger.error(f"Error processing uploaded file {file_obj.name if hasattr(file_obj, 'name') else 'unknown'}: {e}")
            raise