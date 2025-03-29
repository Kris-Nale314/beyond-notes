# core/models/document.py
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

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
        self.filename = filename
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
    
    def _process_text(self, text: str) -> None:
        """Process text content and compute metadata."""
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
        """Load document from bytes with multiple encoding attempts."""
        # Try UTF-8 first (most common)
        try:
            text = bytes_data.decode("utf-8")
            self._process_text(text)
            self.metadata["encoding"] = "utf-8"
            return
        except UnicodeDecodeError:
            pass
        
        # Try other common encodings
        encodings = ["latin-1", "windows-1252", "iso-8859-1", "cp1252"]
        for encoding in encodings:
            try:
                text = bytes_data.decode(encoding)
                self._process_text(text)
                self.metadata["encoding"] = encoding
                return
            except UnicodeDecodeError:
                continue
        
        # If we reach here, we couldn't decode the file
        raise ValueError("Could not decode file with any common encoding")
    
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
        
        with open(file_path, 'rb') as f:
            bytes_data = f.read()
            
        return cls(filename=filename, bytes_data=bytes_data)
    
    @classmethod
    def from_uploaded_file(cls, file_obj) -> 'Document':
        """Create a Document from a Streamlit uploaded file."""
        filename = file_obj.name
        bytes_data = file_obj.read()
        
        return cls(filename=filename, bytes_data=bytes_data)