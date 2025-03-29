# utils/chunking.py (fixed version)
from typing import List, Dict, Any
import re
from core.models.document import Document

def chunk_document(document: Document, target_chunk_size: int = 10000, overlap: int = 500) -> List[Dict[str, Any]]:
    """
    Split a document into chunks of approximately target_chunk_size tokens
    with overlap between chunks.
    
    Args:
        document: Document object to chunk
        target_chunk_size: Target size of each chunk in approximate tokens
        overlap: Number of tokens to overlap between chunks
        
    Returns:
        List of chunks with metadata
    """
    # Get text from document
    text = document.text
    
    # Simple approach: estimate tokens based on words
    words = text.split()
    total_words = len(words)
    
    # Safeguard against empty documents
    if total_words == 0:
        return []
    
    # Ensure reasonable chunk size
    target_words = max(10, min(total_words, int(target_chunk_size / 1.3)))
    overlap_words = min(int(target_words / 2), max(0, int(overlap / 1.3)))
    
    # For very small documents, just return one chunk
    if total_words <= target_words:
        return [{
            "text": text,
            "chunk_index": 0,
            "word_count": total_words,
            "start_position": 0,
            "end_position": total_words - 1,
            "estimated_tokens": int(total_words * 1.3)
        }]
    
    chunks = []
    current_pos = 0
    
    # Instead of using a while loop, use a more deterministic approach
    # Calculate how many chunks we should create
    step_size = target_words - overlap_words
    if step_size <= 0:
        step_size = 1  # Ensure we make progress
    
    # Calculate the number of chunks we'll need
    num_chunks = (total_words + step_size - 1) // step_size  # Ceiling division
    
    # Limit the number of chunks for very large documents
    max_chunks = 100  # Reasonable limit
    if num_chunks > max_chunks:
        # Recalculate the step size to get about max_chunks
        step_size = (total_words + max_chunks - 1) // max_chunks
        target_words = min(total_words, step_size + overlap_words)
        num_chunks = (total_words + step_size - 1) // step_size
    
    # Create chunks deterministically
    for i in range(num_chunks):
        start_pos = i * step_size
        end_pos = min(start_pos + target_words, total_words)
        
        # Get the text for this chunk
        chunk_words = words[start_pos:end_pos]
        chunk_text = " ".join(chunk_words)
        
        # Add metadata
        chunks.append({
            "text": chunk_text,
            "chunk_index": i,
            "word_count": len(chunk_words),
            "start_position": start_pos,
            "end_position": end_pos - 1,
            "estimated_tokens": int(len(chunk_words) * 1.3)
        })
        
        # Stop if we've reached the end of the document
        if end_pos >= total_words:
            break
    
    return chunks