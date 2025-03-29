# tests/test_chunking.py
import sys
import os
import logging

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for more verbose output
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import components
from core.models.document import Document
from utils.chunking import chunk_document

# Test document text
TEST_DOCUMENT = """
Beyond Notes is an AI-powered application that processes Microsoft Teams meeting transcripts 
to extract meaningful insights, action items, and issues. It employs a multi-agent architecture 
where specialized AI agents collaborate to analyze documents and produce structured, consistent outputs.

The application serves two primary purposes:
1. Practical utility - Creating useful summaries and analyses of meeting transcripts for business teams
2. Experimental platform - Providing a playground for exploring advanced AI techniques and agent-based architectures
"""

def test_document_chunking():
    """Test basic document and chunking functionality."""
    try:
        # Create document
        logger.info("Creating document")
        document = Document(text=TEST_DOCUMENT)
        logger.info(f"Created document: {len(document.text)} chars, {document.word_count} words")
        
        # Test chunking function directly
        logger.info("Starting chunking")
        chunks = chunk_document(document, target_chunk_size=150, overlap=20)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Print chunk details
        for i, chunk in enumerate(chunks):
            logger.info(f"Chunk {i+1} details:")
            for key, value in chunk.items():
                if key == 'text':
                    logger.info(f"  {key}: {value[:50]}...")
                else:
                    logger.info(f"  {key}: {value}")
            logger.info("-" * 30)
        
        logger.info("Chunking test completed successfully")
        return chunks
    except Exception as e:
        logger.error(f"Chunking error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting chunking test")
        chunks = test_document_chunking()
        logger.info(f"Test completed. Created {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")