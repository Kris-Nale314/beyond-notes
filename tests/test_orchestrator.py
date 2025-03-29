# tests/test_orchestrator.py
import os
import sys
import asyncio
import logging
import time
from dotenv import load_dotenv

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import components
from core.models.document import Document
from utils.chunking import chunk_document
from core.llm.customllm import CustomLLM
from orchestrator import Orchestrator

# Load environment variables
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    logger.error("No OpenAI API key found in environment variables.")
    sys.exit(1)

# Test document text
TEST_DOCUMENT = """
Beyond Notes is an AI-powered application that processes Microsoft Teams meeting transcripts 
to extract meaningful insights, action items, and issues. It employs a multi-agent architecture 
where specialized AI agents collaborate to analyze documents and produce structured, consistent outputs.

The application serves two primary purposes:
1. Practical utility - Creating useful summaries and analyses of meeting transcripts for business teams
2. Experimental platform - Providing a playground for exploring advanced AI techniques and agent-based architectures

Key Features:
- Meeting Transcript Processing - Clean summaries and automated follow-up actions
- Issue Extraction and Categorization - Identify problems with severity ratings
- Cross-Transcript Analysis - Discover patterns across multiple calls
- Customizable Assessment Types - Configure and extend analysis approaches
- Structured Outputs - Consistent formats for easy consumption
- Interactive UI - Post-analysis conversation with documents

The architecture uses a multi-agent approach where different specialized components work together:
1. Planner Agent - Analyzes documents and creates tailored instructions for other agents
2. Extractor Agent - Identifies relevant information from document chunks
3. Aggregator Agent - Combines similar findings and eliminates duplicates
4. Evaluator Agent - Determines importance, severity, and relationships between findings
5. Formatter Agent - Creates structured, navigable reports
6. Reviewer Agent - Performs quality control on the final output
"""

async def test_progress_callback(progress, message):
    """Simple progress callback for testing."""
    logger.info(f"Progress: {progress:.2f} - {message}")

async def test_document_chunking():
    """Test document creation and chunking."""
    logger.info("-" * 40)
    logger.info("TESTING DOCUMENT CHUNKING")
    logger.info("-" * 40)
    
    # Create document
    start_time = time.time()
    document = Document(text=TEST_DOCUMENT)
    logger.info(f"Created document: {len(document.text)} chars, {document.word_count} words")
    
    # Chunk document
    chunks = chunk_document(document, target_chunk_size=150, overlap=20)
    logger.info(f"Created {len(chunks)} chunks in {time.time() - start_time:.2f}s")
    
    # Print chunks
    for i, chunk in enumerate(chunks):
        logger.info(f"Chunk {i+1}: {chunk['word_count']} words, positions {chunk['start_position']}-{chunk['end_position']}")
        logger.info(f"  Preview: {chunk['text'][:50]}...")
    
    return document, chunks

async def test_llm_connection():
    """Test LLM connection with a simple prompt."""
    logger.info("-" * 40)
    logger.info("TESTING LLM CONNECTION")
    logger.info("-" * 40)
    
    start_time = time.time()
    llm = CustomLLM(api_key)
    
    prompt = "Summarize this in one sentence: Beyond Notes is an AI application for processing meeting transcripts."
    
    logger.info(f"Sending prompt: {prompt}")
    response = await llm.generate_completion(prompt, max_tokens=50)
    logger.info(f"Response received in {time.time() - start_time:.2f}s: {response}")
    
    return llm

async def test_chunk_summarization(llm, chunk):
    """Test summarizing a single chunk."""
    logger.info("-" * 40)
    logger.info("TESTING CHUNK SUMMARIZATION")
    logger.info("-" * 40)
    
    start_time = time.time()
    
    # Create prompt
    prompt = f"""
    Please summarize the following text chunk:
    
    {chunk['text']}
    
    Provide a concise summary that captures the key points.
    """
    
    logger.info(f"Sending chunk with {chunk['word_count']} words for summarization")
    response = await llm.generate_completion(prompt, max_tokens=100)
    
    logger.info(f"Summarization completed in {time.time() - start_time:.2f}s")
    logger.info(f"Summary: {response}")
    
    return response

async def test_orchestrator():
    """Test the orchestrator with a simple document."""
    logger.info("-" * 40)
    logger.info("TESTING ORCHESTRATOR")
    logger.info("-" * 40)
    
    start_time = time.time()
    
    # Create document
    document = Document(text=TEST_DOCUMENT)
    
    # Create orchestrator
    options = {
        "chunk_size": 200,
        "chunk_overlap": 50,
        "assessment_type": "issues"
    }
    
    orchestrator = Orchestrator("issues", options)
    
    # Set up progress logging
    async def log_progress(progress, message):
        logger.info(f"Orchestrator progress: {progress:.2f} - {message}")
    
    logger.info(f"Starting orchestrator.process_document")
    try:
        result = await orchestrator.process_document(document, progress_callback=log_progress)
        logger.info(f"Orchestrator completed in {time.time() - start_time:.2f}s")
        logger.info(f"Result metadata: {result.get('metadata', {}).keys()}")
        return result
    except Exception as e:
        logger.error(f"Orchestrator error: {str(e)}", exc_info=True)
        raise

async def test_orchestrator_step_by_step():
    """Test orchestrator components individually to identify blockage."""
    logger.info("-" * 40)
    logger.info("TESTING ORCHESTRATOR STEP BY STEP")
    logger.info("-" * 40)
    
    # Create document
    document = Document(text=TEST_DOCUMENT)
    logger.info(f"Created document: {document.word_count} words")
    
    # Create orchestrator but don't run full process
    options = {
        "chunk_size": 200,
        "chunk_overlap": 50,
        "assessment_type": "issues"
    }
    
    orchestrator = Orchestrator("issues", options)
    
    # Manually initialize context
    from core.models.context import ProcessingContext
    context = ProcessingContext(document.text, options)
    orchestrator.context = context
    
    # Run document analysis step
    logger.info("Running document analysis step")
    try:
        await orchestrator._analyze_document(document)
        logger.info("Document analysis completed successfully")
        logger.info(f"Document info: {context.document_info}")
    except Exception as e:
        logger.error(f"Document analysis error: {str(e)}", exc_info=True)
        raise
    
    # Run chunking step
    logger.info("Running chunking step")
    try:
        await orchestrator._chunk_document(document)
        logger.info("Chunking completed successfully")
        logger.info(f"Created {len(context.chunks)} chunks")
    except Exception as e:
        logger.error(f"Chunking error: {str(e)}", exc_info=True)
        raise
    
    return context

async def main():
    """Run all tests."""
    logger.info("Starting orchestrator tests")
    
    try:
        # Test document and chunking
        document, chunks = await test_document_chunking()
        
        # Test LLM
        llm = await test_llm_connection()
        
        # Test chunk summarization with first chunk
        if chunks:
            summary = await test_chunk_summarization(llm, chunks[0])
        
        # Test orchestrator step by step
        context = await test_orchestrator_step_by_step()
        
        # Test full orchestrator if previous tests pass
        result = await test_orchestrator()
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())