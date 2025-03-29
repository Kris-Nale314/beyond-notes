# app.py
import os
import time
import streamlit as st
import asyncio
import logging
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
logger.info("Environment variables loaded")

# Import core components
from utils.paths import AppPaths
from core.models.document import Document
from core.llm.customllm import CustomLLM
from utils.chunking import chunk_document

# Ensure directories exist
AppPaths.ensure_dirs()
logger.info(f"Directory structure ensured at {AppPaths.ROOT}")

# Configure Streamlit page
st.set_page_config(
    page_title="Beyond Notes",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #6c5ce7;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: 500;
        color: #a29bfe;
        margin-bottom: 2rem;
    }
    .progress-container {
        padding: 1rem;
        background-color: rgba(108, 92, 231, 0.1);
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .summary-container {
        padding: 1rem;
        background-color: rgba(108, 92, 231, 0.05);
        border-radius: 0.5rem;
        border-left: 5px solid #6c5ce7;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Check for API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ OpenAI API key not found. Please add it to your .env file.")
    logger.error("OpenAI API key not found")
    st.stop()
else:
    logger.info("OpenAI API key found")

# Initialize LLM
llm = CustomLLM(api_key)
logger.info("LLM initialized")

# Async function to summarize a single chunk
async def summarize_chunk(chunk, chunk_index, total_chunks):
    """Summarize a single document chunk."""
    logger.info(f"Starting summarization of chunk {chunk_index+1}/{total_chunks}")
    
    # Create a prompt for summarization
    prompt = f"""
    Please summarize the following text chunk ({chunk_index+1} of {total_chunks}):
    
    {chunk['text']}
    
    Provide a concise summary that captures the key points, maintaining any important details, 
    names, numbers, or findings. The summary should be about 1/4 the length of the original.
    """
    
    try:
        # Call LLM
        summary = await llm.generate_completion(prompt, max_tokens=500, temperature=0.3)
        logger.info(f"Completed summarization of chunk {chunk_index+1}/{total_chunks}")
        return {
            "chunk_index": chunk_index,
            "summary": summary,
            "word_count": len(summary.split()),
            "original_word_count": chunk['word_count']
        }
    except Exception as e:
        logger.error(f"Error summarizing chunk {chunk_index+1}: {str(e)}")
        return {
            "chunk_index": chunk_index,
            "summary": f"Error summarizing chunk: {str(e)}",
            "error": str(e)
        }

# Async function to combine summaries
async def combine_summaries(summaries, document):
    """Combine all chunk summaries into one coherent summary."""
    logger.info("Starting final summary combination")
    
    # Sort summaries by chunk index
    sorted_summaries = sorted(summaries, key=lambda x: x["chunk_index"])
    
    # Extract the summary texts
    summary_texts = [s["summary"] for s in sorted_summaries]
    combined_text = "\n\n".join(summary_texts)
    
    # Create prompt for the combined summary
    prompt = f"""
    Below are summaries of different sections of a document titled "{document.filename or 'Document'}".
    
    {combined_text}
    
    Please create a single coherent summary that combines all of these sections.
    Organize the information logically, eliminate redundancies, and ensure smooth transitions.
    The final summary should provide a comprehensive overview of the entire document, 
    highlighting the most important points, findings, or conclusions.
    """
    
    try:
        # Call LLM
        final_summary = await llm.generate_completion(prompt, max_tokens=1000, temperature=0.3)
        logger.info("Completed final summary combination")
        return final_summary
    except Exception as e:
        logger.error(f"Error combining summaries: {str(e)}")
        return f"Error creating final summary: {str(e)}"

# App title
st.markdown('<div class="main-header">Beyond Notes</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Transform Documents into Structured Insights</div>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("Processing Options")
    
    model = st.selectbox(
        "Model",
        ["gpt-3.5-turbo", "gpt-4"],
        index=0
    )
    
    st.subheader("Chunking Options")
    chunk_size = st.slider(
        "Chunk Size (tokens)",
        min_value=1000,
        max_value=20000,
        value=8000,
        step=1000
    )
    
    chunk_overlap = st.slider(
        "Chunk Overlap (tokens)",
        min_value=0,
        max_value=2000,
        value=300,
        step=100
    )
    
    st.markdown("---")
    st.caption("Beyond Notes v0.1.0")

# Main area
tab1, tab2 = st.tabs(["Upload Document", "Summary Results"])

with tab1:
    st.header("Upload Document")
    
    uploaded_file = st.file_uploader("Choose a text file", type=["txt"])
    
    if uploaded_file:
        # Display file details
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.1f} KB"
        }
        
        st.write("File Details:")
        for k, v in file_details.items():
            st.write(f"- {k}: {v}")
        
        # Load the document
        try:
            # Reset the file pointer
            uploaded_file.seek(0)
            
            # Create document using our Document class
            document = Document.from_uploaded_file(uploaded_file)
            logger.info(f"Loaded document: {document.filename}, {document.word_count} words")
            
            # Display document preview
            with st.expander("Document Preview", expanded=False):
                st.text_area(
                    "Content Preview", 
                    document.text[:1000] + "..." if len(document.text) > 1000 else document.text, 
                    height=200
                )
            
            # Display document metadata
            st.subheader("Document Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Word count: {document.word_count:,}")
                st.write(f"Estimated tokens: {document.estimated_tokens:,}")
            with col2:
                st.write(f"Character count: {document.character_count:,}")
                st.write(f"Line count: {document.line_count:,}")
            
            # Process document button
            if st.button("Process Document"):
                # Record start time
                start_time = time.time()
                logger.info("Starting document processing")
                
                # Progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Chunking document...")
                
                # Step 1: Chunk the document
                chunks = chunk_document(document, target_chunk_size=chunk_size, overlap=chunk_overlap)
                logger.info(f"Document chunked into {len(chunks)} chunks")
                
                # Update progress
                progress_bar.progress(0.1)
                status_text.text(f"Document divided into {len(chunks)} chunks. Starting summarization...")
                
                # Step 2: Process chunks in parallel
                async def process_all_chunks():
                    # Create tasks for all chunks
                    tasks = [
                        summarize_chunk(chunk, i, len(chunks)) 
                        for i, chunk in enumerate(chunks)
                    ]
                    
                    # Process up to 3 chunks in parallel
                    summaries = []
                    for i in range(0, len(tasks), 3):
                        batch = tasks[i:i+3]
                        batch_results = await asyncio.gather(*batch)
                        summaries.extend(batch_results)
                        
                        # Update progress after each batch
                        progress_value = 0.1 + 0.7 * (min(i + 3, len(tasks)) / len(tasks))
                        progress_bar.progress(progress_value)
                        status_text.text(f"Summarized {min(i + 3, len(tasks))}/{len(tasks)} chunks...")
                    
                    return summaries
                
                # Step 3: Execute the processing
                with st.spinner("Processing document chunks..."):
                    # Run the async function
                    chunk_summaries = asyncio.run(process_all_chunks())
                    logger.info(f"Completed summarization of {len(chunk_summaries)} chunks")
                    
                    # Update progress
                    progress_bar.progress(0.8)
                    status_text.text("Combining summaries into final document...")
                    
                    # Combine summaries
                    final_summary = asyncio.run(combine_summaries(chunk_summaries, document))
                    logger.info("Final summary generated")
                    
                    # Complete progress
                    progress_bar.progress(1.0)
                    status_text.text("Processing complete!")
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    logger.info(f"Document processing completed in {processing_time:.2f} seconds")
                    
                    # Store results in session state
                    st.session_state.document = document
                    st.session_state.chunks = chunks
                    st.session_state.chunk_summaries = chunk_summaries
                    st.session_state.final_summary = final_summary
                    st.session_state.processing_time = processing_time
                    
                    # Show success and guide to results tab
                    st.success(f"Document processed in {processing_time:.2f} seconds! See the Summary Results tab.")
                    
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            st.error(f"Error processing document: {str(e)}")

with tab2:
    st.header("Summary Results")
    
    if 'final_summary' not in st.session_state:
        st.info("Upload and process a document to see results here.")
    else:
        document = st.session_state.document
        chunks = st.session_state.chunks
        chunk_summaries = st.session_state.chunk_summaries
        final_summary = st.session_state.final_summary
        processing_time = st.session_state.processing_time
        
        # Display document info
        st.subheader("Document Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Filename: {document.filename}")
            st.write(f"Word count: {document.word_count:,}")
            st.write(f"Processing time: {processing_time:.2f} seconds")
        with col2:
            st.write(f"Chunks created: {len(chunks)}")
            st.write(f"Avg chunk size: {sum(c.get('word_count', 0) for c in chunks) / len(chunks):.0f} words")
        
        # Display final summary
        st.subheader("Final Summary")
        st.markdown(f'<div class="summary-container">{final_summary}</div>', unsafe_allow_html=True)
        
        # Display word count reduction
        original_words = document.word_count
        summary_words = len(final_summary.split())
        reduction = (1 - (summary_words / original_words)) * 100
        
        st.write(f"Original document: {original_words:,} words")
        st.write(f"Summary: {summary_words:,} words")
        st.write(f"Reduction: {reduction:.1f}%")
        
        # Individual chunk summaries
        with st.expander("Individual Chunk Summaries", expanded=False):
            # Sort summaries by chunk index
            sorted_summaries = sorted(chunk_summaries, key=lambda x: x["chunk_index"])
            
            for summary in sorted_summaries:
                st.markdown(f"**Chunk {summary['chunk_index']+1}**")
                st.markdown(f'<div class="summary-container">{summary["summary"]}</div>', unsafe_allow_html=True)
                if "original_word_count" in summary:
                    chunk_reduction = (1 - (summary["word_count"] / summary["original_word_count"])) * 100
                    st.write(f"Original: {summary.get('original_word_count', 0):,} words | Summary: {summary.get('word_count', 0):,} words | Reduction: {chunk_reduction:.1f}%")
                st.markdown("---")
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            summary_text = final_summary
            st.download_button(
                label="Download Summary",
                data=summary_text,
                file_name=f"summary_{document.filename}",
                mime="text/plain"
            )
        
        with col2:
            all_summaries = "# FULL DOCUMENT SUMMARY\n\n" + final_summary + "\n\n# INDIVIDUAL CHUNK SUMMARIES\n\n"
            for summary in sorted_summaries:
                all_summaries += f"\n## Chunk {summary['chunk_index']+1}\n\n" + summary["summary"] + "\n\n"
            
            st.download_button(
                label="Download All Summaries",
                data=all_summaries,
                file_name=f"all_summaries_{document.filename}",
                mime="text/plain"
            )