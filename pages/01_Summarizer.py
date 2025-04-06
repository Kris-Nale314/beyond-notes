# pages/01_Summarizer.py
import streamlit as st
import os
import asyncio
import logging
import json
import time
from pathlib import Path
from datetime import datetime

# Configure logging for both file and console output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("summarizer.log")
    ]
)
logger = logging.getLogger("beyond-notes-summarizer")

# Add debug message to confirm script is running
logger.info("01_Summarizer.py is starting")

# Import components with clear error handling
try:
    logger.info("Importing core components...")
    from core.models.document import Document
    from core.models.summary_renderer import render_summary
    from core.orchestrator import Orchestrator
    from assessments.loader import AssessmentLoader
    from utils.paths import AppPaths
    from utils.formatting import format_assessment_report
    logger.info("Successfully imported all components")
except ImportError as e:
    error_msg = f"Failed to import components: {e}"
    logger.error(error_msg, exc_info=True)
    st.error(error_msg)
    st.stop()

# Ensure directories exist
try:
    logger.info("Ensuring directories exist...")
    AppPaths.ensure_dirs()
    logger.info("Directories verified")
except Exception as e:
    error_msg = f"Failed to ensure directories: {e}"
    logger.error(error_msg, exc_info=True)
    st.error(error_msg)
    st.stop()

# Verify assessment configuration loading
try:
    logger.info("Loading assessment configurations...")
    loader = AssessmentLoader()
    configs = loader.get_assessment_configs_list()
    config_ids = [c.get("assessment_id") for c in configs]
    logger.info(f"Available assessment IDs: {config_ids}")
    
    # Check for base distill assessment
    base_distill_id = "base_distill_summary_v1"  
    if base_distill_id in config_ids:
        logger.info(f"✅ Base distill config '{base_distill_id}' found")
    else:
        logger.warning(f"⚠️ Base distill config '{base_distill_id}' NOT found in available configs")
except Exception as e:
    error_msg = f"Error checking configurations: {e}"
    logger.error(error_msg, exc_info=True)
    st.error(f"Configuration error: {error_msg}")
    # Continue anyway to show UI with error messages

# Page config
st.set_page_config(
    page_title="Beyond Notes - Document Summarizer",
    page_icon="📝",
    layout="wide",
)

# CSS styles
st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .subheader {
        font-size: 1.1rem;
        opacity: 0.8;
        margin-bottom: 2rem;
    }
    
    /* Section headers with numbers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    
    .section-number {
        display: inline-flex;
        justify-content: center;
        align-items: center;
        width: 36px;
        height: 36px;
        background-color: #2196F3;
        color: white;
        border-radius: 50%;
        text-align: center;
        margin-right: 10px;
        font-weight: 600;
    }
    
    /* Document metadata */
    .document-meta {
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 6px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .document-meta-label {
        font-size: 0.9rem;
        opacity: 0.7;
        margin-bottom: 0.25rem;
    }
    
    .document-meta-value {
        font-weight: 500;
    }
    
    /* Progress status */
    .stProgress > div > div > div > div {
        background-color: #2196F3;
    }
    
    /* Format selection cards */
    .format-option {
        background-color: rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
    }
    
    .format-option:hover {
        background-color: rgba(33, 150, 243, 0.1);
        border-color: rgba(33, 150, 243, 0.3);
    }
    
    .format-option.selected {
        background-color: rgba(33, 150, 243, 0.15);
        border-color: #2196F3;
    }
    
    .format-name {
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    
    .format-desc {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    /* Summary container */
    .summary-container {
        margin: 1.5rem 0;
    }
    
    /* Download buttons container */
    .download-container {
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_page():
    """Initialize the session state variables."""
    logger.info("Initializing page session state")
    
    if "document_loaded" not in st.session_state:
        st.session_state.document_loaded = False
    
    if "processing_started" not in st.session_state:
        st.session_state.processing_started = False
    
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    
    if "current_progress" not in st.session_state:
        st.session_state.current_progress = 0.0
    
    if "progress_message" not in st.session_state:
        st.session_state.progress_message = "Not started"
    
    if "summary_result" not in st.session_state:
        st.session_state.summary_result = None
        
    if "selected_format" not in st.session_state:
        st.session_state.selected_format = "executive"
    
    logger.info("Session state initialized")

def load_document(file_object):
    """Load document from uploaded file."""
    logger.info(f"Loading document: {file_object.name}")
    
    try:
        # Save uploaded file temporarily
        temp_dir = AppPaths.get_temp_path("uploads")
        temp_dir.mkdir(exist_ok=True, parents=True)
        temp_file_path = temp_dir / file_object.name
        
        with open(temp_file_path, "wb") as f:
            f.write(file_object.getvalue())
        
        # Create document from file
        logger.info(f"Creating Document object from {temp_file_path}")
        document = Document.from_file(temp_file_path)
        logger.info(f"Successfully loaded document: {document.filename}, {document.word_count} words")
        
        return document, temp_file_path
    except Exception as e:
        error_msg = f"Error loading document: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        return None, None

def progress_callback(progress, data):
    """Callback for progress updates from the orchestrator."""
    # Log progress to console
    if isinstance(data, dict):
        message = data.get("message", "Processing...")
        current_stage = data.get("current_stage", "")
        logger.info(f"Progress {progress:.1%} - Stage: {current_stage} - {message}")
        
        # Update session state
        st.session_state.current_progress = float(progress)
        st.session_state.progress_message = message
        st.session_state.current_stage = current_stage
        st.session_state.stages_info = data.get("stages", {})
    else:
        # Simple string message
        logger.info(f"Progress {progress:.1%} - {data}")
        st.session_state.current_progress = float(progress)
        st.session_state.progress_message = data

def render_pipeline_status():
    """Display the current pipeline status."""
    progress_value = float(st.session_state.get("current_progress", 0.0))
    progress_bar = st.progress(progress_value)
    
    # Current stage and progress message
    current_stage = st.session_state.get("current_stage", "")
    progress_message = st.session_state.get("progress_message", "Waiting to start...")
    
    if current_stage:
        st.write(f"**Current Stage:** {current_stage.replace('_', ' ').title()}")
    
    st.caption(progress_message)
    
    # Show detailed stage information if available
    stages_info = st.session_state.get("stages_info", {})
    if stages_info:
        with st.expander("View detailed progress", expanded=False):
            for stage_name, stage_info in stages_info.items():
                status = stage_info.get("status", "not_started")
                progress = float(stage_info.get("progress", 0.0))
                message = stage_info.get("message", "")
                
                # Determine emoji based on status
                if status == "completed": emoji = "✅"
                elif status == "running": emoji = "⏳"
                elif status == "failed": emoji = "❌"
                else: emoji = "⏱️"
                
                display_name = stage_name.replace("_", " ").title()
                progress_pct = f"{int(progress * 100)}%" if progress > 0 else ""
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"{emoji} **{display_name}** {progress_pct}")
                    if message:
                        st.caption(f"> {message[:100]}")
                with col2:
                    st.markdown(f"`{status.upper()}`")

def display_document_preview(document):
    """Display a preview of the loaded document."""
    logger.info(f"Displaying preview for document: {document.filename}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Document Preview")
        
        preview_length = min(1000, len(document.text))
        preview_text = document.text[:preview_length]
        if len(document.text) > preview_length:
            preview_text += "..."
        
        with st.expander("Document Content Preview", expanded=False):
            st.text_area("", preview_text, height=200, disabled=True)
    
    with col2:
        st.markdown('<div class="document-meta">', unsafe_allow_html=True)
        
        st.markdown('<div class="document-meta-label">Filename</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="document-meta-value">{document.filename}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="document-meta-label">Word Count</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="document-meta-value">{document.word_count:,}</div>', unsafe_allow_html=True)
        
        file_size = os.path.getsize(str(document.file_path)) if hasattr(document, 'file_path') and document.file_path else 0
        if file_size > 0:
            size_kb = file_size / 1024
            size_display = f"{size_kb:.1f} KB"
            st.markdown('<div class="document-meta-label">File Size</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="document-meta-value">{size_display}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def select_summary_format():
    """Let the user select a summary format."""
    # Format options with descriptions
    format_options = {
        "executive": "Very concise overview focusing on key findings, decisions, and implications.",
        "comprehensive": "Detailed summary with supporting information and context.",
        "bullet_points": "Key points organized in an easy-to-scan bullet list format.",
        "narrative": "Flowing narrative that preserves the document's natural story."
    }
    
    # Get current selection
    selected_format = st.session_state.get("selected_format", "comprehensive")
    
    # Display format selection using radio buttons with custom formatting
    st.write("### Select Summary Format")
    
    # Create placeholders for radio buttons with custom styling
    for format_key, description in format_options.items():
        # Format display name
        format_display = format_key.replace("_", " ").title()
        
        # Check if this format is selected
        is_selected = format_key == selected_format
        selected_class = "selected" if is_selected else ""
        
        # Create styled container
        format_html = f"""
        <div class="format-option {selected_class}">
            <div class="format-name">{format_display}</div>
            <div class="format-desc">{description}</div>
        </div>
        """
        
        # Use columns to create the effect of a selectable card
        col1, col2 = st.columns([0.1, 0.9])
        with col1:
            is_checked = st.radio("", [True], key=f"radio_{format_key}", label_visibility="collapsed") if format_key == selected_format else st.radio("", [False], key=f"radio_{format_key}", label_visibility="collapsed")
            if is_checked and format_key != selected_format:
                st.session_state.selected_format = format_key
                logger.info(f"Format selection changed to: {format_key}")
                st.rerun()
        with col2:
            st.markdown(format_html, unsafe_allow_html=True)
    
    logger.info(f"Selected format: {selected_format}")
    return selected_format

async def process_document(document, assessment_id, options):
    """Process document using the orchestrator."""
    logger.info(f"Starting document processing with assessment ID: {assessment_id}")
    logger.info(f"Processing options: {options}")
    
    start_time = time.time()
    
    try:
        # Get API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            error_msg = "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
            logger.error(error_msg)
            st.error(error_msg)
            return None
        
        # Initialize orchestrator
        logger.info("Initializing orchestrator...")
        orchestrator = Orchestrator(assessment_id, options=options, api_key=api_key)
        
        # Set progress callback
        orchestrator.set_progress_callback(progress_callback)
        
        # Process document
        logger.info(f"Processing document with orchestrator: {document.filename}")
        result = await orchestrator.process_document(document)
        
        # Log completion time
        elapsed_time = time.time() - start_time
        logger.info(f"Document processing completed in {elapsed_time:.2f} seconds")
        
        # Update state
        st.session_state.summary_result = result
        st.session_state.processing_complete = True
        
        return result
    except Exception as e:
        error_msg = f"Error processing document: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        
        # Update state even on error
        st.session_state.processing_complete = True
        return None

def display_summary_result(result, format_type):
    """Display the summary result using our renderer."""
    logger.info(f"Displaying summary result with format: {format_type}")
    
    try:
        # Generate HTML with our renderer
        logger.info("Generating HTML with summary renderer...")
        html_content = render_summary(result, format_type)
        
        # Display the summary in Streamlit
        st.markdown('<div class="summary-container">', unsafe_allow_html=True)
        st.markdown(html_content, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Download options
        st.markdown('<div class="download-container">', unsafe_allow_html=True)
        
        # Generate report markdown
        report_md = format_assessment_report(result, "distill")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download as markdown
            st.download_button(
                "Download as Markdown",
                data=report_md,
                file_name=f"summary_{timestamp}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with col2:
            # Download as JSON
            json_str = json.dumps(result, indent=2)
            st.download_button(
                "Download Raw Data",
                data=json_str,
                file_name=f"summary_data_{timestamp}.json",
                mime="application/json",
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        logger.info("Summary display completed successfully")
        
    except Exception as e:
        error_msg = f"Error rendering summary: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        
        # Fallback to basic JSON display
        st.write("### Summary Data (Fallback View)")
        st.json(result.get("result", {}))

def main():
    """Main function for the summarizer page."""
    logger.info("--- Starting Summarizer Page Execution ---")
    
    # Initialize page
    initialize_page()
    
    # Header
    st.markdown('<div class="main-header">📄 Document Summarizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Transform documents into clear, readable summaries</div>', unsafe_allow_html=True)
    
    # Document upload section
    st.markdown('<div class="section-header"><span class="section-number">1</span>Upload Document</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a document to summarize", type=["txt", "md", "pdf", "docx"])
    
    if uploaded_file:
        logger.info(f"File uploaded: {uploaded_file.name}")
        
        # Load document
        document, file_path = load_document(uploaded_file)
        if document:
            st.session_state.document = document
            st.session_state.document_loaded = True
            display_document_preview(document)
    
    # Configuration section
    if st.session_state.document_loaded:
        st.markdown('<div class="section-header"><span class="section-number">2</span>Configure Summary</div>', unsafe_allow_html=True)
        
        # Format selection
        selected_format = select_summary_format()
        
        # Model settings in expander
        with st.expander("Advanced Options", expanded=False):
            # Model selection
            model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
            selected_model = st.selectbox("LLM Model", options=model_options, index=1)  # Default to GPT-4
            
            # Temperature
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
            
            # Include quotes option
            include_quotes = st.checkbox("Include Key Quotes", value=True)
            
            # Chunk settings
            chunk_size = st.slider("Chunk Size", min_value=1000, max_value=15000, value=8000, step=1000)
            chunk_overlap = st.slider("Chunk Overlap", min_value=100, max_value=1000, value=300, step=100)
        
        # Process button
        col1, col2 = st.columns([3, 1])
        with col1:
            if not st.session_state.processing_started:
                if st.button("Generate Summary", type="primary", use_container_width=True):
                    logger.info("Generate Summary button clicked")
                    
                    # Create summary options
                    summary_options = {
                        "format": selected_format,
                        "include_quotes": include_quotes
                    }
                    
                    # Create processing options
                    processing_options = {
                        "model": selected_model,
                        "temperature": temperature,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "user_options": summary_options
                    }
                    
                    # Set the base distill assessment ID
                    # Use your actual assessment ID here
                    selected_assessment_id = "base_distill_summary_v1"
                    
                    logger.info(f"Using assessment ID: {selected_assessment_id}")
                    logger.info(f"Processing options: {processing_options}")
                    
                    # Store options for async processing
                    st.session_state.processing_options = processing_options
                    st.session_state.selected_assessment_id = selected_assessment_id
                    
                    # Update state
                    st.session_state.processing_started = True
                    st.session_state.processing_complete = False
                    st.session_state.current_progress = 0.0
                    st.session_state.progress_message = "Starting..."
                    
                    st.rerun()
        
        with col2:
            if st.button("Reset", use_container_width=True):
                logger.info("Reset button clicked")
                
                # Reset session state
                st.session_state.document_loaded = False
                st.session_state.processing_started = False
                st.session_state.processing_complete = False
                st.session_state.summary_result = None
                st.session_state.current_progress = 0.0
                st.session_state.progress_message = "Not started"
                
                logger.info("Session state reset")
                st.rerun()
    
    # Processing section
    if st.session_state.processing_started:
        st.markdown('<div class="section-header"><span class="section-number">3</span>Processing Status</div>', unsafe_allow_html=True)
        
        if not st.session_state.processing_complete:
            logger.info("Processing in progress, showing status...")
            
            # Show progress
            render_pipeline_status()
            
            # Process document if not already processing
            if 'is_processing' not in st.session_state:
                logger.info("Starting document processing...")
                
                document_to_process = st.session_state.document
                assessment_id = st.session_state.selected_assessment_id
                options = st.session_state.processing_options
                
                status_placeholder = st.empty()
                status_placeholder.info("Processing document... Please wait")
                
                # Mark as processing to prevent multiple runs
                st.session_state.is_processing = True
                
                try:
                    # Run async processing
                    asyncio.run(process_document(document_to_process, assessment_id, options))
                    logger.info("Document processing completed")
                except Exception as e:
                    error_msg = f"Error during document processing: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    status_placeholder.error(error_msg)
                finally:
                    # Reset processing flag
                    if 'is_processing' in st.session_state:
                        del st.session_state.is_processing
                    
                    # This rerun will either show results or error message
                    st.rerun()
        
        # Results section
        if st.session_state.processing_complete:
            st.markdown('<div class="section-header"><span class="section-number">4</span>Summary</div>', unsafe_allow_html=True)
            
            if st.session_state.summary_result:
                logger.info("Displaying summary results")
                result = st.session_state.summary_result
                format_type = st.session_state.selected_format
                display_summary_result(result, format_type)
            else:
                error_msg = "Processing completed but no summary was generated. Please check the logs for details."
                logger.error(error_msg)
                st.error(error_msg)

if __name__ == "__main__":
    try:
        logger.info("Starting 01_Summarizer.py main execution")
        main()
        logger.info("01_Summarizer.py execution completed successfully")
    except Exception as e:
        error_msg = f"Unhandled exception in main execution: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}")