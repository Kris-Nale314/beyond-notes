# pages/03_Issues.py
import streamlit as st
import os
import asyncio
import logging
import json
import time
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("issues.log")
    ]
)
logger = logging.getLogger("beyond-notes-issues")

# Add debug message to confirm script is running
logger.info("02_Issues.py is starting")

# Import components with clear error handling
try:
    logger.info("Importing core components...")
    from core.models.document import Document
    from core.orchestrator import Orchestrator
    from assessments.loader import AssessmentLoader
    from utils.paths import AppPaths
    # Import our new UI components
    from utils.ui.styles import get_base_styles, get_issues_styles
    from utils.ui.components import page_header, section_header, display_document_preview
    from utils.ui.progress import render_detailed_progress
    from utils.ui.renderers import render_issues_result
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

# Set up loader and find available assessments
loader = None
assess_configs = []
base_assess_id = "base_assess_issue_v1"
assessment_available = False

try:
    logger.info("Loading assessment configurations...")
    loader = AssessmentLoader()
    configs = loader.get_assessment_configs_list()
    
    # Log all available configs for debugging
    config_ids = [c.get("assessment_id") for c in configs]
    logger.info(f"All available config IDs: {config_ids}")
    
    # Check for base assess assessment directly
    base_assess_config = loader.load_config(base_assess_id)
    if base_assess_config:
        logger.info(f"✅ Successfully loaded base assess config '{base_assess_id}'")
        assessment_available = True
        # Get type from loaded config
        logger.info(f"Config assessment_type: {base_assess_config.get('assessment_type')}")
    else:
        logger.warning(f"⚠️ Could not load base assess config '{base_assess_id}'")
        
except Exception as e:
    error_msg = f"Error checking configurations: {e}"
    logger.error(error_msg, exc_info=True)
    # Continue to show UI with error messages

# Page config
st.set_page_config(
    page_title="Beyond Notes - Issue Assessment",
    page_icon="⚠️",
    layout="wide",
)

# Apply our shared styles
st.markdown(get_base_styles(), unsafe_allow_html=True)
st.markdown(get_issues_styles(), unsafe_allow_html=True)

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
    
    if "issues_result" not in st.session_state:
        st.session_state.issues_result = None
    
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
        
        # Store stage information
        stages = data.get("stages", {})
        if stages:
            st.session_state.stages_info = stages
            
            # Log detailed stage information
            for stage_name, stage_info in stages.items():
                if stage_info.get("status") == "running":
                    logger.info(f"Active stage: {stage_name} - {stage_info.get('message', '')}")
    else:
        # Simple string message
        logger.info(f"Progress {progress:.1%} - {data}")
        st.session_state.current_progress = float(progress)
        st.session_state.progress_message = data

async def process_document(document, assessment_id, options):
    """Process document using the orchestrator."""
    start_time = time.time()
    logger.info(f"Starting document processing with assessment ID: {assessment_id}")
    logger.info(f"Processing options: {options}")
    
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
        st.session_state.issues_result = result
        st.session_state.processing_complete = True
        
        return result
    except Exception as e:
        error_msg = f"Error processing document: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        
        # Update state even on error
        st.session_state.processing_complete = True
        return None

def main():
    """Main function for the issues page."""
    # Initialize page
    initialize_page()
    
    # Page header using our component
    page_header("⚠️ Issue Assessment", "Identify problems, risks, and challenges in your documents")
    
    # Show warning if assessment is not available
    if not assessment_available:
        st.warning(
            f"The issue assessment definition '{base_assess_id}' could not be loaded. " 
            "This may affect the functionality of this page. Please check your configuration."
        )
    
    # Document upload section
    section_header("Upload Document", 1)
    
    uploaded_file = st.file_uploader("Upload a document to analyze", type=["txt"])
    
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
        section_header("Configure Assessment", 2)
        
        # Simplified options
        with st.expander("Assessment Settings", expanded=True):
            # Model selection
            model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
            selected_model = st.selectbox("LLM Model", options=model_options, index=0)  
            
            # Detail level - only setting we're keeping
            detail_options = {
                "essential": "Focus only on the most significant issues",
                "standard": "Balanced analysis of problems and challenges",
                "comprehensive": "In-depth analysis of all potential issues"
            }
            detail_level = st.radio(
                "Detail Level",
                options=list(detail_options.keys()),
                format_func=lambda x: x.title(),
                index=1  # Default to standard
            )
            st.caption(detail_options[detail_level])
            
            # Add a divider for advanced settings
            st.divider()
            
            # Advanced settings section
            st.markdown("##### Advanced Settings")
            
            # Temperature
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
            
            # Chunk settings
            chunk_size = st.slider("Chunk Size", min_value=1000, max_value=15000, value=6000, step=1000)
            chunk_overlap = st.slider("Chunk Overlap", min_value=100, max_value=1000, value=300, step=100)
        
        # Process button
        col1, col2 = st.columns([3, 1])
        with col1:
            if not st.session_state.processing_started:
                # Disable button if assessment not available
                button_disabled = not assessment_available
                
                if st.button("🔍 Identify Issues", type="primary", use_container_width=True, disabled=button_disabled):
                    logger.info("Identify Issues button clicked")
                    
                    # Create assessment options - simplified
                    assessment_options = {
                        "detail_level": detail_level,
                        "minimum_severity": "low",  # Always include all severities
                        "focus_categories": []  # No category filtering
                    }
                    
                    # Create processing options
                    processing_options = {
                        "model": selected_model,
                        "temperature": temperature,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "user_options": assessment_options
                    }
                    
                    # Set the base assess assessment ID
                    selected_assessment_id = base_assess_id
                    
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
            if st.button("↺ Reset", use_container_width=True):
                logger.info("Reset button clicked")
                
                # Reset session state
                st.session_state.document_loaded = False
                st.session_state.processing_started = False
                st.session_state.processing_complete = False
                st.session_state.issues_result = None
                st.session_state.current_progress = 0.0
                st.session_state.progress_message = "Not started"
                
                logger.info("Session state reset")
                st.rerun()
    
    # Processing section
    if st.session_state.processing_started:
        section_header("Processing Status", 3)
        
        if not st.session_state.processing_complete:
            logger.info("Processing in progress, showing status...")
            
            # Use our enhanced detailed progress display
            render_detailed_progress(st.session_state)
            
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
        
       
        # Results section with enhanced display
        if st.session_state.processing_complete:
            section_header("Issue Assessment Results", 4)
            
            if st.session_state.issues_result:
                logger.info("Displaying issues assessment results")
                result = st.session_state.issues_result
                
                # Check for explicit error
                if "error" in result and result["error"]:
                    st.error(f"Error in assessment: {result['error']}")
                    with st.expander("Show Error Details", expanded=True):
                        st.json(result)
                else:
                    # Use our enhanced renderer
                    render_issues_result(result)
                    
                    # Add download options
                    st.markdown("### Download Options")
                    # ... rest of the download options code



            
                
                # Generate report markdown
                report_md = format_assessment_report(result, "assess")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download as markdown
                    download_md_button = st.download_button(
                        "⬇️ Download as Markdown",
                        data=report_md,
                        file_name=f"issues_report_{timestamp}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                
                with col2:
                    # Download as JSON
                    json_str = json.dumps(result, indent=2)
                    download_json_button = st.download_button(
                        "⬇️ Download Raw Data",
                        data=json_str,
                        file_name=f"issues_data_{timestamp}.json",
                        mime="application/json",
                        use_container_width=True
                    )
            else:
                error_msg = "Processing completed but no issues assessment was generated. Please check the logs for details."
                logger.error(error_msg)
                st.error(error_msg)
                
if __name__ == "__main__":
    try:
        logger.info("Starting 02_Issues.py main execution")
        main()
        logger.info("02_Issues.py execution completed successfully")
    except Exception as e:
        error_msg = f"Unhandled exception in main execution: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}")