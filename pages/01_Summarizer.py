# pages/01_Summarizer.py
import streamlit as st
import os
import asyncio
import logging
import json
import time
import uuid
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
    base_distill_id = "base_distill_summary_v1"  # Use your actual base distill ID
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

# Enhanced CSS styles with improved spacing, transitions, and visual hierarchy
st.markdown("""
<style>
    /* --- Main Layout --- */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white;
        padding: 0.5rem 0;
    }
    
    .subheader {
        font-size: 1.1rem;
        opacity: 0.8;
        margin-bottom: 2.5rem;
        font-weight: 300;
    }
    
    /* --- Section Headers with Numbers --- */
    .section-header {
        font-size: 1.6rem;
        font-weight: 600;
        margin: 2.5rem 0 1.5rem 0;
        display: flex;
        align-items: center;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .section-number {
        display: inline-flex;
        justify-content: center;
        align-items: center;
        width: 38px;
        height: 38px;
        background-color: #2196F3;
        color: white;
        border-radius: 50%;
        text-align: center;
        margin-right: 12px;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    
    /* --- Document Preview --- */
    .document-meta {
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .document-meta-label {
        font-size: 0.9rem;
        opacity: 0.7;
        margin-bottom: 0.3rem;
        font-weight: 500;
    }
    
    .document-meta-value {
        font-weight: 500;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    
    /* --- Progress Status --- */
    .progress-container {
        background-color: rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .progress-stage {
        margin-top: 0.8rem;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    .progress-message {
        opacity: 0.8;
        margin-top: 0.3rem;
        font-size: 0.95rem;
    }
    
    /* Custom progress bar */
    .stProgress > div > div > div > div {
        background-color: #2196F3;
        background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
        background-size: 40px 40px;
        animation: progress-bar-stripes 2s linear infinite;
    }
    
    @keyframes progress-bar-stripes {
        0% {background-position: 40px 0;}
        100% {background-position: 0 0;}
    }
    
    /* --- Enhanced Radio Buttons --- */
    .format-radio {
        margin: 1.5rem 0;
    }
    
    .format-option {
        background-color: rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        display: flex;
        align-items: center;
        transition: all 0.2s ease;
    }
    
    .format-option:hover {
        background-color: rgba(33, 150, 243, 0.1);
        border-color: rgba(33, 150, 243, 0.3);
        transform: translateY(-1px);
    }
    
    .format-option.selected {
        background-color: rgba(33, 150, 243, 0.15);
        border-color: #2196F3;
        box-shadow: 0 2px 6px rgba(33, 150, 243, 0.2);
    }
    
    .format-icon {
        margin-right: 0.75rem;
        opacity: 0.9;
        flex-shrink: 0;
    }
    
    .format-content {
        flex-grow: 1;
    }
    
    .format-name {
        font-weight: 600;
        margin-bottom: 0.2rem;
        font-size: 1rem;
        color: white;
    }
    
    .format-desc {
        font-size: 0.85rem;
        opacity: 0.7;
        line-height: 1.4;
    }
    
    /* --- Summary Results --- */
    .summary-container {
        margin: 1.5rem 0;
        animation: fadeIn 0.5s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .action-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background-color: rgba(33, 150, 243, 0.1);
        color: white;
        border: 1px solid rgba(33, 150, 243, 0.3);
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        font-size: 0.9rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        text-decoration: none;
    }
    
    .action-button:hover {
        background-color: rgba(33, 150, 243, 0.25);
        border-color: rgba(33, 150, 243, 0.5);
        transform: translateY(-1px);
    }
    
    .action-button i {
        margin-right: 0.5rem;
    }
    
    .action-container {
        margin: 1.5rem 0;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    
    /* --- Download/Copy Buttons --- */
    .download-container {
        margin-top: 1.5rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.7rem;
    }
    
    /* --- General Button Improvements --- */
    .stButton > button {
        font-weight: 500;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button[data-baseweb="button"] {
        background-color: #2196F3;
    }
    
    /* --- Statistics Display --- */
    .stats-container {
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 2rem;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stat-item {
        display: flex;
        flex-direction: column;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.7;
    }
    
    /* --- Copy Feedback --- */
    .copy-feedback {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background-color: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 1rem 2rem;
        border-radius: 6px;
        z-index: 1000;
        animation: fadeInOut 1.5s ease forwards;
    }
    
    @keyframes fadeInOut {
        0% { opacity: 0; }
        20% { opacity: 1; }
        80% { opacity: 1; }
        100% { opacity: 0; }
    }
    
    /* Hide our summarize-again button visually */
    div[data-testid="stButton"] button:has(div:contains("Summarize Again")) {
        display: none !important;
    }
    
    /* --- Responsive Adjustments --- */
    @media (max-width: 768px) {
        .stats-container {
            grid-template-columns: 1fr 1fr;
        }
    }
</style>

<!-- SVG Icons for buttons and formats -->
<svg style="display:none;">
    <symbol id="icon-copy" viewBox="0 0 24 24">
        <path d="M16 1H4C2.9 1 2 1.9 2 3V17H4V3H16V1ZM19 5H8C6.9 5 6 5.9 6 7V21C6 22.1 6.9 23 8 23H19C20.1 23 21 22.1 21 21V7C21 5.9 20.1 5 19 5ZM19 21H8V7H19V21Z"/>
    </symbol>
    <symbol id="icon-download" viewBox="0 0 24 24">
        <path d="M19 9H15V3H9V9H5L12 16L19 9ZM5 18V20H19V18H5Z"/>
    </symbol>
    <symbol id="icon-executive" viewBox="0 0 24 24">
        <path d="M19 3H5C3.9 3 3 3.9 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.9 20.1 3 19 3ZM9 17H7V10H9V17ZM13 17H11V7H13V17ZM17 17H15V13H17V17Z"/>
    </symbol>
    <symbol id="icon-comprehensive" viewBox="0 0 24 24">
        <path d="M3 13H5V11H3V13ZM3 17H5V15H3V17ZM3 9H5V7H3V9ZM7 13H21V11H7V13ZM7 17H21V15H7V17ZM7 7V9H21V7H7Z"/>
    </symbol>
    <symbol id="icon-bullets" viewBox="0 0 24 24">
        <path d="M4 10.5C3.17 10.5 2.5 11.17 2.5 12C2.5 12.83 3.17 13.5 4 13.5C4.83 13.5 5.5 12.83 5.5 12C5.5 11.17 4.83 10.5 4 10.5ZM4 4.5C3.17 4.5 2.5 5.17 2.5 6C2.5 6.83 3.17 7.5 4 7.5C4.83 7.5 5.5 6.83 5.5 6C5.5 5.17 4.83 4.5 4 4.5ZM4 16.5C3.17 16.5 2.5 17.18 2.5 18C2.5 18.82 3.18 19.5 4 19.5C4.82 19.5 5.5 18.82 5.5 18C5.5 17.18 4.83 16.5 4 16.5ZM7 19H21V17H7V19ZM7 13H21V11H7V13ZM7 5V7H21V5H7Z"/>
    </symbol>
    <symbol id="icon-narrative" viewBox="0 0 24 24">
        <path d="M4 6H2V20C2 21.1 2.9 22 4 22H18V20H4V6ZM20 2H8C6.9 2 6 2.9 6 4V16C6 17.1 6.9 18 8 18H20C21.1 18 22 17.1 22 16V4C22 2.9 21.1 2 20 2ZM19 11H9V9H19V11ZM15 15H9V13H15V15ZM19 7H9V5H19V7Z"/>
    </symbol>
    <symbol id="icon-summarize-again" viewBox="0 0 24 24">
        <path d="M17.65 6.35C16.2 4.9 14.21 4 12 4C7.58 4 4 7.58 4 12C4 16.42 7.58 20 12 20C15.73 20 18.84 17.45 19.73 14H17.65C16.83 16.33 14.61 18 12 18C8.69 18 6 15.31 6 12C6 8.69 8.69 6 12 6C13.66 6 15.14 6.69 16.22 7.78L13 11H20V4L17.65 6.35Z"/>
    </symbol>
</svg>
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
    
    if "show_copy_feedback" not in st.session_state:
        st.session_state.show_copy_feedback = False
    
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
    """Display the current pipeline status with enhanced visuals."""
    progress_value = float(st.session_state.get("current_progress", 0.0))
    
    # Progress container with better styling
    st.markdown('<div class="progress-container">', unsafe_allow_html=True)
    
    # Enhanced progress bar
    progress_bar = st.progress(progress_value)
    
    # Current stage and progress message
    current_stage = st.session_state.get("current_stage", "")
    progress_message = st.session_state.get("progress_message", "Waiting to start...")
    
    if current_stage:
        st.markdown(f'<div class="progress-stage">{current_stage.replace("_", " ").title()}</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div class="progress-message">{progress_message}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
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
    """Display a preview of the loaded document with enhanced styling."""
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
    """Enhanced radio buttons for format selection."""
    # Format options with descriptions and icons
    format_options = {
        "executive": {
            "desc": "Very concise overview focusing on key findings and decisions.",
            "icon": "#icon-executive"
        },
        "comprehensive": {
            "desc": "Detailed summary with supporting information and context.",
            "icon": "#icon-comprehensive"
        },
        "bullet_points": {
            "desc": "Key points organized in an easy-to-scan bullet list format.",
            "icon": "#icon-bullets"
        },
        "narrative": {
            "desc": "Flowing narrative that preserves the document's natural story.",
            "icon": "#icon-narrative"
        }
    }
    
    # Get current selection
    current_format = st.session_state.get("selected_format", "executive")
    
    # Display format selection header
    st.write("### Select Summary Format")
    
    # Use a standard radio component but visually hide it
    selected_format = st.radio(
        "Summary format",
        options=list(format_options.keys()),
        index=list(format_options.keys()).index(current_format),
        format_func=lambda x: x.replace("_", " ").title(),
        label_visibility="collapsed",
        horizontal=True
    )
    
    # Display enhanced radio options
    st.markdown('<div class="format-radio">', unsafe_allow_html=True)
    
    for format_key, format_data in format_options.items():
        # Determine if this format is selected
        is_selected = format_key == selected_format
        selected_class = "selected" if is_selected else ""
        
        # Format display name
        format_display = format_key.replace("_", " ").title()
        
        # Create styled radio option
        format_html = f"""
        <div class="format-option {selected_class}" onclick="document.getElementById('radio_{format_key}').click()">
            <svg class="format-icon" width="24" height="24">
                <use xlink:href="{format_data['icon']}"></use>
            </svg>
            <div class="format-content">
                <div class="format-name">{format_display}</div>
                <div class="format-desc">{format_data['desc']}</div>
            </div>
        </div>
        """
        
        st.markdown(format_html, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add JavaScript to handle clicking on our custom radio options
    # Note: We use a hidden Streamlit radio button for state management
    radio_js = """
    <script>
    // Create hidden radio buttons for each format
    document.addEventListener('DOMContentLoaded', function() {
        const formatKeys = ['executive', 'comprehensive', 'bullet_points', 'narrative'];
        formatKeys.forEach(key => {
            const radio = document.createElement('input');
            radio.type = 'radio';
            radio.id = 'radio_' + key;
            radio.style.display = 'none';
            radio.name = 'custom_format_radio';
            document.body.appendChild(radio);
        });
    });
    </script>
    """
    st.markdown(radio_js, unsafe_allow_html=True)
    
    # Update session state if changed
    if selected_format != current_format:
        logger.info(f"Format selection changed to: {selected_format}")
        st.session_state.selected_format = selected_format
    
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
    """Display the summary result with enhanced styling and additional functionality."""
    logger.info(f"Displaying summary result with format: {format_type}")
    
    try:
        # Generate HTML with our renderer
        logger.info("Generating HTML with summary renderer...")
        html_content = render_summary(result, format_type)
        
        # Display the summary in Streamlit with better container
        st.markdown('<div class="summary-container">', unsafe_allow_html=True)
        st.markdown(html_content, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add action buttons
        st.markdown('<div class="action-container">', unsafe_allow_html=True)
        
        # Get summary text and statistics
        summary_text = result.get("result", {}).get("summary", "")
        summary_stats = result.get("result", {}).get("statistics", {})
        
        # Generate a unique ID for this summary
        summary_id = f"summary_{uuid.uuid4().hex[:8]}"
        
        # Store the summary text in a hidden element
        st.markdown(f'<div id="{summary_id}" style="display:none;">{summary_text}</div>', unsafe_allow_html=True)
        
        # Copy button with the ID approach
        if summary_text:
            copy_button_html = f"""
            <button class="action-button" onclick="copyTextById('{summary_id}')">
                <svg width="16" height="16">
                    <use xlink:href="#icon-copy"></use>
                </svg>
                Copy Summary
            </button>
            """
            st.markdown(copy_button_html, unsafe_allow_html=True)
            
            # Add JavaScript for clipboard functionality
            clipboard_js = """
            <script>
            function copyTextById(elementId) {
                const text = document.getElementById(elementId).innerText;
                
                // Create temporary element
                const el = document.createElement('textarea');
                el.value = text;
                document.body.appendChild(el);
                el.select();
                document.execCommand('copy');
                document.body.removeChild(el);
                
                // Show feedback
                const feedback = document.createElement('div');
                feedback.className = 'copy-feedback';
                feedback.textContent = 'Summary copied to clipboard!';
                document.body.appendChild(feedback);
                
                // Remove feedback after animation
                setTimeout(() => {
                    document.body.removeChild(feedback);
                }, 1500);
            }
            </script>
            """
            st.markdown(clipboard_js, unsafe_allow_html=True)
        
        # Summarize Again button
        if summary_text:
            # Create a hidden button that will handle the actual functionality
            if st.button("Summarize Again", key="summarize-again-button"):
                logger.info("Summarize Again button clicked")
                
                # Create a new document from the summary
                temp_dir = AppPaths.get_temp_path("summaries")
                temp_dir.mkdir(exist_ok=True, parents=True)
                temp_file_path = temp_dir / "previous_summary.txt"
                
                with open(temp_file_path, "w", encoding="utf-8") as f:
                    f.write(summary_text)
                
                # Create document from file
                try:
                    document = Document.from_file(temp_file_path)
                    st.session_state.document = document
                    st.session_state.document_loaded = True
                    st.session_state.processing_started = False
                    st.session_state.processing_complete = False
                    st.rerun()
                except Exception as e:
                    logger.error(f"Error creating document from summary: {e}")
                    st.error("Could not create a new summary from the current one.")
            
            # Visual button that will click the hidden button
            summarize_again_html = f"""
            <button class="action-button" onclick="document.querySelector('button:has(div:contains(\"Summarize Again\"))').click()">
                <svg width="16" height="16">
                    <use xlink:href="#icon-summarize-again"></use>
                </svg>
                Summarize Again
            </button>
            """
            st.markdown(summarize_again_html, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display enhanced statistics if available
        if summary_stats:
            # Get key statistics
            original_words = summary_stats.get("original_word_count", 0)
            summary_words = summary_stats.get("summary_word_count", 0)
            topics_count = summary_stats.get("topics_covered", 0)
            
            # Calculate compression ratio
            compression_ratio = 0
            if original_words > 0 and summary_words > 0:
                compression_ratio = (summary_words / original_words) * 100
            
            # Display in a nicely formatted grid
            st.markdown('<div class="stats-container">', unsafe_allow_html=True)
            
            # Original word count
            st.markdown(f"""
            <div class="stat-item">
                <div class="stat-value">{original_words:,}</div>
                <div class="stat-label">Original Words</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Summary word count
            st.markdown(f"""
            <div class="stat-item">
                <div class="stat-value">{summary_words:,}</div>
                <div class="stat-label">Summary Words</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Compression ratio
            st.markdown(f"""
            <div class="stat-item">
                <div class="stat-value">{compression_ratio:.1f}%</div>
                <div class="stat-label">Compression Ratio</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Topics count
            if topics_count > 0:
                st.markdown(f"""
                <div class="stat-item">
                    <div class="stat-value">{topics_count}</div>
                    <div class="stat-label">Topics Covered</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Download options with improved styling
        st.markdown('<div class="download-container">', unsafe_allow_html=True)
        
        # Generate report markdown
        report_md = format_assessment_report(result, "distill")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download as markdown with icon
            download_md_button = st.download_button(
                "⬇️ Download as Markdown",
                data=report_md,
                file_name=f"summary_{timestamp}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with col2:
            # Download as JSON with icon
            json_str = json.dumps(result, indent=2)
            download_json_button = st.download_button(
                "⬇️ Download Raw Data",
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
    
    # Header with enhanced styling
    st.markdown('<div class="main-header">📄 Document Summarizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Transform documents into clear, readable summaries with our AI system</div>', unsafe_allow_html=True)
    
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
        
        # Format selection with enhanced visuals
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
                if st.button("✨ Generate Summary", type="primary", use_container_width=True):
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
            if st.button("↺ Reset", use_container_width=True):
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
            
            # Show enhanced progress tracking
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
        
        # Results section with enhanced display
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