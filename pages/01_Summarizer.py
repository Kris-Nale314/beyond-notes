# pages/05_Summarizer.py
import streamlit as st
import os
import asyncio
import logging
import json
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("beyond-notes-summarizer")

# Import components
try:
    from core.models.document import Document
    from core.orchestrator import Orchestrator
    from assessments.loader import AssessmentLoader
    from utils.paths import AppPaths
    from utils.formatting import format_assessment_report
except ImportError as e:
    logger.error(f"Error importing components: {e}", exc_info=True)
    st.error(f"Failed to load application components: {e}")
    st.stop()

# Ensure directories exist
try:
    AppPaths.ensure_dirs()
except Exception as e:
    logger.error(f"Failed to ensure directories: {e}", exc_info=True)
    st.error(f"Failed to setup application directories: {e}")
    st.stop()

# Page config
st.set_page_config(
    page_title="Beyond Notes - Document Summarizer",
    page_icon="📝",
    layout="wide",
)

# CSS styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .section-number {
        display: inline-block;
        width: 28px;
        height: 28px;
        background-color: #2196F3;
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 28px;
        margin-right: 8px;
    }
    .stage-progress {
        padding: 0.5rem; 
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
        background-color: rgba(255, 255, 255, 0.03);
    }
    .completed-stage { border-left: 3px solid #4CAF50; }
    .running-stage { border-left: 3px solid #2196F3; }
    .failed-stage { border-left: 3px solid #F44336; }
</style>
""", unsafe_allow_html=True)

def initialize_page():
    """Initialize the session state variables."""
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

def load_document(file_object):
    """Load document from uploaded file."""
    try:
        # Save uploaded file temporarily
        temp_dir = AppPaths.get_temp_path("uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file_path = temp_dir / file_object.name
        
        with open(temp_file_path, "wb") as f:
            f.write(file_object.getvalue())
        
        # Create document from file
        document = Document.from_file(temp_file_path)
        logger.info(f"Document loaded: {document.filename}, {document.word_count} words")
        
        return document, temp_file_path
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        logger.error(f"Error loading document: {e}", exc_info=True)
        return None, None

def progress_callback(progress, data):
    """Callback for progress updates from the orchestrator."""
    if isinstance(data, dict):
        st.session_state.current_progress = float(progress)
        st.session_state.progress_message = data.get("message", "Processing...")
        st.session_state.current_stage = data.get("current_stage")
        st.session_state.stages_info = data.get("stages", {})
    else:
        # Simple string message
        st.session_state.current_progress = float(progress)
        st.session_state.progress_message = data

def render_pipeline_status():
    """Display the current pipeline status."""
    progress_value = float(st.session_state.get("current_progress", 0.0))
    st.progress(progress_value)
    
    # Current stage and progress message
    current_stage = st.session_state.get("current_stage")
    if current_stage:
        st.markdown(f"**Current Stage:** {current_stage.replace('_', ' ').title()}")
    
    st.caption(st.session_state.get("progress_message", "Waiting to start..."))
    
    # Show detailed stage information if available
    stages_info = st.session_state.get("stages_info", {})
    if stages_info:
        st.markdown("#### Stage Progress")
        
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
    st.markdown("### Document Preview")
    st.markdown(f"**Filename:** `{document.filename}`")
    st.markdown(f"**Word Count:** `{document.word_count}`")
    
    preview_length = min(1000, len(document.text))
    preview_text = document.text[:preview_length]
    if len(document.text) > preview_length:
        preview_text += "..."
    
    with st.expander("Document Content Preview", expanded=False):
        st.text_area("", preview_text, height=200, disabled=True)

async def process_document(document, assessment_id, options):
    """Process document using the orchestrator."""
    try:
        # Get API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key not found.")
            return None
        
        # Initialize orchestrator
        orchestrator = Orchestrator(assessment_id, options=options, api_key=api_key)
        
        # Set progress callback
        orchestrator.set_progress_callback(progress_callback)
        
        # Process document
        logger.info(f"Processing started for assessment {assessment_id}")
        result = await orchestrator.process_document(document)
        logger.info(f"Processing completed for assessment {assessment_id}")
        
        # Update state
        st.session_state.summary_result = result
        st.session_state.processing_complete = True
        
        return result
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        logger.error(f"Error processing document: {e}", exc_info=True)
        st.session_state.processing_complete = True
        return None

def main():
    """Main function for the summarizer page."""
    # Initialize page
    initialize_page()
    
    # Header
    st.markdown('<div class="main-header">📝 Document Summarizer</div>', unsafe_allow_html=True)
    st.markdown("""
    Transform documents into concise, structured summaries using our multi-agent AI system. 
    Configure how your summary is generated using the options in the sidebar.
    """)
    
    # Info expander
    with st.expander("How the summarizer works", expanded=False):
        st.markdown("""
        ### Beyond Notes Multi-Agent Summarization
        
        Our document summarizer uses five specialized AI agents working together:
        
        1. **Planner Agent** analyzes document structure and creates a summarization strategy
        2. **Extractor Agent** identifies important information from each section
        3. **Aggregator Agent** combines findings and removes duplicates
        4. **Evaluator Agent** assesses accuracy and balance
        5. **Formatter Agent** structures everything into a readable summary
        """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("### Summary Options")
        
        # Hard-coded format options from the distill JSON
        st.markdown("**Summary Format:**")
        format_options = {
            "executive": "Very concise overview of main points (5-10% of original)",
            "comprehensive": "Detailed summary with supporting information (15-25% of original)",
            "bullet_points": "Key points in bullet list format (10-15 key points)",
            "narrative": "Flowing narrative that preserves the document's story (10-20% of original)"
        }
        
        format_list = list(format_options.keys())
        format_index = format_list.index(st.session_state.selected_format) if st.session_state.selected_format in format_list else 0
        
        selected_format = st.selectbox(
            "Select format",
            options=format_list,
            index=format_index,
            format_func=lambda x: f"{x.title()} - {format_options.get(x, '').split('(')[0].strip()}",
            help="Determines the style and structure of your summary"
        )
        
        st.session_state.selected_format = selected_format
        
        # Hard-coded length options
        st.markdown("**Summary Length:**")
        length_options = {
            "brief": "Shortest possible summary capturing only essentials",
            "standard": "Standard length appropriate for the format",
            "detailed": "Longer summary with more supporting details"
        }
        
        selected_length = st.selectbox(
            "Select length",
            options=list(length_options.keys()),
            format_func=lambda x: f"{x.title()} - {length_options.get(x)}",
            help="Controls the length of your summary"
        )
        
        # Hard-coded structure options
        st.markdown("**Organization Structure:**")
        structure_options = {
            "topic_based": "Organized by main topics discussed",
            "chronological": "Follows the timeline of the document",
            "speaker_based": "Organized by participants' contributions",
            "decision_focused": "Emphasizes decisions and agreements"
        }
        
        selected_structure = st.selectbox(
            "Select structure",
            options=list(structure_options.keys()),
            format_func=lambda x: f"{x.replace('_', ' ').title()} - {structure_options.get(x)}",
            help="Determines how information is organized in the summary"
        )
        
        # Include quotes option
        include_quotes = st.checkbox(
            "Include Key Quotes",
            value=False,
            help="Include direct quotes from the document in the summary"
        )
        
        # Focus areas multi-select
        st.markdown("**Focus Areas (Optional):**")
        focus_options = [
            "decisions", "action_items", "findings", "background", 
            "methodology", "results", "recommendations", "discussions", 
            "problems", "solutions"
        ]
        
        selected_focus = st.multiselect(
            "Select areas to emphasize",
            options=focus_options,
            default=[],
            help="Optionally focus on specific aspects (leave empty for balanced coverage)"
        )
        
        # Model settings
        st.markdown("### Model Settings")
        model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        selected_model = st.selectbox("LLM Model", options=model_options)
        
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1,
                               help="Lower values: more factual, Higher values: more creative")
        
        # Advanced processing settings
        with st.expander("Advanced Processing Options", expanded=False):
            chunk_size = st.slider("Chunk Size", min_value=1000, max_value=15000, value=8000, step=1000,
                                 help="Size of text chunks processed by the LLM. Larger chunks provide more context.")
            chunk_overlap = st.slider("Chunk Overlap", min_value=100, max_value=1000, value=300, step=100,
                                    help="Amount of text overlap between chunks to maintain context.")
    
    # Main content area - Document upload section
    st.markdown('<div class="section-header"><span class="section-number">1</span>Upload Document</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a document to summarize", type=["txt", "md", "pdf", "docx"])
    
    if uploaded_file:
        # Load document
        document, file_path = load_document(uploaded_file)
        if document:
            st.session_state.document = document
            st.session_state.document_loaded = True
            display_document_preview(document)
    
    # Options and processing
    if st.session_state.document_loaded:
        st.markdown('<div class="section-header"><span class="section-number">2</span>Generate Summary</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if not st.session_state.processing_started:
                if st.button("Generate Summary", type="primary", use_container_width=True):
                    # Create summary options
                    summary_options = {
                        "format": selected_format,
                        "length": selected_length,
                        "structure_preference": selected_structure,
                        "include_quotes": include_quotes,
                        "focus_areas": selected_focus
                    }
                    
                    # Create processing options
                    processing_options = {
                        "model": selected_model,
                        "temperature": temperature,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "user_options": summary_options
                    }
                    
                    # Always use base_distill_summary_v1 as assessment ID for summaries
                    selected_assessment_id = "base_distill_summary_v1"
                    
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
                # Reset session state
                st.session_state.document_loaded = False
                st.session_state.processing_started = False
                st.session_state.processing_complete = False
                st.session_state.summary_result = None
                st.session_state.current_progress = 0.0
                st.session_state.progress_message = "Not started"
                
                st.rerun()
    
    # Results section
    if st.session_state.processing_started:
        st.markdown('<div class="section-header"><span class="section-number">3</span>Processing Status</div>', unsafe_allow_html=True)
        
        if not st.session_state.processing_complete:
            # Show progress
            render_pipeline_status()
            
            # Process the document
            document_to_process = st.session_state.document
            assessment_id = st.session_state.selected_assessment_id
            options = st.session_state.processing_options
            
            processing_placeholder = st.empty()
            processing_placeholder.info("Processing document...")
            
            if 'is_processing' not in st.session_state:
                st.session_state.is_processing = True
                try:
                    # Run the async processing
                    asyncio.run(process_document(document_to_process, assessment_id, options))
                finally:
                    # Reset the processing flag
                    if 'is_processing' in st.session_state:
                        del st.session_state.is_processing
                    st.rerun()
        
        # Show results if processing is complete
        if st.session_state.processing_complete:
            st.markdown('<div class="section-header"><span class="section-number">4</span>Summary Results</div>', unsafe_allow_html=True)
            
            if st.session_state.summary_result:
                result = st.session_state.summary_result
                
                # Extract data
                formatted_data = result.get("result", {})
                
                # Create tabs for different views
                tabs = st.tabs(["Summary", "Details", "Statistics"])
                
                # Summary Tab
                with tabs[0]:
                    st.markdown(f"## Summary ({selected_format.title()} Format)")
                    
                    # Show the options used
                    with st.expander("Summary options used", expanded=False):
                        st.markdown("**Format:** " + selected_format.title())
                        st.markdown("**Length:** " + selected_length.title())
                        st.markdown("**Structure:** " + selected_structure.replace("_", " ").title())
                        st.markdown("**Include Quotes:** " + ("Yes" if include_quotes else "No"))
                        if selected_focus:
                            st.markdown("**Focus Areas:** " + ", ".join(selected_focus))
                    
                    # Get summary text
                    summary_text = formatted_data.get("summary", "No summary text found.")
                    
                    # Display summary in a nice container
                    with st.container(border=True):
                        st.markdown(summary_text)
                    
                    # Try to display as markdown format if available
                    try:
                        report_md = format_assessment_report(result, "distill")
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        st.download_button(
                            "Download Summary",
                            data=report_md,
                            file_name=f"summary_{timestamp}.md",
                            mime="text/markdown"
                        )
                    except Exception as e:
                        logger.error(f"Error formatting report: {e}", exc_info=True)
                
                # Details Tab
                with tabs[1]:
                    # Display topics if available
                    topics = formatted_data.get("topics", [])
                    if topics:
                        st.markdown("### Topics")
                        for topic in topics:
                            with st.expander(topic.get("topic", "Topic")):
                                # Display key points
                                if "key_points" in topic:
                                    for point in topic["key_points"]:
                                        st.markdown(f"- {point}")
                                
                                # Display details
                                if "details" in topic:
                                    st.markdown(topic["details"])
                
                # Statistics Tab
                with tabs[2]:
                    st.markdown("### Statistics")
                    
                    metadata = result.get("metadata", {})
                    stats = formatted_data.get("statistics", result.get("statistics", {}))
                    
                    # Display basic statistics
                    col1, col2, col3 = st.columns(3)
                    
                    original_words = stats.get("original_word_count", metadata.get("document_info", {}).get("word_count", 0))
                    summary_words = stats.get("summary_word_count", 0)
                    
                    col1.metric("Original Words", f"{original_words:,}")
                    col2.metric("Summary Words", f"{summary_words:,}")
                    
                    # Calculate compression ratio if both word counts are available
                    if original_words > 0 and summary_words > 0:
                        compression = (summary_words / original_words) * 100
                        col3.metric("Compression Ratio", f"{compression:.1f}%")
            else:
                st.error("Processing completed but no summary was generated.")

if __name__ == "__main__":
    main()