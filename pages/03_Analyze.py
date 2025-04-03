# pages/02_Assess.py
import streamlit as st
import os
import asyncio
import time
import json
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("beyond-notes-assess")

# Import components
from core.models.document import Document
from core.orchestrator import Orchestrator
from assessments.loader import AssessmentLoader
from utils.paths import AppPaths
from utils.formatting import format_assessment_report

# Ensure directories exist
AppPaths.ensure_dirs()

# Page config
st.set_page_config(
    page_title="Beyond Notes - Document Assessment",
    page_icon="📝",
    layout="wide",
)

# Define accent colors
PRIMARY_COLOR = "#4CAF50"  # Green
SECONDARY_COLOR = "#2196F3"  # Blue
ACCENT_COLOR = "#FF9800"  # Orange
ERROR_COLOR = "#F44336"  # Red

# CSS to customize the appearance
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
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .accent-border {
        border-left: 4px solid #FF9800;
        padding-left: 1rem;
    }
    .progress-container {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    .results-container {
        margin-top: 2rem;
    }
    .stage-progress {
        margin-bottom: 0.5rem;
        padding: 0.5rem;
        border-radius: 0.3rem;
        background-color: rgba(255, 255, 255, 0.05);
    }
    .completed-stage {
        border-left: 4px solid #4CAF50;
    }
    .running-stage {
        border-left: 4px solid #2196F3;
    }
    .failed-stage {
        border-left: 4px solid #F44336;
    }
</style>
""", unsafe_allow_html=True)

def initialize_assessment_page():
    """Initialize the session state variables for the assessment page."""
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
    
    if "assessment_result" not in st.session_state:
        st.session_state.assessment_result = None
    
    # Set default assessment type from main app if available
    if "default_assessment" in st.session_state and "selected_assessment_type" not in st.session_state:
        st.session_state.selected_assessment_type = st.session_state.default_assessment

def load_assessment_configs():
    """Load assessment configurations."""
    try:
        # Load assessment configurations
        assessment_loader = AssessmentLoader()
        configs = assessment_loader.get_assessment_configs_list()
        if not configs:
            st.error("No assessment configurations found. Please check your setup.")
            return None
        return configs
    except Exception as e:
        st.error(f"Error loading assessment configurations: {str(e)}")
        logger.error(f"Error loading assessment configurations: {e}", exc_info=True)
        return None

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
        
        # Log successful document loading
        logger.info(f"Successfully loaded document: {document.filename}, {document.word_count} words")
        
        return document, temp_file_path
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        logger.error(f"Error loading document: {e}", exc_info=True)
        return None, None

def progress_callback(progress, data):
    """Callback function for progress updates from the orchestrator."""
    if isinstance(data, dict):
        st.session_state.current_progress = progress
        st.session_state.progress_message = data.get("message", "Processing...")
        st.session_state.current_stage = data.get("current_stage")
        st.session_state.stages_info = data.get("stages", {})
    else:
        # Simple string message
        st.session_state.current_progress = progress
        st.session_state.progress_message = data

def render_pipeline_status():
    """Render the current pipeline status."""
    progress_bar = st.progress(st.session_state.current_progress)
    
    # Current stage and progress message
    if hasattr(st.session_state, "current_stage") and st.session_state.current_stage:
        current_stage = st.session_state.current_stage.replace("_", " ").title()
        st.markdown(f"**Current Stage:** {current_stage}")
    
    st.caption(st.session_state.progress_message)
    
    # Show detailed stage information if available
    if hasattr(st.session_state, "stages_info") and st.session_state.stages_info:
        st.markdown("#### Stage Progress")
        
        for stage_name, stage_info in st.session_state.stages_info.items():
            status = stage_info.get("status", "not_started")
            progress = stage_info.get("progress", 0)
            message = stage_info.get("message", "")
            
            # Determine stage class based on status
            if status == "completed":
                stage_class = "completed-stage"
                emoji = "✅"
            elif status == "running":
                stage_class = "running-stage"
                emoji = "⏳"
            elif status == "failed":
                stage_class = "failed-stage"
                emoji = "❌"
            else:
                stage_class = ""
                emoji = "⏱️"
            
            display_name = stage_name.replace("_", " ").title()
            progress_pct = f"{int(progress * 100)}%" if progress > 0 else ""
            
            # Display stage progress
            st.markdown(f"""
            <div class="stage-progress {stage_class}">
                <strong>{emoji} {display_name}</strong> {progress_pct}<br>
                <small>{message}</small>
            </div>
            """, unsafe_allow_html=True)

def display_document_preview(document):
    """Display a preview of the loaded document."""
    st.markdown("### Document Preview")
    
    # Display document metadata
    st.markdown(f"**Filename:** {document.filename}")
    st.markdown(f"**Word Count:** {document.word_count}")
    
    # Display a preview of the document text
    preview_length = min(1000, len(document.text))
    preview_text = document.text[:preview_length]
    if len(document.text) > preview_length:
        preview_text += "..."
    
    st.text_area("Document Content Preview", preview_text, height=200)

async def process_document(document, assessment_id, options):
    """Process document using the orchestrator."""
    try:
        # Get API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            return None
        
        # Initialize orchestrator
        orchestrator = Orchestrator(assessment_id, options, api_key)
        
        # Set progress callback
        orchestrator.set_progress_callback(progress_callback)
        
        # Process document
        result = await orchestrator.process_document(document)
        
        # Set result in session state
        st.session_state.assessment_result = result
        st.session_state.processing_complete = True
        
        return result
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        logger.error(f"Error processing document: {e}", exc_info=True)
        st.session_state.processing_complete = True
        return None

def display_assessment_result(result):
    """Display the assessment result."""
    if not result:
        st.error("No assessment result available.")
        return
    
    try:
        # Get metadata from result
        metadata = result.get("metadata", {})
        assessment_type = metadata.get("assessment_type", "unknown")
        assessment_id = metadata.get("assessment_id", "unknown")
        processing_time = metadata.get("processing_time_seconds", 0)
        
        # Create a tab view for different result views
        tabs = st.tabs(["Formatted Result", "Raw Data", "Performance Stats"])
        
        # Formatted Result Tab
        with tabs[0]:
            try:
                # Try to format the result as markdown
                report_md = format_assessment_report(result, assessment_type)
                
                # Create a download button for the markdown
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                md_path = AppPaths.get_assessment_output_dir(assessment_type) / f"{assessment_type}_report_{timestamp}.md"
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(report_md)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("### Assessment Report")
                with col2:
                    with open(md_path, "rb") as f:
                        st.download_button(
                            label="Download Report",
                            data=f,
                            file_name=f"{assessment_type}_report_{timestamp}.md",
                            mime="text/markdown"
                        )
                
                # Display the formatted result
                st.markdown(report_md)
                
            except Exception as e:
                st.error(f"Error formatting result: {str(e)}")
                st.json(result.get("result", {}))
        
        # Raw Data Tab
        with tabs[1]:
            st.markdown("### Raw Assessment Data")
            
            # Create a download button for the JSON
            json_path = AppPaths.get_assessment_output_dir(assessment_type) / f"{assessment_type}_result_{timestamp}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            with st.expander("Download Raw Data", expanded=False):
                with open(json_path, "rb") as f:
                    st.download_button(
                        label="Download JSON",
                        data=f,
                        file_name=f"{assessment_type}_result_{timestamp}.json",
                        mime="application/json"
                    )
            
            # Display the JSON result
            st.json(result)
        
        # Performance Stats Tab
        with tabs[2]:
            st.markdown("### Performance Statistics")
            
            # Display processing time
            st.metric("Total Processing Time", f"{processing_time:.2f} seconds")
            
            # Get statistics from result
            statistics = result.get("statistics", {})
            
            # Display token usage
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Tokens", statistics.get("total_tokens", 0))
            with col2:
                st.metric("LLM API Calls", statistics.get("total_llm_calls", 0))
            with col3:
                st.metric("Document Chunks", statistics.get("total_chunks", 0))
            
            # Display stage durations
            st.markdown("#### Stage Durations")
            stage_durations = statistics.get("stage_durations", {})
            for stage, duration in stage_durations.items():
                st.metric(stage.replace("_", " ").title(), f"{duration:.2f} seconds")
            
            # Display token usage per stage
            st.markdown("#### Token Usage by Stage")
            stage_tokens = statistics.get("stage_tokens", {})
            for stage, tokens in stage_tokens.items():
                st.metric(stage.replace("_", " ").title(), tokens)
    
    except Exception as e:
        st.error(f"Error displaying assessment result: {str(e)}")
        logger.error(f"Error displaying assessment result: {e}", exc_info=True)

def main():
    """Main function for the assessment page."""
    # Initialize page
    initialize_assessment_page()
    
    # Header
    st.markdown('<div class="main-header">Document Assessment</div>', unsafe_allow_html=True)
    st.markdown("Upload a document and analyze it using Beyond Notes' specialized AI agents.")
    
    # Load assessment configurations
    configs = load_assessment_configs()
    if not configs:
        return
    
    # Sidebar - Configuration
    with st.sidebar:
        st.markdown("### Assessment Configuration")
        
        # Group configs by assessment type
        assessment_types = {}
        for cfg in configs:
            if not cfg.get("is_template", False):  # Filter out templates for now
                a_type = cfg.get("assessment_type", "unknown")
                if a_type not in assessment_types:
                    assessment_types[a_type] = []
                assessment_types[a_type].append(cfg)
        
        # Let user select assessment type first
        selected_type = st.selectbox(
            "Assessment Type",
            options=list(assessment_types.keys()),
            index=list(assessment_types.keys()).index(st.session_state.get("selected_assessment_type", list(assessment_types.keys())[0])) if "selected_assessment_type" in st.session_state else 0,
            format_func=lambda x: {
                "distill": "📝 Summarization",
                "extract": "📋 Action Items",
                "assess": "⚠️ Issues & Risks",
                "analyze": "📊 Framework Analysis"
            }.get(x, x.title())
        )
        
        # Save selection to session state
        st.session_state.selected_assessment_type = selected_type
        
        # Then show available configs for that type
        type_configs = assessment_types.get(selected_type, [])
        if type_configs:
            type_options = {cfg["id"]: f"{cfg['display_name']}" for cfg in type_configs}
            selected_assessment_id = st.selectbox(
                "Configuration", 
                options=list(type_options.keys()),
                format_func=lambda x: type_options.get(x, x)
            )
            
            # Show description of selected assessment
            selected_config = next((cfg for cfg in configs if cfg["id"] == selected_assessment_id), None)
            if selected_config:
                st.info(selected_config.get("description", "No description available"))
        else:
            st.warning(f"No configurations found for type '{selected_type}'")
            selected_assessment_id = None
        
        # Model selection
        st.markdown("### Model Configuration")
        model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        selected_model = st.selectbox("LLM Model", options=model_options)
        
        # Temperature
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
        
        # Chunking options
        st.markdown("### Document Processing")
        chunk_size = st.slider("Chunk Size", min_value=1000, max_value=15000, value=8000, step=1000)
        chunk_overlap = st.slider("Chunk Overlap", min_value=100, max_value=1000, value=300, step=100)
        
        # Create processing options
        processing_options = {
            "model": selected_model,
            "temperature": temperature,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
    
    # Main content
    # Document upload panel
    st.markdown('<div class="section-header">Document Input</div>', unsafe_allow_html=True)
    
    # Document upload
    upload_container = st.container()
    with upload_container:
        uploaded_file = st.file_uploader("Upload a document", type=["txt", "md", "pdf", "docx"])
        
        if uploaded_file:
            # Load document
            document, file_path = load_document(uploaded_file)
            if document:
                st.session_state.document = document
                st.session_state.document_loaded = True
                display_document_preview(document)
        
        # Sample document option
        if not uploaded_file:
            st.markdown("#### Or use a sample document")
            # Add sample document selection here
    
    # Processing panel
    if st.session_state.document_loaded:
        st.markdown('<div class="section-header">Process Document</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if selected_assessment_id:
                process_desc = f"Process with {selected_type.title()} Assessment"
            else:
                process_desc = "Select an assessment first"
            
            if not st.session_state.processing_started:
                if st.button(process_desc, disabled=not selected_assessment_id, type="primary", use_container_width=True):
                    # Mark as processing started
                    st.session_state.processing_started = True
                    st.session_state.processing_complete = False
                    st.session_state.current_progress = 0.0
                    st.session_state.progress_message = "Starting..."
                    st.session_state.assessment_result = None
                    
                    # Rerun to update UI
                    st.rerun()
        
        with col2:
            if st.button("Reset", use_container_width=True):
                # Reset session state
                st.session_state.document_loaded = False
                st.session_state.processing_started = False
                st.session_state.processing_complete = False
                st.session_state.current_progress = 0.0
                st.session_state.progress_message = "Not started"
                st.session_state.assessment_result = None
                
                # Rerun to update UI
                st.rerun()
    
    # Progress tracking
    if st.session_state.processing_started and not st.session_state.processing_complete:
        st.markdown('<div class="progress-container"></div>', unsafe_allow_html=True)
        render_pipeline_status()
        
        # Process the document (async)
        if "document" in st.session_state and selected_assessment_id:
            process_placeholder = st.empty()
            process_placeholder.info("Processing document...")
            
            # Start async processing
            result = asyncio.run(process_document(
                st.session_state.document,
                selected_assessment_id,
                processing_options
            ))
            
            process_placeholder.empty()
            
            # Rerun to update UI with results
            st.rerun()
    
    # Display results
    if st.session_state.processing_complete and st.session_state.assessment_result:
        st.markdown('<div class="results-container"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Assessment Results</div>', unsafe_allow_html=True)
        display_assessment_result(st.session_state.assessment_result)

if __name__ == "__main__":
    main()