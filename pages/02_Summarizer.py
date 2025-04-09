# pages/01_Summarizer.py
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
    from core.orchestrator import Orchestrator
    from assessments.loader import AssessmentLoader
    from utils.paths import AppPaths
    # Import our new UI components
    from utils.ui.styles import get_base_styles, get_issues_styles
    from utils.ui.components import page_header, section_header, display_document_preview
    from utils.ui.progress import render_detailed_progress
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
distill_configs = []
base_distill_id = "base_distill_summary_v1"
assessment_available = False

try:
    logger.info("Loading assessment configurations...")
    loader = AssessmentLoader()
    configs = loader.get_assessment_configs_list()
    
    # Log all available configs for debugging
    config_ids = [c.get("assessment_id") for c in configs]
    logger.info(f"All available config IDs: {config_ids}")
    
    # Check for base distill assessment directly
    base_distill_config = loader.load_config(base_distill_id)
    if base_distill_config:
        logger.info(f"✅ Successfully loaded base distill config '{base_distill_id}'")
        assessment_available = True
        # Get type from loaded config
        logger.info(f"Config assessment_type: {base_distill_config.get('assessment_type')}")
    else:
        logger.warning(f"⚠️ Could not load base distill config '{base_distill_id}'")
        
except Exception as e:
    error_msg = f"Error checking configurations: {e}"
    logger.error(error_msg, exc_info=True)
    # Continue to show UI with error messages

# Page config
st.set_page_config(
    page_title="Beyond Notes - Document Summarizer",
    page_icon="📝",
    layout="wide",
)

# Apply our shared styles
st.markdown(get_base_styles(), unsafe_allow_html=True)

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

def save_summary_output(result, filename_base):
    """
    Save summary output to the distill output folder.
    
    Args:
        result: The summary result dictionary
        filename_base: Base name for the output file
    
    Returns:
        Tuple of (markdown_path, json_path) with the saved file paths
    """
    try:
        # Ensure distill output directory exists
        output_dir = AppPaths.get_assessment_output_dir("distill")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate markdown report
        report_md = format_assessment_report(result, "distill")
        md_filename = f"distill_report_{filename_base}_{timestamp}.md"
        md_path = output_dir / md_filename
        
        # Save markdown report
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        logger.info(f"Saved markdown summary to: {md_path}")
        
        # Save JSON data
        json_filename = f"distill_data_{filename_base}_{timestamp}.json"
        json_path = output_dir / json_filename
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved JSON summary to: {json_path}")
        
        return (md_path, json_path)
    except Exception as e:
        logger.error(f"Error saving summary output: {e}")
        return (None, None)

def display_summary_result(result):
    """
    Display the summary result with proper data extraction from complex nested structures.
    
    Args:
        result: The summary result dictionary
    """
    # Log keys for debugging
    logger.info(f"Result top-level keys: {list(result.keys())}")
    
    # Find all key_points across different possible locations
    key_points = []
    
    # Check extracted_info first (most likely location based on your data)
    if "extracted_info" in result and "key_points" in result["extracted_info"]:
        key_points = result["extracted_info"]["key_points"]
        logger.info(f"Found {len(key_points)} key points in extracted_info")
    
    # Check other possible locations if nothing found
    if not key_points:
        if "key_points" in result:
            key_points = result["key_points"]
        elif "result" in result and "key_points" in result["result"]:
            key_points = result["result"]["key_points"]
    
    # Get executive summary if available
    executive_summary = None
    if "overall_assessment" in result and "executive_summary" in result["overall_assessment"]:
        executive_summary = result["overall_assessment"]["executive_summary"]
        logger.info("Found executive summary in overall_assessment")
    
    # Display content
    if executive_summary:
        st.markdown("## Executive Summary")
        st.markdown(executive_summary)
        st.markdown("---")
    
    if key_points:
        # Group key points by topic
        topics = {}
        for point in key_points:
            if isinstance(point, dict):
                topic = point.get("topic", "General")
                if topic not in topics:
                    topics[topic] = []
                topics[topic].append(point)
        
        # Display key points by topic
        st.markdown("## Key Points by Topic")
        
        for topic, points in sorted(topics.items()):
            with st.expander(f"Topic: {topic} ({len(points)} points)", expanded=True):
                for point in points:
                    # Get importance and create colored badge
                    importance = point.get("importance", "Medium")
                    color_map = {"High": "#F44336", "Medium": "#2196F3", "Low": "#4CAF50"}
                    color = color_map.get(importance, "#9E9E9E")
                    
                    # Create a container for each point with styled HTML
                    st.markdown(f"""
                    <div style="margin-bottom: 16px; padding: 12px; border-left: 4px solid {color}; 
                         background-color: rgba(0,0,0,0.1); border-radius: 4px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <span style="font-weight: bold; font-size: 1.05rem;">{point.get('text', '')}</span>
                            <span style="background-color: {color}; color: white; 
                                 padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; 
                                 display: inline-block; margin-left: 10px; white-space: nowrap;">
                                 {importance}
                            </span>
                        </div>
                        <div style="display: flex; margin-top: 8px; font-size: 0.9rem;">
                            <span style="opacity: 0.7; margin-right: 12px;">
                                <strong>Type:</strong> {point.get("point_type", "N/A")}
                            </span>
                            <span style="opacity: 0.7;">
                                <strong>Relevance:</strong> {point.get("evaluated_relevance_score", "N/A")}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        # If no key points found, show the raw JSON and warn the user
        st.warning("Could not find key points in the result structure. Showing raw data for debugging.")
        
        # Show raw data in an expander for debugging
        with st.expander("View Raw Result", expanded=True):
            st.json(result)
            
    # Display statistics and additional information
    display_summary_statistics(result)
    
    # Show strategic recommendations if available
    if "overall_assessment" in result and "strategic_recommendations" in result["overall_assessment"]:
        recommendations = result["overall_assessment"]["strategic_recommendations"]
        if recommendations:
            st.markdown("## Strategic Recommendations")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")

def display_summary_statistics(result):
    """Display summary statistics with improved data extraction."""
    # Find statistics in multiple possible locations
    statistics = None
    
    # Check common paths for statistics
    if "statistics" in result:
        statistics = result["statistics"] 
    elif "result" in result and "statistics" in result["result"]:
        statistics = result["result"]["statistics"]
    
    # Get metadata for additional stats
    metadata = result.get("metadata", {})
    if not metadata and "result" in result:
        metadata = result.get("result", {}).get("metadata", {})
    
    # Display stats if found
    if statistics or metadata:
        st.markdown("## Document Statistics")
        
        # Create metrics list
        metrics = []
        
        # Word count from metadata (more reliable)
        word_count = metadata.get("word_count", 0)
        if not word_count:
            word_count = metadata.get("document_info", {}).get("word_count", 0)
        
        # Get processing time
        processing_time = metadata.get("processing_time", metadata.get("processing_time_seconds", 0))
        
        # Get token usage if available
        total_tokens = statistics.get("total_tokens", 0)
        
        # Add metrics
        if word_count:
            metrics.append({"label": "Document Size", "value": f"{word_count:,} words"})
        
        if processing_time:
            metrics.append({"label": "Processing Time", "value": f"{processing_time:.1f}s"})
        
        if total_tokens:
            metrics.append({"label": "Total Tokens Used", "value": f"{total_tokens:,}"})
        
        # Get key points count
        key_points = result.get("extracted_info", {}).get("key_points", [])
        if key_points:
            metrics.append({"label": "Key Points", "value": len(key_points)})
        
        # Display metrics in a grid if we have any
        if metrics:
            # Determine columns (3 max, fewer if less metrics)
            columns = min(3, len(metrics))
            cols = st.columns(columns)
            
            # Place metrics in columns
            for i, metric in enumerate(metrics):
                col_index = i % columns
                with cols[col_index]:
                    st.metric(
                        label=metric.get("label", ""),
                        value=metric.get("value", "")
                    )
        
        # Display token usage by stage if available
        stage_tokens = statistics.get("stage_tokens", {})
        if stage_tokens:
            st.markdown("### Token Usage by Stage")
            stage_names = {
                "planning": "Planning",
                "extraction": "Extraction",
                "aggregation": "Aggregation",
                "evaluation": "Evaluation",
                "formatting": "Formatting",
                "review": "Review"
            }
            
            # Calculate percentages for bars
            max_tokens = max(stage_tokens.values()) if stage_tokens else 0
            
            # Display bars for each stage
            for stage, tokens in stage_tokens.items():
                percentage = min(100, (tokens / max_tokens) * 100) if max_tokens > 0 else 0
                display_name = stage_names.get(stage, stage.capitalize())
                
                # Create the progress bar HTML
                st.markdown(f"""
                <div style="margin: 8px 0;">
                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                        <div style="flex-grow: 1; font-size: 0.9rem;">{display_name}</div>
                        <div style="font-weight: 500; font-size: 0.9rem;">{tokens:,} tokens</div>
                    </div>
                    <div style="background-color: rgba(0, 0, 0, 0.1); height: 8px; border-radius: 4px; overflow: hidden;">
                        <div style="background-color: #2196F3; width: {percentage}%; height: 100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

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
        
        # Save output to files
        filename_base = document.filename.rsplit(".", 1)[0] if "." in document.filename else document.filename
        save_summary_output(result, filename_base)
        
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

def main():
    """Main function for the summarizer page."""
    # Initialize page
    initialize_page()
    
    # Page header using our component
    page_header("📝 Document Summarizer", "Transform documents into clear, readable summaries")
    
    # Show warning if assessment is not available
    if not assessment_available:
        st.warning(
            f"The summary assessment definition '{base_distill_id}' could not be loaded. " 
            "This may affect the functionality of this page. Please check your configuration."
        )
    
    # Document upload section
    section_header("Upload Document", 1)
    
    uploaded_file = st.file_uploader("Upload a document to summarize", type=["txt", "md", "docx", "pdf"])
    
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
        section_header("Configure Summary", 2)
        
        # Format selection (simplified)
        format_options = {
            "executive": "Concise overview (5-10% of original)",
            "comprehensive": "Detailed summary (15-25% of original)",
            "bullet_points": "Key points in list format",
            "narrative": "Flowing narrative style"
        }
        
        selected_format = st.radio(
            "Summary Format",
            options=list(format_options.keys()),
            format_func=lambda x: x.title().replace("_", " "),
            index=list(format_options.keys()).index(st.session_state.get("selected_format", "executive")),
            horizontal=True
        )
        
        # Update session state with selected format
        if selected_format != st.session_state.get("selected_format"):
            st.session_state.selected_format = selected_format
        
        # Display format description
        st.caption(format_options[selected_format])
        
        # Advanced settings in expander
        with st.expander("Advanced Options", expanded=False):
            # Model selection
            model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
            selected_model = st.selectbox("LLM Model", options=model_options, index=0)
            
            # Temperature
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
            
            # Include quotes option
            include_quotes = st.checkbox("Include Key Quotes", value=True)
            
            # Chunk settings
            chunk_size = st.slider("Chunk Size", min_value=1000, max_value=15000, value=6000, step=1000)
            chunk_overlap = st.slider("Chunk Overlap", min_value=100, max_value=1000, value=300, step=100)
        
        # Process button
        col1, col2 = st.columns([3, 1])
        with col1:
            if not st.session_state.processing_started:
                # Disable button if assessment not available
                button_disabled = not assessment_available
                
                if st.button("📝 Generate Summary", type="primary", use_container_width=True, disabled=button_disabled):
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
                    selected_assessment_id = base_distill_id
                    
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
            section_header("Summary Results", 4)
            
            if st.session_state.summary_result:
                logger.info("Displaying summary results")
                result = st.session_state.summary_result
                
                # Check for explicit error
                if "error" in result and result["error"]:
                    st.error(f"Error in summary generation: {result['error']}")
                    with st.expander("Show Error Details", expanded=True):
                        st.json(result)
                else:
                    # Display the summary result with improved rendering
                    display_summary_result(result)
                    
                    # Add download options
                    st.markdown("## Download Options")
                    
                    # Generate report markdown
                    report_md = format_assessment_report(result, "distill")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Download as markdown
                        download_md_button = st.download_button(
                            "⬇️ Download as Markdown",
                            data=report_md,
                            file_name=f"summary_{timestamp}.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Download as JSON
                        json_str = json.dumps(result, indent=2)
                        download_json_button = st.download_button(
                            "⬇️ Download Raw Data",
                            data=json_str,
                            file_name=f"summary_data_{timestamp}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    # Add debug expander
                    with st.expander("Debug: View Raw JSON Result", expanded=False):
                        st.json(result)
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