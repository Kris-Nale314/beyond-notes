"""
Enhanced Test Runner for Beyond Notes

This script provides a simple Streamlit interface for testing the multi-agent
document processing pipeline. It allows you to select an assessment type,
configure basic parameters, and run the assessment on a test document.

The interface includes detailed logging, status tracking, and result visualization.
"""

import os
import sys
import asyncio
import time
import json
import logging
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
APP_ROOT = Path(__file__).parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

# Import project components
try:
    from core.models.document import Document
    from core.orchestrator import Orchestrator
    from assessments.loader import AssessmentLoader
    from utils.paths import AppPaths
except ImportError as e:
    st.error(f"Failed to import necessary modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Assessment Test Runner - Beyond Notes",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants for default settings
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_CHUNK_SIZE = 8000
DEFAULT_CHUNK_OVERLAP = 300

# Initialize paths
AppPaths.ensure_dirs()

def load_test_document(file_path=None):
    """Load a test document, either from a specified path or using default sample."""
    try:
        if file_path is None:
            # Use default test file
            sample_dir = AppPaths.SAMPLES
            test_files = list(sample_dir.glob("*.txt"))
            if not test_files:
                st.error(f"No test files found in {sample_dir}. Please add a .txt file to this directory.")
                return None
            file_path = test_files[0]  # Use first available .txt file
            
        if not Path(file_path).exists():
            st.error(f"File not found: {file_path}")
            return None
            
        with open(file_path, 'r', encoding='utf-8') as f:
            doc_text = f.read()
        
        document = Document(text=doc_text, filename=Path(file_path).name)
        return document
        
    except Exception as e:
        st.error(f"Error loading document: {e}")
        logger.error(f"Document loading error: {e}", exc_info=True)
        return None

async def run_assessment(document, assessment_id, options, progress_placeholder):
    """Run the assessment pipeline with the given options."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        return None
    
    # Progress display elements
    progress_bar = progress_placeholder.progress(0.0)
    status_text = st.empty()
    
    # Test metrics
    metrics = {
        "start_time": time.time(),
        "stage_times": {},
        "current_stage": None
    }
    
    # Status tracking
    status_container = st.empty()
    status_data = []
    
    def update_progress(progress, message):
        """Callback for updating progress UI and collecting metrics."""
        progress_bar.progress(progress)
        status_text.write(f"**Status:** {message}")
        
        # If orchestrator is available, update status
        if 'orchestrator_instance' in st.session_state:
            try:
                progress_data = st.session_state.orchestrator_instance.get_progress()
                current_stage = progress_data.get("current_stage")
                
                # Track stage timing
                if current_stage and current_stage != metrics.get("current_stage"):
                    metrics["stage_times"][current_stage] = {"start": time.time()}
                    metrics["current_stage"] = current_stage
                    
                    # Add stage to status tracking
                    status_data.append({
                        "Stage": current_stage,
                        "Status": "Running",
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Details": message
                    })
                    
                # Check for stage completion
                for stage_name, stage_info in progress_data.get("stages", {}).items():
                    stage_status = stage_info.get("status")
                    if stage_status in ["completed", "failed"]:
                        # If we haven't recorded this status yet
                        if stage_name in metrics["stage_times"] and "end" not in metrics["stage_times"][stage_name]:
                            metrics["stage_times"][stage_name]["end"] = time.time()
                            metrics["stage_times"][stage_name]["status"] = stage_status
                            
                            # Update status tracking
                            status_data.append({
                                "Stage": stage_name,
                                "Status": stage_status.capitalize(),
                                "Time": datetime.now().strftime("%H:%M:%S"),
                                "Details": stage_info.get("message", "")
                            })
                
                # Update status display
                status_df = pd.DataFrame(status_data)
                if not status_df.empty:
                    status_container.dataframe(status_df, use_container_width=True)
            except Exception as e:
                logger.error(f"Error updating status: {e}")
    
    try:
        # Initialize orchestrator
        st.session_state.orchestrator_instance = Orchestrator(
            assessment_id=assessment_id,
            options=options,
            api_key=api_key
        )
        
        # Process document
        update_progress(0.0, f"Starting assessment: {st.session_state.orchestrator_instance.display_name}")
        result = await st.session_state.orchestrator_instance.process_with_progress(
            document=document,
            progress_callback=update_progress
        )
        
        # Store context for inspection if available
        if hasattr(st.session_state.orchestrator_instance, 'context'):
            st.session_state.processing_context = st.session_state.orchestrator_instance.context
        
        # Compute final metrics
        metrics["end_time"] = time.time()
        metrics["total_duration"] = metrics["end_time"] - metrics["start_time"]
        
        # Update final progress
        progress_bar.progress(1.0)
        status_text.write(f"**Status:** Completed in {metrics['total_duration']:.2f} seconds")
        
        return result
    
    except Exception as e:
        st.error(f"Assessment execution error: {e}")
        logger.error(f"Assessment execution error: {e}", exc_info=True)
        return None

def format_json(json_obj):
    """Format JSON for display with syntax highlighting."""
    if json_obj is None:
        return "No data available"
    
    # Convert to string if it's not already
    if not isinstance(json_obj, str):
        json_str = json.dumps(json_obj, indent=2)
    else:
        json_str = json_obj
        
    return json_str

def display_stage_timing(metrics):
    """Create a bar chart of stage durations."""
    if not metrics or "stage_times" not in metrics:
        return
    
    # Calculate durations
    durations = {}
    for stage, timing in metrics["stage_times"].items():
        if "start" in timing and "end" in timing:
            durations[stage] = timing["end"] - timing["start"]
    
    # Sort by duration (descending)
    sorted_stages = sorted(durations.items(), key=lambda x: x[1], reverse=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data
    stages = [s[0] for s in sorted_stages]
    times = [s[1] for s in sorted_stages]
    bars = ax.barh(stages, times)
    
    # Add labels
    ax.set_xlabel("Duration (seconds)")
    ax.set_title("Processing Time by Stage")
    
    # Add values on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.1, 
                bar.get_y() + bar.get_height()/2, 
                f"{width:.2f}s", 
                ha='left', 
                va='center')
    
    # Show plot
    st.pyplot(fig)

def display_token_usage(result):
    """Display token usage statistics if available."""
    if not result or "statistics" not in result:
        return
    
    stats = result["statistics"]
    
    # Check if we have token data
    if "total_tokens" not in stats:
        st.write("No token usage data available")
        return
    
    # Basic token info
    st.metric("Total Tokens Used", f"{stats['total_tokens']:,}")
    
    # Per-stage token usage if available
    if "stage_tokens" in stats and stats["stage_tokens"]:
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get data
        stages = list(stats["stage_tokens"].keys())
        tokens = list(stats["stage_tokens"].values())
        
        # Sort by usage (descending)
        sorted_data = sorted(zip(stages, tokens), key=lambda x: x[1], reverse=True)
        stages = [s[0] for s in sorted_data]
        tokens = [s[1] for s in sorted_data]
        
        # Plot data
        bars = ax.barh(stages, tokens)
        
        # Add labels
        ax.set_xlabel("Tokens")
        ax.set_title("Token Usage by Stage")
        
        # Add values on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.1, 
                    bar.get_y() + bar.get_height()/2, 
                    f"{width:,}", 
                    ha='left', 
                    va='center')
        
        # Show plot
        st.pyplot(fig)

def main():
    """Main application logic."""
    st.title("🧪 Beyond Notes Assessment Test Runner")
    st.caption("Test your multi-agent document processing pipeline")
    
    # Load assessment configurations
    if "assessment_loader" not in st.session_state:
        st.session_state.assessment_loader = AssessmentLoader()
    
    loader = st.session_state.assessment_loader
    
    # Sidebar for test configuration
    with st.sidebar:
        st.header("Test Configuration")
        
        # 1. Select test document
        st.subheader("Test Document")
        sample_dir = AppPaths.SAMPLES
        test_files = [f.name for f in sample_dir.glob("*.txt") if f.is_file()]
        
        if not test_files:
            st.warning(f"No test documents found in {sample_dir}")
            selected_file = None
        else:
            selected_file = st.selectbox(
                "Test Document", 
                test_files,
                help="Select a test document to process"
            )
        
        # 2. Select assessment type
        st.subheader("Assessment Type")
        assessment_configs = loader.get_assessment_configs_list()
        
        # Organize by base type vs template
        base_types = [cfg for cfg in assessment_configs if not cfg.get("is_template", False)]
        templates = [cfg for cfg in assessment_configs if cfg.get("is_template", False)]
        
        # Default to templates if available, otherwise base types
        use_template = len(templates) > 0 and st.checkbox("Use Template", value=False)
        
        if use_template:
            selected_config = st.selectbox(
                "Template",
                templates,
                format_func=lambda x: x.get("display_name", x.get("id", "Unknown")),
                help="Select an assessment template"
            )
        else:
            selected_config = st.selectbox(
                "Assessment Type",
                base_types,
                format_func=lambda x: x.get("display_name", x.get("id", "Unknown")),
                help="Select a base assessment type"
            )
        
        assessment_id = selected_config.get("id") if selected_config else None
        
        # 3. Model and chunking settings
        st.subheader("LLM Configuration")
        model = st.selectbox(
            "Model", 
            ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"], 
            index=0, 
            help="Select LLM model (gpt-3.5-turbo is faster for testing)"
        )
        
        chunk_size = st.slider(
            "Chunk Size", 
            1000, 16000, DEFAULT_CHUNK_SIZE, 500, 
            help="Text characters per chunk"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap", 
            0, 1000, DEFAULT_CHUNK_OVERLAP, 50, 
            help="Overlap between chunks"
        )
        
        # 4. Debug options
        st.subheader("Debug Options")
        debug_mode = st.checkbox("Debug Mode", value=True, help="Show detailed debug information")
        
        # Test execution button
        st.markdown("---")
        run_test = st.button("🚀 Run Assessment Test", type="primary", use_container_width=True)
    
    # Main content area
    if not selected_file:
        st.warning("Please select a test document in the sidebar to continue.")
        return
    
    if not assessment_id:
        st.warning("Please select an assessment type in the sidebar to continue.")
        return
    
    # Document preview
    document = None
    if selected_file:
        document = load_test_document(sample_dir / selected_file)
        if document:
            st.subheader("📄 Test Document")
            col1, col2, col3 = st.columns(3)
            col1.metric("Filename", document.filename)
            col2.metric("Word Count", f"{document.word_count:,}")
            col3.metric("Est. Tokens", f"{document.estimated_tokens:,}")
            
            with st.expander("Document Preview", expanded=False):
                st.text_area("Content", document.text[:2000] + ("..." if len(document.text) > 2000 else ""), 
                           height=200, disabled=True)
    
    # Run assessment
    if run_test and document and assessment_id:
        st.subheader("⚙️ Assessment Execution")
        
        # Prepare runtime options
        runtime_options = {
            "model": model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "debug_mode": debug_mode
        }
        
        # Clear previous results
        if "assessment_result" in st.session_state:
            del st.session_state.assessment_result
        if "processing_context" in st.session_state:
            del st.session_state.processing_context
        if "test_metrics" in st.session_state:
            del st.session_state.test_metrics
        
        # Progress placeholder
        progress_placeholder = st.empty()
        
        # Run assessment
        st.session_state.test_metrics = {"start_time": time.time()}
        result = asyncio.run(run_assessment(document, assessment_id, runtime_options, progress_placeholder))
        st.session_state.test_metrics["end_time"] = time.time()
        st.session_state.test_metrics["total_duration"] = (
            st.session_state.test_metrics["end_time"] - st.session_state.test_metrics["start_time"]
        )
        
        if result:
            st.session_state.assessment_result = result
            st.success(f"✅ Assessment completed in {st.session_state.test_metrics['total_duration']:.2f} seconds")
            
            # Results section
            st.subheader("📊 Assessment Results")
            
            # Result tabs
            summary_tab, timing_tab, content_tab, debug_tab = st.tabs([
                "Summary", "Timing & Tokens", "Content", "Debug"
            ])
            
            with summary_tab:
                # Basic assessment info
                st.markdown("### Assessment Information")
                col1, col2, col3 = st.columns(3)
                
                assessment_type = result.get("metadata", {}).get("assessment_type", "unknown")
                assessment_name = result.get("metadata", {}).get("assessment_display_name", "Unknown")
                
                col1.metric("Assessment Type", assessment_type)
                col2.metric("Assessment Name", assessment_name)
                col3.metric("Processing Time", f"{st.session_state.test_metrics['total_duration']:.2f}s")
                
                # Warnings and errors
                st.markdown("### Warnings and Errors")
                warnings = result.get("metadata", {}).get("warnings", [])
                errors = result.get("metadata", {}).get("errors", [])
                
                if not warnings and not errors:
                    st.success("No warnings or errors reported")
                
                if warnings:
                    with st.expander(f"Warnings ({len(warnings)})", expanded=True):
                        for i, warning in enumerate(warnings):
                            st.markdown(f"**Warning {i+1}**: {warning.get('message')} *(Stage: {warning.get('stage')})*")
                
                if errors:
                    with st.expander(f"Errors ({len(errors)})", expanded=True):
                        for i, error in enumerate(errors):
                            st.error(f"**Error {i+1}**: {error.get('message')} *(Stage: {error.get('stage')})*")
                
                # Stages summary
                st.markdown("### Processing Stages")
                stages_completed = result.get("metadata", {}).get("stages_completed", [])
                stages_failed = result.get("metadata", {}).get("stages_failed", [])
                
                if stages_completed:
                    st.markdown(f"**Completed Stages**: {', '.join(stages_completed)}")
                
                if stages_failed:
                    st.error(f"**Failed Stages**: {', '.join(stages_failed)}")
            
            with timing_tab:
                # Stage timing
                st.markdown("### Stage Processing Times")
                display_stage_timing(st.session_state.test_metrics)
                
                # Token usage
                st.markdown("### Token Usage")
                display_token_usage(result)
            
            with content_tab:
                # Show main result content
                st.markdown("### Assessment Output")
                
                result_data = result.get("result", {})
                if not result_data:
                    st.warning("No result data available")
                else:
                    # Format based on assessment type
                    if assessment_type == "distill":
                        # Show summary and key points
                        st.markdown("#### Summary")
                        st.markdown(result_data.get("summary", "No summary available"))
                        
                        st.markdown("#### Topics")
                        topics = result_data.get("topics", [])
                        for topic in topics:
                            with st.expander(topic.get("topic", "Unnamed Topic"), expanded=False):
                                for point in topic.get("key_points", []):
                                    st.markdown(f"- {point}")
                                
                    elif assessment_type == "extract":
                        # Show action items
                        action_items = result_data.get("action_items", [])
                        st.markdown(f"#### Action Items ({len(action_items)})")
                        
                        # Group by owner if available
                        action_items_by_owner = {}
                        for item in action_items:
                            owner = item.get("owner", "Unassigned")
                            if owner not in action_items_by_owner:
                                action_items_by_owner[owner] = []
                            action_items_by_owner[owner].append(item)
                        
                        # Display by owner
                        for owner, items in action_items_by_owner.items():
                            with st.expander(f"{owner} ({len(items)} items)", expanded=True):
                                for item in items:
                                    priority = item.get("priority", "medium")
                                    due_date = item.get("due_date", "Unspecified")
                                    
                                    st.markdown(f"**{item.get('description', 'Action Item')}**")
                                    st.markdown(f"Priority: {priority.upper()} | Due: {due_date}")
                        
                    elif assessment_type == "assess":
                        # Show issues
                        issues = result_data.get("issues", [])
                        st.markdown(f"#### Issues ({len(issues)})")
                        
                        # Group by severity
                        issues_by_severity = {}
                        for issue in issues:
                            severity = issue.get("severity", "medium")
                            if severity not in issues_by_severity:
                                issues_by_severity[severity] = []
                            issues_by_severity[severity].append(issue)
                        
                        # Display by severity (in order)
                        for severity in ["critical", "high", "medium", "low"]:
                            if severity in issues_by_severity:
                                with st.expander(f"{severity.upper()} ({len(issues_by_severity[severity])} issues)", 
                                               expanded=severity in ["critical", "high"]):
                                    for issue in issues_by_severity[severity]:
                                        st.markdown(f"**{issue.get('title', 'Issue')}**")
                                        st.markdown(f"Category: {issue.get('category', 'Uncategorized')}")
                                        st.markdown(issue.get('description', ''))
                        
                    elif assessment_type == "analyze":
                        # Show framework analysis
                        st.markdown("#### Framework Analysis")
                        
                        executive_summary = result_data.get("executive_summary", "")
                        if executive_summary:
                            st.markdown("##### Executive Summary")
                            st.markdown(executive_summary)
                        
                        # Show dimensions
                        dimensions = result_data.get("dimension_assessments", {})
                        if isinstance(dimensions, dict):
                            for dim_name, dimension in dimensions.items():
                                with st.expander(f"{dim_name} (Rating: {dimension.get('dimension_rating', 'N/A')})", 
                                               expanded=False):
                                    st.markdown(dimension.get('dimension_summary', ''))
                                    
                                    # Show criteria
                                    criteria = dimension.get('criteria_assessments', [])
                                    for criterion in criteria:
                                        st.markdown(f"**{criterion.get('criteria_name', 'Criterion')}** - Rating: {criterion.get('rating', 'N/A')}")
                                        st.markdown(criterion.get('rationale', ''))
                    
                    # Raw view option
                    with st.expander("View Raw Result JSON", expanded=False):
                        st.code(format_json(result_data), language="json")
            
            with debug_tab:
                if debug_mode:
                    # Debug information
                    st.markdown("### Debug Information")
                    
                    # Raw result
                    with st.expander("Full Result", expanded=False):
                        st.code(format_json(result), language="json")
                    
                    # Context data if available
                    if "processing_context" in st.session_state:
                        with st.expander("Processing Context", expanded=False):
                            # Only show key context data, not the full document text
                            context = st.session_state.processing_context
                            if hasattr(context, 'to_dict'):
                                context_dict = context.to_dict(include_document=False)
                                st.code(format_json(context_dict), language="json")
                            else:
                                st.write("Context object doesn't have to_dict method")
                else:
                    st.info("Enable Debug Mode in the sidebar to see detailed debug information.")
        else:
            st.error("❌ Assessment failed - no result returned.")
            # Check for any logs or error info
            if "processing_context" in st.session_state:
                context = st.session_state.processing_context
                if hasattr(context, 'metadata'):
                    errors = context.metadata.get("errors", [])
                    if errors:
                        st.error(f"Errors: {json.dumps(errors, indent=2)}")
    else:
        # Initial state - show instructions
        st.info("Configure test parameters in the sidebar and click 'Run Assessment Test' to start.")

if __name__ == "__main__":
    main()