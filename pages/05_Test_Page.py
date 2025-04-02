# pages/05_Test_Page.py
import streamlit as st
import asyncio
import time
import json
import os
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("beyond-notes-tester")

# Import your core components
from core.models.document import Document
from core.orchestrator import Orchestrator
from assessments.loader import AssessmentLoader
from utils.paths import AppPaths
from utils.formatting import format_assessment_report
from utils.result_accessor import get_assessment_data, debug_result_structure

# Initialize paths
AppPaths.ensure_dirs()

# Page config
st.set_page_config(
    page_title="Beyond Notes - Advanced Testing",
    page_icon="🧪",
    layout="wide"
)

# Page title and introduction
st.title("🧪 Beyond Notes Advanced Testing")
st.markdown("""
This page allows you to test the assessment pipeline with detailed logging and 
visualization of agent progress. Select an assessment type, upload a document, 
and see how the multi-agent system processes your content.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("Test Configuration")
    
    # Assessment selection
    st.subheader("Assessment Type")
    assessment_loader = AssessmentLoader()
    assessment_configs = assessment_loader.get_assessment_configs_list()
    
    # Group configs by assessment_type for better organization
    assessment_types = {}
    for cfg in assessment_configs:
        if not cfg.get("is_template", False):  # Only show base types in the main selection
            a_type = cfg.get("assessment_type", "unknown")
            if a_type not in assessment_types:
                assessment_types[a_type] = []
            assessment_types[a_type].append(cfg)
    
    # Let user select assessment type first
    selected_type = st.selectbox(
        "Select Assessment Type",
        options=list(assessment_types.keys()),
        format_func=lambda x: {
            "distill": "📝 Summarization",
            "extract": "📋 Action Items",
            "assess": "⚠️ Issues & Risks",
            "analyze": "📊 Framework Analysis"
        }.get(x, x.title())
    )
    
    # Then show available configs for that type
    type_configs = assessment_types.get(selected_type, [])
    if type_configs:
        type_options = {cfg["id"]: f"{cfg['display_name']}" for cfg in type_configs}
        selected_assessment_id = st.selectbox(
            "Select Configuration", 
            options=list(type_options.keys()),
            format_func=lambda x: type_options.get(x, x)
        )
        
        # Show description of selected assessment
        selected_config = next((cfg for cfg in assessment_configs if cfg["id"] == selected_assessment_id), None)
        if selected_config:
            st.info(selected_config.get("description", "No description available"))
    else:
        st.warning(f"No configurations found for type '{selected_type}'")
        selected_assessment_id = None
    
    # Model selection
    st.subheader("Model")
    model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    selected_model = st.selectbox("Select LLM Model", options=model_options)
    
    # Chunking options
    st.subheader("Chunking Options")
    chunk_size = st.slider("Chunk Size", min_value=1000, max_value=15000, value=8000, step=1000)
    chunk_overlap = st.slider("Chunk Overlap", min_value=100, max_value=1000, value=300, step=100)
    
    # Debug options
    st.subheader("Debug Options")
    show_agent_outputs = st.checkbox("Show Agent Outputs", value=True)
    show_context_data = st.checkbox("Show Context Data", value=True)
    show_time_stats = st.checkbox("Show Time Statistics", value=True)
    
    # Create test options
    options = {
        "model": selected_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "debug_level": "verbose" if show_agent_outputs else "standard"
    }

# Create a two-column layout for input and results
col1, col2 = st.columns([3, 5])

with col1:
    st.header("Document Selection")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "md", "pdf", "docx"])
    
    # Sample document selection
    st.subheader("Or use a sample document")
    sample_dir = AppPaths.SAMPLES
    sample_files = list(sample_dir.glob("*.*")) if sample_dir.exists() else []
    
    if not sample_files:
        st.warning(f"No sample files found in {sample_dir}. Please upload a file instead.")
    
    sample_options = {str(file): file.name for file in sample_files}
    selected_sample = st.selectbox(
        "Select Sample Document", 
        options=[""] + list(sample_options.keys()),
        format_func=lambda x: "Select a sample..." if x == "" else sample_options.get(x, x)
    )
    
    # Document preview
    preview_container = st.container()
    
    # Test controls
    st.subheader("Test Controls")
    test_cols = st.columns(2)
    with test_cols[0]:
        start_test_button = st.button("Start Test", type="primary", use_container_width=True)
    with test_cols[1]:
        reset_button = st.button("Reset", type="secondary", use_container_width=True)
    
    if reset_button:
        # Clear session state
        for key in list(st.session_state.keys()):
            if key not in ["uploaded_file", "selected_sample"]:
                del st.session_state[key]
        st.rerun()

with col2:
    st.header("Test Results")
    
    # Initialize tabs for results
    tabs = st.tabs(["Pipeline Status", "Agent Progress", "Final Result", "Performance", "Debug"])
    pipeline_tab, agent_tab, result_tab, perf_tab, debug_tab = tabs
    
    with pipeline_tab:
        pipeline_status = st.empty()
    
    with agent_tab:
        agent_outputs = st.empty()
    
    with result_tab:
        final_result = st.empty()
    
    with perf_tab:
        performance_metrics = st.empty()
        
    with debug_tab:
        debug_output = st.empty()

# Document loading logic
def load_document():
    """Load document from upload or sample selection."""
    document = None
    
    if uploaded_file:
        # Save uploaded file temporarily
        temp_dir = AppPaths.get_temp_path("uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file_path = temp_dir / uploaded_file.name
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
            
        # Create document from file
        document = Document.from_file(temp_file_path)
        source = f"Uploaded file: {uploaded_file.name}"
        
    elif selected_sample:
        sample_path = Path(selected_sample)
        document = Document.from_file(sample_path)
        source = f"Sample file: {sample_path.name}"
    
    return document, source if document else (None, None)

# Progress callback
def progress_callback(progress, data):
    """Callback function for progress updates from the orchestrator."""
    if isinstance(data, dict):
        message = data.get("message", "Processing...")
        current_stage = data.get("current_stage")
        
        # Update session state
        st.session_state.current_progress = progress
        st.session_state.current_message = message
        st.session_state.current_stage = current_stage
        st.session_state.current_stages = data.get("stages", {})
        
        # Update timestamp
        st.session_state.last_update = time.time()
    else:
        # Simpler update if data is just a message string
        st.session_state.current_progress = progress
        st.session_state.current_message = data
        st.session_state.last_update = time.time()

# Pipeline visualization function
def render_pipeline_status():
    """Render the current pipeline status."""
    if not hasattr(st.session_state, "orchestrator") or not st.session_state.orchestrator:
        pipeline_status.info("No test running. Configure options and click 'Start Test'.")
        return
    
    # Get progress from orchestrator
    progress_data = st.session_state.orchestrator.get_progress()
    
    # Create progress bar
    overall_progress = progress_data.get("progress", 0)
    current_stage = progress_data.get("current_stage", "Not started")
    message = progress_data.get("message", "Waiting to start...")
    
    # Main progress bar
    pipeline_status.progress(overall_progress, text=f"Overall Progress: {overall_progress*100:.0f}%")
    
    # Current stage indicator
    pipeline_status.info(f"**Current Stage:** {current_stage} - {message}")
    
    # Show stages
    stages = progress_data.get("stages", {})
    if stages:
        # Create a visual pipeline diagram
        pipeline_html = """
        <style>
        .pipeline-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-top: 20px;
        }
        .pipeline-stage {
            padding: 10px;
            border-radius: 5px;
            position: relative;
        }
        .stage-completed {
            background-color: #d4f7d4;
            border: 1px solid #52c41a;
        }
        .stage-running {
            background-color: #e6f7ff;
            border: 1px solid #1890ff;
            animation: pulse 2s infinite;
        }
        .stage-failed {
            background-color: #fff1f0;
            border: 1px solid #ff4d4f;
        }
        .stage-waiting {
            background-color: #f5f5f5;
            border: 1px solid #d9d9d9;
        }
        .stage-header {
            font-weight: bold;
            display: flex;
            justify-content: space-between;
        }
        .stage-time {
            font-size: 0.8em;
            color: #888;
        }
        .stage-progress {
            height: 6px;
            background-color: #f0f0f0;
            border-radius: 3px;
            margin-top: 8px;
        }
        .stage-progress-bar {
            height: 100%;
            border-radius: 3px;
            background-color: #1890ff;
        }
        .stage-message {
            margin-top: 5px;
            font-size: 0.9em;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(24, 144, 255, 0.5); }
            70% { box-shadow: 0 0 0 5px rgba(24, 144, 255, 0); }
            100% { box-shadow: 0 0 0 0 rgba(24, 144, 255, 0); }
        }
        .stage-connector {
            width: 2px;
            height: 15px;
            background-color: #d9d9d9;
            margin: -10px auto 0 30px;
        }
        </style>
        <div class="pipeline-container">
        """
        
        # Get workflow stages in order
        workflow_stages = st.session_state.orchestrator.workflow_config.get("enabled_stages", [])
        
        for i, stage_name in enumerate(workflow_stages):
            stage_info = stages.get(stage_name, {"status": "not_started"})
            status = stage_info.get("status", "not_started")
            
            if status == "completed":
                stage_class = "stage-completed"
                icon = "✅"
                progress = 100
                progress_style = f"width: 100%;"
            elif status == "running":
                stage_class = "stage-running"
                icon = "⏳"
                progress = stage_info.get("progress", 0) * 100
                progress_style = f"width: {progress}%;"
            elif status == "failed":
                stage_class = "stage-failed"
                icon = "❌"
                progress = stage_info.get("progress", 0) * 100
                progress_style = f"width: {progress}%;"
            else:
                stage_class = "stage-waiting"
                icon = "⏱️"
                progress = 0
                progress_style = "width: 0%;"
                
            # Format duration if available
            duration = stage_info.get("duration")
            duration_str = f" ({duration:.2f}s)" if duration else ""
            
            # Build stage HTML
            pipeline_html += f"""
            <div class="pipeline-stage {stage_class}">
                <div class="stage-header">
                    <div>{icon} {stage_name.title()}</div>
                    <div class="stage-time">{progress:.0f}%{duration_str}</div>
                </div>
                <div class="stage-progress">
                    <div class="stage-progress-bar" style="{progress_style}"></div>
                </div>
            """
            
            # Add message or error if available
            if status == "running" and "message" in stage_info:
                pipeline_html += f'<div class="stage-message">{stage_info["message"]}</div>'
            elif status == "failed" and "error" in stage_info:
                pipeline_html += f'<div class="stage-message">Error: {stage_info["error"]}</div>'
                
            pipeline_html += "</div>"
            
            # Add connector if not the last stage
            if i < len(workflow_stages) - 1:
                pipeline_html += '<div class="stage-connector"></div>'
            
        pipeline_html += "</div>"
        
        # Display the pipeline visualization
        pipeline_status.markdown(pipeline_html, unsafe_allow_html=True)
    else:
        pipeline_status.info("Pipeline not started yet.")

# Render agent progress and outputs
def render_agent_progress():
    """Render detailed information about agent progress and outputs."""
    if not hasattr(st.session_state, "test_data") or not st.session_state.test_data:
        agent_outputs.info("No test data available yet.")
        return
    
    # Get the context's data store if available
    if show_context_data and hasattr(st.session_state, "orchestrator") and st.session_state.orchestrator.context:
        context = st.session_state.orchestrator.context
        
        # Data overview panel
        agent_outputs.markdown("## Agent Outputs")
        
        # Planning data
        planning_data = context.get_data("planning")
        if planning_data:
            with agent_outputs.expander("🧭 Planner Output", expanded=True):
                st.markdown("### Document Analysis Plan")
                st.json(planning_data)
        
        # Assessment type specific data
        assessment_type = context.assessment_type
        if assessment_type == "extract":
            item_type = "action_items"
            icon = "📋"
        elif assessment_type == "assess":
            item_type = "issues"
            icon = "⚠️"
        elif assessment_type == "distill":
            item_type = "key_points"
            icon = "📝"
        elif assessment_type == "analyze":
            item_type = "evidence"
            icon = "📊"
        else:
            item_type = "items"
            icon = "📄"
            
        # Extraction data
        extracted_items = context.get_data("extracted", item_type)
        if extracted_items:
            with agent_outputs.expander(f"{icon} Extracted {item_type.title().replace('_', ' ')} ({len(extracted_items)})", expanded=True):
                st.json(extracted_items)
        
        # Aggregation data
        aggregated_items = context.get_data("aggregated", item_type)
        if aggregated_items:
            with agent_outputs.expander(f"🧩 Aggregated {item_type.title().replace('_', ' ')} ({len(aggregated_items)})", expanded=True):
                st.json(aggregated_items)
        
        # Evaluation data
        evaluated_items = context.get_data("evaluated", item_type)
        if evaluated_items:
            with agent_outputs.expander(f"⚖️ Evaluated {item_type.title().replace('_', ' ')} ({len(evaluated_items)})", expanded=True):
                st.json(evaluated_items)
        
        # Overall assessment
        overall_assessment = context.get_data("evaluated", "overall_assessment")
        if overall_assessment:
            with agent_outputs.expander("📊 Overall Assessment", expanded=True):
                st.json(overall_assessment)
        
        # Formatted output
        formatted_data = context.get_data("formatted")
        if formatted_data:
            with agent_outputs.expander("📋 Formatter Output", expanded=True):
                st.json(formatted_data)
    
    else:
        agent_outputs.warning("Context data not available or display disabled.")

# Render final result
def render_final_result():
    """Render the final formatted result."""
    if not hasattr(st.session_state, "test_data") or not st.session_state.test_data:
        final_result.info("No test results available. Run a test first.")
        return
    
    test_data = st.session_state.test_data
    assessment_type = test_data.get("metadata", {}).get("assessment_type")
    
    if "result" in test_data and assessment_type:
        # Create download buttons
        download_cols = final_result.columns(2)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save files for download
        json_path = AppPaths.get_assessment_output_dir(assessment_type) / f"{assessment_type}_result_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
            
        with download_cols[0]:
            with open(json_path, "rb") as f:
                st.download_button(
                    label="Download JSON",
                    data=f,
                    file_name=f"{assessment_type}_result_{timestamp}.json",
                    mime="application/json"
                )
        
        # Try to format the result as markdown
        try:
            report_md = format_assessment_report(test_data, assessment_type)
            
            # Markdown download
            md_path = AppPaths.get_assessment_output_dir(assessment_type) / f"{assessment_type}_report_{timestamp}.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(report_md)
            
            with download_cols[1]:
                with open(md_path, "rb") as f:
                    st.download_button(
                        label="Download Markdown",
                        data=f,
                        file_name=f"{assessment_type}_report_{timestamp}.md",
                        mime="text/markdown"
                    )
            
            # Display the formatted result
            final_result.markdown(report_md)
            
        except Exception as e:
            final_result.error(f"Error formatting result: {str(e)}")
            
            # Use the result_accessor utility to get normalized data
            try:
                normalized_data = get_assessment_data(test_data, assessment_type)
                final_result.subheader("Raw Result Data (Normalized)")
                final_result.json(normalized_data)
            except Exception as e2:
                final_result.error(f"Error normalizing result: {str(e2)}")
                final_result.json(test_data.get("result", {}))
            
    elif "error" in test_data.get("metadata", {}):
        final_result.error(f"Test failed: {test_data['metadata']['error']}")
    else:
        final_result.warning("No result found in test data.")

# Render performance metrics
def render_performance_metrics():
    """Render detailed performance metrics."""
    if not hasattr(st.session_state, "test_data") or not st.session_state.test_data:
        performance_metrics.info("No performance data available yet.")
        return
    
    test_data = st.session_state.test_data
    
    if "statistics" in test_data:
        stats = test_data["statistics"]
        
        # Create dashboard layout
        metric_cols = performance_metrics.columns(4)
        
        # Total tokens
        total_tokens = stats.get("total_tokens", 0)
        with metric_cols[0]:
            st.metric("Total Tokens Used", f"{total_tokens:,}")
        
        # Total time
        total_time = test_data.get("metadata", {}).get("total_processing_time", 0)
        with metric_cols[1]:
            st.metric("Processing Time", f"{total_time:.2f}s")
            
        # LLM calls
        llm_calls = stats.get("total_llm_calls", 0)
        with metric_cols[2]:
            st.metric("LLM API Calls", f"{llm_calls:,}")
            
        # Chunks
        chunk_count = stats.get("total_chunks", 0)
        with metric_cols[3]:
            st.metric("Document Chunks", f"{chunk_count:,}")
        
        # Create charts
        chart_cols = performance_metrics.columns(2)
        
        # Tokens by stage
        stage_tokens = stats.get("stage_tokens", {})
        if stage_tokens:
            token_data = []
            for stage, tokens in stage_tokens.items():
                token_data.append({"stage": stage.title(), "tokens": tokens})
                
            if token_data:
                with chart_cols[0]:
                    st.subheader("Token Usage by Stage")
                    st.bar_chart(
                        token_data, 
                        x="stage", 
                        y="tokens"
                    )
        
        # Time by stage
        stage_durations = stats.get("stage_durations", {})
        if stage_durations:
            time_data = []
            for stage, duration in stage_durations.items():
                time_data.append({"stage": stage.title(), "seconds": duration})
                
            if time_data:
                with chart_cols[1]:
                    st.subheader("Processing Time by Stage")
                    st.bar_chart(
                        time_data, 
                        x="stage", 
                        y="seconds"
                    )
                    
        # Token efficiency
        if total_tokens > 0 and total_time > 0:
            efficiency_cols = performance_metrics.columns(3)
            with efficiency_cols[0]:
                tokens_per_second = total_tokens / total_time
                st.metric("Tokens/Second", f"{tokens_per_second:.2f}")
                
            with efficiency_cols[1]:
                if chunk_count > 0:
                    tokens_per_chunk = total_tokens / chunk_count
                    st.metric("Tokens/Chunk", f"{tokens_per_chunk:.2f}")
                
            with efficiency_cols[2]:
                if llm_calls > 0:
                    tokens_per_call = total_tokens / llm_calls
                    st.metric("Tokens/API Call", f"{tokens_per_call:.2f}")
    else:
        performance_metrics.warning("No statistics found in test data.")

# Debug information rendering
def render_debug_info():
    """Render detailed debug information."""
    if not hasattr(st.session_state, "test_data") or not st.session_state.test_data:
        debug_output.info("No debug data available yet.")
        return
    
    test_data = st.session_state.test_data
    
    # Use result_accessor's debug utility
    try:
        debug_info = debug_result_structure(test_data)
        
        with debug_output.expander("Result Structure Analysis", expanded=True):
            st.subheader("Top-Level Keys")
            st.write(debug_info.get("top_level_keys", []))
            
            st.subheader("Data Locations")
            for loc, content in debug_info.get("primary_data_locations", {}).items():
                st.write(f"**{loc}:** {content}")
            
            st.subheader("Path Analysis")
            for path in debug_info.get("paths", []):
                st.code(path, language=None)
        
        with debug_output.expander("Raw Result JSON", expanded=False):
            st.json(test_data)
            
        with debug_output.expander("Orchestrator & Context State", expanded=False):
            if hasattr(st.session_state, "orchestrator") and st.session_state.orchestrator:
                orchestrator = st.session_state.orchestrator
                
                st.subheader("Orchestrator Info")
                st.write({
                    "assessment_id": orchestrator.assessment_id,
                    "assessment_type": orchestrator.assessment_type,
                    "display_name": orchestrator.display_name,
                    "model": orchestrator.llm.model if hasattr(orchestrator, "llm") else "Unknown"
                })
                
                st.subheader("Pipeline State")
                if hasattr(orchestrator, "context") and orchestrator.context:
                    st.write(orchestrator.context.pipeline_state)
                else:
                    st.write("No context available")
    except Exception as e:
        debug_output.error(f"Error generating debug information: {str(e)}")
        debug_output.json(test_data)

# Main test function
async def run_test(document, options):
    """Run the test with the orchestrator and document."""
    try:
        # Initialize orchestrator
        orchestrator = Orchestrator(
            assessment_id=selected_assessment_id,
            options=options
        )
        
        # Store orchestrator in session state
        st.session_state.orchestrator = orchestrator
        
        # Set progress callback
        orchestrator.set_progress_callback(progress_callback)
        
        # Process document
        result = await orchestrator.process_document(document)
        
        # Update session state
        st.session_state.test_data = result
        st.session_state.test_completed = True
        
        return result
    except Exception as e:
        logger.exception("Error during test execution")
        st.error(f"Test execution failed: {str(e)}")
        st.session_state.test_completed = True
        return {"result": None, "metadata": {"error": str(e)}, "statistics": {}}

# Initialize session state
if "test_started" not in st.session_state:
    st.session_state.test_started = False
    
if "test_completed" not in st.session_state:
    st.session_state.test_completed = False
    
if "current_progress" not in st.session_state:
    st.session_state.current_progress = 0
    
if "current_message" not in st.session_state:
    st.session_state.current_message = "Not started"
    
if "current_stage" not in st.session_state:
    st.session_state.current_stage = None
    
if "last_update" not in st.session_state:
    st.session_state.last_update = time.time()

# Show document preview
if uploaded_file or selected_sample:
    document, source = load_document()
    
    if document:
        with preview_container:
            st.markdown(f"**Source:** {source}")
            st.markdown(f"**Word count:** {document.word_count}")
            
            # Show preview of document text
            preview_text = document.text[:1000] + ("..." if len(document.text) > 1000 else "")
            st.text_area("Document Preview", preview_text, height=200)

# Run test when button is clicked
if start_test_button:
    if not selected_assessment_id:
        st.error("Please select an assessment configuration before starting the test.")
    elif not uploaded_file and not selected_sample:
        st.error("Please upload a file or select a sample document.")
    else:
        document, _ = load_document()
        
        if document:
            # Mark as started but not completed
            st.session_state.test_started = True
            st.session_state.test_completed = False
            
            # Create progress placeholder
            progress_placeholder = st.empty()
            progress_placeholder.info("Starting test...")
            
            # Run test
            asyncio.run(run_test(document, options))
            
            # Clear progress placeholder
            progress_placeholder.empty()

# Update displays if test completed
if hasattr(st.session_state, "test_completed") and st.session_state.test_completed:
    render_pipeline_status()
    render_agent_progress()
    render_final_result()
    render_performance_metrics()
    render_debug_info()
# Update pipeline status during test
elif hasattr(st.session_state, "test_started") and st.session_state.test_started:
    render_pipeline_status()

# Auto-refresh during active test
if st.session_state.get("test_started", False) and not st.session_state.get("test_completed", False):
    time_since_update = time.time() - st.session_state.get("last_update", time.time())
    if time_since_update < 30:  # Only auto-refresh for 30 seconds after last update
        st.rerun()