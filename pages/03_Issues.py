# pages/02_Issues.py
import streamlit as st
import os
import asyncio
import logging
import json
import time
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("beyond-notes-issues")

# Import components with clear error handling
try:
    logger.info("Importing core components...")
    from core.models.document import Document
    from core.orchestrator import Orchestrator
    from assessments.loader import AssessmentLoader
    from utils.paths import AppPaths
    from utils.ui.styles import get_base_styles
    from utils.ui.components import page_header, section_header, display_document_preview
    
    # Import our new renderers and data accessor
    from utils.ui.renderers import render_issues_result, render_detailed_progress
    from utils.accessor import DataAccessor
    
    logger.info("Successfully imported all components")
except ImportError as e:
    error_msg = f"Failed to import components: {str(e)}"
    logger.error(error_msg, exc_info=True)
    st.error(error_msg)
    st.stop()

# Ensure directories exist
try:
    logger.info("Ensuring directories exist...")
    AppPaths.ensure_dirs()
    logger.info("Directories verified")
except Exception as e:
    error_msg = f"Failed to ensure directories: {str(e)}"
    logger.error(error_msg, exc_info=True)
    st.error(error_msg)
    st.stop()

# Set up loader and assessment ID
BASE_ASSESS_ID = "base_assess_issue_v1"
assessment_available = False

try:
    logger.info("Loading assessment configurations...")
    loader = AssessmentLoader()
    
    # Check for base assess assessment
    base_assess_config = loader.load_config(BASE_ASSESS_ID)
    if base_assess_config:
        logger.info(f"✅ Successfully loaded base assess config '{BASE_ASSESS_ID}'")
        assessment_available = True
    else:
        logger.warning(f"⚠️ Could not load base assess config '{BASE_ASSESS_ID}'")
        
except Exception as e:
    error_msg = f"Error checking configurations: {str(e)}"
    logger.error(error_msg, exc_info=True)

# Page config
st.set_page_config(
    page_title="Beyond Notes - Issue Assessment",
    page_icon="⚠️",
    layout="wide",
)

# Apply shared styles
st.markdown(get_base_styles(), unsafe_allow_html=True)

# Add enhanced styling for assessment options
st.markdown("""
<style>
    /* Assessment option styles */
    .assessment-option {
        background-color: rgba(0,0,0,0.05);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        border: 2px solid transparent;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    
    .assessment-option:hover {
        background-color: rgba(244, 67, 54, 0.1);
        transform: translateY(-2px);
    }
    
    .assessment-option.selected {
        border-color: #F44336;
        background-color: rgba(244, 67, 54, 0.15);
    }
    
    .assessment-title {
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 8px;
    }
    
    .assessment-description {
        opacity: 0.85;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    /* Agent status indicator */
    .agent-indicator {
        display: flex;
        align-items: center;
        background-color: rgba(244, 67, 54, 0.1);
        border-radius: 8px;
        padding: 10px 16px;
        margin: 16px 0;
        border-left: 3px solid #F44336;
    }
    
    .agent-avatar {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #F44336;
        color: white;
        font-size: 18px;
        margin-right: 12px;
        flex-shrink: 0;
    }
    
    .agent-info {
        flex-grow: 1;
    }
    
    .agent-name {
        font-weight: 600;
        margin-bottom: 4px;
    }
    
    .agent-task {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .agent-status {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #4CAF50;
        margin-left: 12px;
        animation: pulse 1.5s infinite;
    }
    
    /* Issue severity options */
    .severity-option {
        display: flex;
        align-items: center;
        padding: 12px;
        margin-bottom: 8px;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .severity-critical {
        background-color: rgba(244, 67, 54, 0.1);
        border-left: 3px solid #F44336;
    }
    
    .severity-high {
        background-color: rgba(255, 152, 0, 0.1);
        border-left: 3px solid #FF9800;
    }
    
    .severity-medium {
        background-color: rgba(33, 150, 243, 0.1);
        border-left: 3px solid #2196F3;
    }
    
    .severity-low {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 3px solid #4CAF50;
    }
    
    .severity-icon {
        font-size: 1.5rem;
        margin-right: 12px;
    }
    
    .severity-label {
        font-weight: 500;
    }
    
    .severity-description {
        font-size: 0.85rem;
        opacity: 0.8;
        margin-top: 4px;
    }
    
    /* Debug panel */
    .debug-panel {
        background-color: rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        padding: 12px;
        margin-top: 24px;
    }
    
    .debug-header {
        font-weight: 600;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
    }
    
    .debug-icon {
        margin-right: 8px;
    }
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
    
    if "issues_result" not in st.session_state:
        st.session_state.issues_result = None
    
    if "selected_detail_level" not in st.session_state:
        st.session_state.selected_detail_level = "standard"
        
    if "context_path" not in st.session_state:
        st.session_state.context_path = None
        
    if "context_obj" not in st.session_state:
        st.session_state.context_obj = None
        
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False

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
    """Callback for progress updates from the orchestrator with enhanced agent tracking."""
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
            
            # Track the current active agent
            for stage_name, stage_info in stages.items():
                if stage_info.get("status") == "running":
                    agent_name = stage_info.get("agent", "Unknown")
                    st.session_state.current_agent = agent_name
                    st.session_state.current_agent_task = message
                    logger.info(f"Active agent: {agent_name} - Task: {message}")
    else:
        logger.info(f"Progress {progress:.1%} - {data}")
        st.session_state.current_progress = float(progress)
        st.session_state.progress_message = data

def save_output_files(result, document, orchestrator):
    """Save all output files: JSON result, context pickle, and markdown report."""
    try:
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get base filename from document
        filename_base = document.filename.rsplit(".", 1)[0] if "." in document.filename else document.filename
        # Clean up for safe filenames
        filename_base = ''.join(c for c in filename_base if c.isalnum() or c in "._- ").strip()
        
        # Get run ID if available
        run_id = ""
        if orchestrator.context and hasattr(orchestrator.context, "run_id"):
            run_id = f"_{orchestrator.context.run_id}"
        
        # Ensure output directory exists
        output_dir = AppPaths.get_assessment_output_dir("assess")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 1. Save JSON result
        json_filename = f"assess_result_{filename_base}_{timestamp}{run_id}.json"
        json_path = output_dir / json_filename
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved JSON result to: {json_path}")
        
        # 2. Save context object if available
        context_path = None
        if orchestrator.context:
            try:
                # Create a deep copy to avoid modifying the original
                import copy
                import pickle
                
                context_copy = copy.deepcopy(orchestrator.context)
                
                # Remove potentially problematic attributes for serialization
                if hasattr(context_copy, '_progress_callback'):
                    context_copy._progress_callback = None
                
                # Save using pickle
                context_filename = f"assess_context_{filename_base}_{timestamp}{run_id}.pkl"
                context_path = output_dir / context_filename
                
                with open(context_path, 'wb') as f:
                    pickle.dump(context_copy, f)
                    
                logger.info(f"Saved context to: {context_path}")
                
                # Store in session state for later use
                st.session_state.context_path = str(context_path)
                st.session_state.context_obj = context_copy
                
            except Exception as e:
                logger.error(f"Error saving context: {str(e)}", exc_info=True)
                context_path = None
        
        # 3. Create and save markdown report using our data accessor
        try:
            report_filename = f"assess_report_{filename_base}_{timestamp}{run_id}.md"
            report_path = output_dir / report_filename
            
            # Use DataAccessor to get standardized data
            data = DataAccessor.get_issues_data(result, orchestrator.context)
            
            # Generate a simple markdown report from the data
            report_md = generate_markdown_report(data)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_md)
            logger.info(f"Saved markdown report to: {report_path}")
            
            return (json_path, context_path, report_path)
        except Exception as e:
            logger.error(f"Error generating markdown report: {e}", exc_info=True)
            return (json_path, context_path, None)
            
    except Exception as e:
        logger.error(f"Error saving output files: {e}", exc_info=True)
        return (None, None, None)

def generate_markdown_report(data):
    """Generate a markdown report from standardized issues data."""
    # Get key data for the report
    issues = data.get("issues", [])
    executive_summary = data.get("executive_summary", "")
    statistics = data.get("statistics", {})
    
    # Start building the report
    report = "# Issue Assessment Report\n\n"
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report += f"*Generated on: {timestamp}*\n\n"
    
    # Add executive summary if available
    if executive_summary:
        report += "## Executive Summary\n\n"
        report += f"{executive_summary}\n\n"
    
    # Group issues by severity
    issues_by_severity = {"critical": [], "high": [], "medium": [], "low": []}
    for issue in issues:
        if isinstance(issue, dict):
            # Use evaluated_severity if available, otherwise use severity
            severity = issue.get("evaluated_severity", issue.get("severity", "medium")).lower()
            if severity in issues_by_severity:
                issues_by_severity[severity].append(issue)
    
    # Add issues grouped by severity
    for severity in ["critical", "high", "medium", "low"]:
        severity_issues = issues_by_severity.get(severity, [])
        if severity_issues:
            report += f"## {severity.upper()} Severity Issues ({len(severity_issues)})\n\n"
            
            for i, issue in enumerate(severity_issues, 1):
                title = issue.get("title", "Untitled Issue")
                description = issue.get("description", "")
                category = issue.get("category", "process")
                impact = issue.get("potential_impact", issue.get("impact", ""))
                
                report += f"### {i}. {title}\n\n"
                report += f"**Category:** {category.upper()}\n\n"
                report += f"{description}\n\n"
                
                if impact:
                    report += f"**Potential Impact:** {impact}\n\n"
                
                # Add recommendations if available
                recommendations = issue.get("recommendations", issue.get("suggested_recommendations", []))
                if recommendations:
                    report += "**Recommendations:**\n\n"
                    for rec in recommendations:
                        report += f"- {rec}\n"
                    report += "\n"
    
    # Add statistics
    if statistics:
        report += "## Statistics\n\n"
        
        # Add issue count by severity
        by_severity = statistics.get("by_severity", {})
        if by_severity:
            report += "### Issues by Severity\n\n"
            for severity in ["critical", "high", "medium", "low"]:
                count = by_severity.get(severity, 0)
                report += f"- **{severity.title()}:** {count}\n"
            report += "\n"
        
        # Add issue count by category
        by_category = statistics.get("by_category", {})
        if by_category:
            report += "### Issues by Category\n\n"
            for category, count in by_category.items():
                report += f"- **{category.title()}:** {count}\n"
            report += "\n"
        
        # Add processing stats
        if "total_tokens" in statistics:
            report += f"- **Processing Tokens:** {statistics['total_tokens']:,}\n"
        
        if "total_issues" in statistics:
            report += f"- **Total Issues:** {statistics['total_issues']}\n"
    
    return report

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
        
        # Save all output files and get paths
        json_path, context_path, md_path = save_output_files(result, document, orchestrator)
        
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

def display_agent_status():
    """Display the current active agent with enhanced styling."""
    if "current_agent" not in st.session_state:
        return
    
    agent_name = st.session_state.current_agent
    agent_task = st.session_state.get("current_agent_task", "Working...")
    
    # Map agent names to friendly display names and icons
    agent_display_names = {
        "PlannerAgent": ("Planner", "🧭"),
        "ExtractorAgent": ("Extractor", "🔍"),
        "AggregatorAgent": ("Aggregator", "🧩"),
        "EvaluatorAgent": ("Evaluator", "⚖️"),
        "FormatterAgent": ("Formatter", "📊"),
        "ReviewerAgent": ("Reviewer", "🔍")
    }
    
    # Get display name and icon
    display_name, icon = agent_display_names.get(agent_name, ("Agent", "🤖"))
    
    # Render the agent indicator
    st.markdown(f"""
    <div class="agent-indicator">
        <div class="agent-avatar">{icon}</div>
        <div class="agent-info">
            <div class="agent-name">{display_name} Agent</div>
            <div class="agent-task">{agent_task}</div>
        </div>
        <div class="agent-status"></div>
    </div>
    """, unsafe_allow_html=True)

def display_detail_level_options():
    """Display detail level options with enhanced styling."""
    # Detail level options with descriptions
    detail_options = {
        "essential": {
            "title": "Essential Issues Only",
            "description": "Focus only on the most significant issues and problems."
        },
        "standard": {
            "title": "Standard Assessment",
            "description": "Balanced analysis of issues with moderate detail."
        },
        "comprehensive": {
            "title": "Comprehensive Analysis",
            "description": "In-depth assessment of all potential issues and risks."
        }
    }
    
    # Get currently selected level
    selected_level = st.session_state.get("selected_detail_level", "standard")
    
    # Display options in a grid
    st.subheader("Detail Level")
    
    for level_key, level_info in detail_options.items():
        selected_class = "selected" if level_key == selected_level else ""
        
        st.markdown(f"""
        <div class="assessment-option {selected_class}" onclick="selectLevel('{level_key}')">
            <div class="assessment-title">{level_info["title"]}</div>
            <div class="assessment-description">{level_info["description"]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add JavaScript to handle clicks and update radio buttons
    st.markdown("""
    <script>
    function selectLevel(level) {
        // Find the corresponding radio button and click it
        const radios = document.querySelectorAll('input[type="radio"]');
        for (const radio of radios) {
            if (radio.value === level) {
                radio.click();
                break;
            }
        }
    }
    </script>
    """, unsafe_allow_html=True)
    
    # Hidden radio buttons for actual selection
    level_choice = st.radio(
        "Select Detail Level",
        options=list(detail_options.keys()),
        format_func=lambda x: detail_options[x]["title"],
        index=list(detail_options.keys()).index(selected_level),
        key="detail_level_radio",
        label_visibility="collapsed"
    )
    
    # Update session state when selection changes
    if level_choice != st.session_state.get("selected_detail_level"):
        st.session_state.selected_detail_level = level_choice
        logger.info(f"Detail level changed to: {level_choice}")

def display_severity_options():
    """Display severity level descriptions."""
    st.markdown("### Severity Levels")
    st.caption("Issues will be classified into these severity levels:")
    
    # Severity descriptions
    severity_descriptions = {
        "critical": {
            "icon": "🔴",
            "description": "Immediate threat requiring urgent attention. Significant impact on objectives."
        },
        "high": {
            "icon": "🟠",
            "description": "Major issue with significant impact. Should be addressed promptly."
        },
        "medium": {
            "icon": "🔵",
            "description": "Moderate issue with noteworthy impact. Should be addressed in due course."
        },
        "low": {
            "icon": "🟢",
            "description": "Minor issue with limited impact. Address when convenient."
        }
    }
    
    # Display each severity level
    for severity, info in severity_descriptions.items():
        st.markdown(f"""
        <div class="severity-option severity-{severity}">
            <div class="severity-icon">{info["icon"]}</div>
            <div>
                <div class="severity-label">{severity.upper()}</div>
                <div class="severity-description">{info["description"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_debug_panel():
    """Display debugging information when debug mode is enabled."""
    if not st.session_state.debug_mode:
        return
    
    st.markdown("""
    <div class="debug-panel">
        <div class="debug-header">
            <span class="debug-icon">🔍</span> Debug Panel
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show current session state
    st.write("### Session State")
    session_state_display = {
        "document_loaded": st.session_state.document_loaded,
        "processing_started": st.session_state.processing_started,
        "processing_complete": st.session_state.processing_complete,
        "current_progress": st.session_state.current_progress,
        "current_stage": st.session_state.get("current_stage", "None"),
        "selected_detail_level": st.session_state.selected_detail_level,
        "context_path": st.session_state.context_path
    }
    st.json(session_state_display)
    
    # Show result data structure if available
    if st.session_state.issues_result:
        st.write("### Result Structure")
        result = st.session_state.issues_result
        if isinstance(result, dict):
            st.write("Top-level keys:", list(result.keys()))
            if "result" in result:
                st.write("Result-level keys:", list(result["result"].keys()))
        
        # Option to see full result
        if st.checkbox("Show full result data"):
            st.json(result)
    
    # Show context data structure if available
    if st.session_state.context_obj:
        st.write("### Context Structure")
        context = st.session_state.context_obj
        
        # Show context data keys
        if hasattr(context, "data"):
            st.write("Context data categories:", list(context.data.keys()))
            
            # Show data for each category
            for category, data in context.data.items():
                if isinstance(data, dict):
                    st.write(f"{category} keys:", list(data.keys()))
                elif isinstance(data, list):
                    st.write(f"{category} length:", len(data))
                else:
                    st.write(f"{category} type:", type(data).__name__)
        
        # Option to see full context data
        if st.checkbox("Show full context data"):
            if hasattr(context, "data"):
                st.json({k: str(type(v))[:100] for k, v in context.data.items()})

def main():
    """Main function for the issues assessment page."""
    # Initialize page
    initialize_page()
    
    # Page header
    page_header("⚠️ Issue Assessment", "Identify problems, challenges, risks, and concerns in your documents")
    
    # Debug mode toggle in sidebar
    with st.sidebar:
        st.session_state.debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
    
    # Show warning if assessment is not available
    if not assessment_available:
        st.warning(
            f"The issue assessment definition '{BASE_ASSESS_ID}' could not be loaded. " 
            "This may affect the functionality of this page. Please check your configuration."
        )
    
    # Document upload section
    section_header("Upload Document", 1)
    
    uploaded_file = st.file_uploader("Upload a document to analyze", type=["txt", "md", "docx", "pdf"])
    
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
        
        # Display detail level options with enhanced styling
        display_detail_level_options()
        
        # Severity level explainer
        with st.expander("Understanding Severity Levels", expanded=False):
            display_severity_options()
        
        # Advanced settings in expander
        with st.expander("Advanced Options", expanded=False):
            # Model selection
            model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
            selected_model = st.selectbox("LLM Model", options=model_options, index=0)
            
            # Temperature
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
            
            # Category focus
            categories = ["technical", "process", "resource", "quality", "risk", "compliance"]
            focus_categories = st.multiselect(
                "Focus Categories (Optional)",
                options=categories,
                default=[],
                help="Optionally focus on specific issue categories. Leave empty to include all."
            )
            
            # Minimum severity
            min_severity_options = {
                "low": "Include all issues",
                "medium": "Medium severity and above", 
                "high": "Only high and critical",
                "critical": "Critical issues only"
            }
            min_severity = st.select_slider(
                "Minimum Severity",
                options=list(min_severity_options.keys()),
                value="low",
                format_func=lambda x: min_severity_options[x]
            )
            
            # Chunk settings
            chunk_size = st.slider("Chunk Size", min_value=1000, max_value=15000, value=5000, step=1000)
            chunk_overlap = st.slider("Chunk Overlap", min_value=100, max_value=1000, value=300, step=100)
        
        # Process button
        col1, col2 = st.columns([3, 1])
        with col1:
            if not st.session_state.processing_started:
                # Disable button if assessment not available
                button_disabled = not assessment_available
                
                if st.button("🔍 Identify Issues", type="primary", use_container_width=True, disabled=button_disabled):
                    logger.info("Identify Issues button clicked")
                    
                    # Create assessment options
                    assessment_options = {
                        "detail_level": st.session_state.selected_detail_level,
                        "focus_categories": focus_categories,
                        "minimum_severity": min_severity
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
                    selected_assessment_id = BASE_ASSESS_ID
                    
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
                st.session_state.context_path = None
                st.session_state.context_obj = None
                
                logger.info("Session state reset")
                st.rerun()
    
    # Processing section
    if st.session_state.processing_started:
        section_header("Processing Status", 3)
        
        if not st.session_state.processing_complete:
            logger.info("Processing in progress, showing status...")
            
            # Use the imported render_detailed_progress function
            render_detailed_progress(st.session_state)
            
            # Display current active agent
            display_agent_status()
            
            # Process document if not already processing
            if 'is_processing' not in st.session_state:
                logger.info("Starting document processing...")
                
                document_to_process = st.session_state.document
                assessment_id = st.session_state.selected_assessment_id
                options = st.session_state.processing_options
                
                # Mark as processing to prevent multiple runs
                st.session_state.is_processing = True
                
                try:
                    # Run async processing
                    asyncio.run(process_document(document_to_process, assessment_id, options))
                    logger.info("Document processing completed")
                except Exception as e:
                    error_msg = f"Error during document processing: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    st.error(error_msg)
                finally:
                    # Reset processing flag
                    if 'is_processing' in st.session_state:
                        del st.session_state.is_processing
                    
                    # This rerun will either show results or error message
                    st.rerun()
        
        # Results section
        if st.session_state.processing_complete:
            section_header("Issue Assessment Results", 4)
            
            if st.session_state.issues_result:
                logger.info("Displaying issues assessment results")
                result = st.session_state.issues_result
                
                # Check for explicit error
                if isinstance(result, dict) and "error" in result and result["error"]:
                    st.error(f"Error in issue assessment: {result['error']}")
                    with st.expander("Show Error Details", expanded=True):
                        st.json(result)
                else:
                    try:
                        # Load context object for enhanced rendering
                        context = st.session_state.context_obj
                        
                        # Use our improved renderer with the DataAccessor
                        render_issues_result(result, context)
                        logger.info("Successfully rendered issues result")
                        
                    except Exception as e:
                        logger.error(f"Error rendering issues result: {e}", exc_info=True)
                        st.error(f"Error displaying issues: {str(e)}")
                        
                        # Show debug info automatically on error
                        st.session_state.debug_mode = True
                        display_debug_panel()
                    
                    # Add an expander for viewing raw JSON output
                    with st.expander("View Raw Output Data", expanded=False):
                        st.write("This is the raw JSON data generated by the processing pipeline:")
                        st.json(result)
                
                # Add download options
                st.markdown("### Download Options")
                
                # Generate download data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Generate markdown report
                try:
                    # Use the DataAccessor to get standardized data
                    data = DataAccessor.get_issues_data(result, st.session_state.context_obj)
                    report_md = generate_markdown_report(data)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Download as markdown
                        st.download_button(
                            "⬇️ Download as Markdown",
                            data=report_md,
                            file_name=f"issues_report_{timestamp}.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Download as JSON
                        json_str = json.dumps(result, indent=2)
                        st.download_button(
                            "⬇️ Download Raw Data",
                            data=json_str,
                            file_name=f"issues_data_{timestamp}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                except Exception as e:
                    logger.error(f"Error generating download options: {e}", exc_info=True)
                    st.error(f"Could not prepare download options: {str(e)}")
            else:
                error_msg = "Processing completed but no issues were identified. Please check the logs for details."
                logger.error(error_msg)
                st.error(error_msg)
    
    # Display debug panel if debug mode is enabled
    display_debug_panel()
                
if __name__ == "__main__":
    try:
        logger.info("Starting 02_Issues.py main execution")
        main()
        logger.info("02_Issues.py execution completed successfully")
    except Exception as e:
        error_msg = f"Unhandled exception in main execution: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}")