# pages/06_Test.py
import streamlit as st
import os
import asyncio
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timezone # Added timezone for consistency

# Configure logging
# Use a unique name for this page's logger if desired
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("beyond-notes.summarizer_page")

# --- Add project root to sys.path if running as a script (might not be needed in Streamlit multipage context) ---
# import sys
# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# --- Assume imports work correctly in Streamlit multipage app context ---

# Import components
# Use try-except for robustness, especially if running outside main project easily
try:
    from core.models.document import Document
    from core.orchestrator import Orchestrator
    from assessments.loader import AssessmentLoader
    from utils.paths import AppPaths
    from utils.formatting import format_assessment_report
except ImportError as e:
    logger.error(f"Error importing components: {e}. Ensure page is run within Streamlit context or paths are correct.", exc_info=True)
    st.error(f"Failed to load application components: {e}")
    st.stop() # Stop if essential components are missing


# Ensure directories exist (moved to run once potentially)
try:
    AppPaths.ensure_dirs()
except Exception as e:
    logger.error(f"Failed to ensure app directories: {e}", exc_info=True)
    st.error(f"Failed to setup application directories: {e}")
    st.stop()


# Page config
st.set_page_config(
    page_title="Beyond Notes - Smart Summarizer",
    page_icon="📝",
    layout="wide",
)

# --- (Keep CSS and Color definitions as they were) ---
# Define accent colors
PRIMARY_COLOR = "#4CAF50"  # Green
SECONDARY_COLOR = "#2196F3"  # Blue
ACCENT_COLOR = "#FF9800"  # Orange
ERROR_COLOR = "#F44336"  # Red

# CSS to customize the appearance
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem; }
    .section-header { font-size: 1.5rem; font-weight: 600; margin-top: 1rem; margin-bottom: 0.5rem; }
    .info-box { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .accent-border { border-left: 4px solid #FF9800; padding-left: 1rem; }
    .progress-container { margin-top: 2rem; margin-bottom: 2rem; }
    .results-container { margin-top: 2rem; }
    .format-card { padding: 1rem; border-radius: 0.5rem; background-color: rgba(255, 255, 255, 0.05); margin-bottom: 1rem; border-left: 4px solid #2196F3; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; }
    .format-card:hover { transform: translateY(-2px); box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
    .format-card.selected { border-left: 4px solid #4CAF50; background-color: rgba(76, 175, 80, 0.1); }
    .option-group { padding: 1rem; border-radius: 0.5rem; background-color: rgba(255, 255, 255, 0.03); margin-bottom: 1rem; }
    .summary-container { padding: 1.5rem; border-radius: 0.5rem; background-color: rgba(255, 255, 255, 0.05); margin-bottom: 1rem; }
    .stage-progress { padding: 0.5rem; border-radius: 0.3rem; margin-bottom: 0.5rem; background-color: rgba(255, 255, 255, 0.03); }
    .stage-progress.completed-stage { border-left: 3px solid #4CAF50; }
    .stage-progress.running-stage { border-left: 3px solid #2196F3; }
    .stage-progress.failed-stage { border-left: 3px solid #F44336; }
</style>
""", unsafe_allow_html=True)


def initialize_summarizer_page():
    """Initialize the session state variables for the summarizer page."""
    # Initialize only if keys don't exist to preserve state across reruns
    if "summarizer_initialized" not in st.session_state:
        st.session_state.document_loaded = False
        st.session_state.processing_started = False
        st.session_state.processing_complete = False
        st.session_state.current_progress = 0.0
        st.session_state.progress_message = "Not started"
        st.session_state.summary_result = None
        st.session_state.selected_format = "executive" # Default format
        st.session_state.original_document = None
        st.session_state.user_summary_options = {} # Store user selections here
        st.session_state.current_stage = None
        st.session_state.stages_info = {}
        st.session_state.summarizer_initialized = True # Mark as initialized


# --- (Keep find_distill_assessment_configs, get_distill_user_options functions as they were) ---
@st.cache_data(ttl=3600) # Cache loaded configs for an hour
def find_distill_assessment_configs():
    """Find and load all distill-type assessment configurations."""
    # ... (function content remains the same) ...
    try:
        # Load assessment configurations
        assessment_loader = AssessmentLoader()
        all_configs = assessment_loader.get_assessment_configs_list()

        # Filter for distill type assessments
        distill_configs = [cfg for cfg in all_configs if cfg.get("assessment_type") == "distill"]

        if not distill_configs:
            st.error("No distill assessment configurations found. Please check your setup.")
            return None

        # Return all distill configs, with base configs first, then templates
        base_configs = [cfg for cfg in distill_configs if not cfg.get("is_template", False)]
        template_configs = [cfg for cfg in distill_configs if cfg.get("is_template", False)]

        # Sort base configs and template configs alphabetically by display name
        base_configs.sort(key=lambda x: x.get("display_name", x["id"]))
        template_configs.sort(key=lambda x: x.get("display_name", x["id"]))

        return base_configs + template_configs
    except Exception as e:
        st.error(f"Error loading assessment configurations: {str(e)}")
        logger.error(f"Error loading assessment configurations: {e}", exc_info=True)
        return None

@st.cache_data(ttl=3600) # Cache user options
def get_distill_user_options(assessment_id):
    """Get user options for a specific distill assessment."""
    # ... (function content remains the same) ...
    try:
        # Load assessment loader
        assessment_loader = AssessmentLoader()

        # Load the assessment configuration
        config = assessment_loader.load_config(assessment_id)
        if not config:
            logger.error(f"Configuration not found for assessment_id: {assessment_id}")
            return {}

        # Get user options
        return config.get("user_options", {})
    except Exception as e:
        logger.error(f"Error loading user options: {e}", exc_info=True)
        return {}

# --- (Keep load_document function as it was) ---
def load_document(file_object):
    """Load document from uploaded file."""
    # ... (function content remains the same) ...
    try:
        # Save uploaded file temporarily
        temp_dir = AppPaths.get_temp_path("uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file_path = temp_dir / file_object.name

        with open(temp_file_path, "wb") as f:
            f.write(file_object.getvalue())

        # Create document from file
        document = Document.from_file(temp_file_path)
        return document, temp_file_path
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        logger.error(f"Error loading document: {e}", exc_info=True)
        return None, None

def progress_callback(progress, data):
    """Callback function for progress updates from the orchestrator."""
    # Ensure updates happen to session state for Streamlit reactivity
    if isinstance(data, dict):
        st.session_state.current_progress = float(progress) # Ensure float
        st.session_state.progress_message = data.get("message", "Processing...")
        st.session_state.current_stage = data.get("current_stage")
        # Update stages info carefully
        current_stages = st.session_state.get("stages_info", {})
        current_stages.update(data.get("stages", {}))
        st.session_state.stages_info = current_stages

    elif isinstance(data, str):
        # Handle simple string message
        st.session_state.current_progress = float(progress)
        st.session_state.progress_message = data
    else:
        # Fallback if data type is unexpected
        st.session_state.current_progress = float(progress)
        st.session_state.progress_message = "Processing..."

    # Log progress updates for debugging if needed
    # logger.debug(f"Progress Callback: {progress:.2f} - {st.session_state.progress_message}")

    # Optional: Force a rerun if immediate UI update is desired,
    # but this can be disruptive. Usually, rely on widget interactions or final rerun.
    # st.rerun()


def render_pipeline_status():
    """Render the current pipeline status using progress bar and stage details."""
    # Uses session state updated by the progress_callback
    progress_value = float(st.session_state.get("current_progress", 0.0))
    st.progress(progress_value)

    # Current stage and progress message
    current_stage_name = st.session_state.get("current_stage")
    if current_stage_name:
        stage_display = current_stage_name.replace("_", " ").title()
        st.markdown(f"**Current Stage:** `{stage_display}`")
    else:
         st.markdown(f"**Current Stage:** `Initializing`")

    st.caption(st.session_state.get("progress_message", "Waiting to start..."))

    # Show detailed stage information if available
    stages_info = st.session_state.get("stages_info", {})
    if stages_info:
        st.markdown("---") # Divider
        # Define expected order (adjust based on your typical pipeline)
        stage_order = ['document_analysis', 'chunking', 'planning', 'extraction', 'aggregation', 'evaluation', 'formatting']
        # Sort stages based on defined order, putting unknown stages last
        sorted_stage_names = sorted(stages_info.keys(), key=lambda x: stage_order.index(x) if x in stage_order else float('inf'))

        for stage_name in sorted_stage_names:
            stage_info = stages_info[stage_name]
            status = stage_info.get("status", "not_started")
            progress = float(stage_info.get("progress", 0.0)) # Ensure float
            message = stage_info.get("message", "")

            # Determine emoji based on status
            if status == "completed": emoji = "✅"
            elif status == "running": emoji = "⏳"
            elif status == "failed": emoji = "❌"
            else: emoji = "⏱️"

            display_name = stage_name.replace("_", " ").title()
            progress_pct_str = f"({progress:.0%})" if status == "running" and progress > 0 else ""
            status_str = f"`{status.upper()}`"

            # Use columns for better layout
            col1, col2 = st.columns([3, 1])
            with col1:
                 st.markdown(f"{emoji} **{display_name}** {progress_pct_str}")
                 if message and status != "completed": # Show message unless completed
                      st.caption(f"> {message[:100]}") # Truncate long messages
            with col2:
                 st.markdown(f"<div style='text-align: right;'>{status_str}</div>", unsafe_allow_html=True)


# --- (Keep display_document_preview function as it was) ---
def display_document_preview(document):
    """Display a preview of the loaded document."""
    # ... (function content remains the same) ...
    st.markdown("### Document Preview")
    st.markdown(f"**Filename:** `{document.filename}`") # Use markdown code format
    st.markdown(f"**Word Count:** `{document.word_count}`")
    preview_length = min(1000, len(document.text))
    preview_text = document.text[:preview_length]
    if len(document.text) > preview_length:
        preview_text += "..."
    st.text_area("Document Content Preview", preview_text, height=200, disabled=True) # Make preview read-only


# --- (Keep async process_document function as it was, ensures errors are caught) ---
async def process_document(document, assessment_id, options):
    """Process document using the orchestrator."""
    # ... (function content remains the same) ...
    try:
        # Get API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key not found. Cannot process.")
            return None

        # Initialize orchestrator (assuming it takes api_key now or loads it internally)
        # Adjust if Orchestrator init signature changed
        orchestrator = Orchestrator(assessment_id, options=options, api_key=api_key)

        # Set progress callback
        orchestrator.set_progress_callback(progress_callback)

        # Process document
        logger.info(f"Orchestrator starting processing for assessment {assessment_id}")
        result = await orchestrator.process_document(document)
        logger.info(f"Orchestrator finished processing for assessment {assessment_id}")

        # Update session state *after* completion
        st.session_state.summary_result = result
        st.session_state.processing_complete = True # Mark as complete

        return result
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        logger.error(f"Error processing document: {e}", exc_info=True)
        st.session_state.processing_complete = True # Mark as complete even on error
        st.session_state.summary_result = None # Ensure result is None on error
        return None


# --- (Keep display_summary_formats function as it was) ---
# Note: This function uses buttons which cause reruns. State needs careful handling.
def display_summary_formats(assessment_id):
    """Display available summary formats for selection."""
    # ... (function content largely remains the same) ...
    user_options = get_distill_user_options(assessment_id)
    format_options = user_options.get("format", {}).get("options", {})

    if not format_options:
        st.info("No specific format options found for this assessment type.")
        return

    st.markdown("### Choose Summary Format")

    # Create columns dynamically based on number of options
    num_options = len(format_options)
    cols = st.columns(num_options)

    for i, (format_key, format_desc) in enumerate(format_options.items()):
        with cols[i]:
            is_selected = (format_key == st.session_state.selected_format)
            # Use a container with border and click simulation
            container = st.container(border=True)
            container.markdown(f"**{format_key.title()}**")
            container.caption(format_desc)
            # Use a real button for selection action
            if container.button(f"Select {format_key.title()}", key=f"btn_format_{format_key}", use_container_width=True, type="primary" if is_selected else "secondary"):
                st.session_state.selected_format = format_key
                # No need for callback, state change triggers rerun
                st.rerun() # Rerun to update selection visually


# --- (Keep display_summary_options function as it was) ---
# IMPORTANT: This function *returns* the selected options. We need to store them.
def display_summary_options(assessment_id):
    """Display available summary options and return selected values."""
    # ... (function content remains the same) ...
    user_options = get_distill_user_options(assessment_id)

    st.markdown("### Customize Summary")

    options_values = {} # Dictionary to hold selected values

    # Create option groups in columns
    col1, col2 = st.columns(2)

    with col1:
        if "detail_level" in user_options:
            opt_cfg = user_options["detail_level"]
            with st.container(border=True):
                st.markdown(f"**{opt_cfg.get('display_name', 'Detail Level')}**")
                st.caption(opt_cfg.get("description", ""))
                options = list(opt_cfg.get("options", {}).keys())
                default_index = options.index(opt_cfg.get("default")) if opt_cfg.get("default") in options else 0
                options_values["detail_level"] = st.selectbox("Select level", options, index=default_index,
                                                            format_func=lambda x: opt_cfg.get("options", {}).get(x, x),
                                                            key="detail_level_select", label_visibility="collapsed")

        if "length" in user_options:
            opt_cfg = user_options["length"]
            with st.container(border=True):
                st.markdown(f"**{opt_cfg.get('display_name', 'Length')}**")
                st.caption(opt_cfg.get("description", ""))
                options = list(opt_cfg.get("options", {}).keys())
                default_index = options.index(opt_cfg.get("default")) if opt_cfg.get("default") in options else 0
                options_values["length"] = st.selectbox("Select length", options, index=default_index,
                                                        format_func=lambda x: opt_cfg.get("options", {}).get(x, x),
                                                        key="length_select", label_visibility="collapsed")

    with col2:
        if "structure_preference" in user_options:
             opt_cfg = user_options["structure_preference"]
             with st.container(border=True):
                 st.markdown(f"**{opt_cfg.get('display_name', 'Structure')}**")
                 st.caption(opt_cfg.get("description", ""))
                 options = list(opt_cfg.get("options", {}).keys())
                 default_index = options.index(opt_cfg.get("default")) if opt_cfg.get("default") in options else 0
                 options_values["structure_preference"] = st.selectbox("Select structure", options, index=default_index,
                                                                      format_func=lambda x: opt_cfg.get("options", {}).get(x, x),
                                                                      key="structure_select", label_visibility="collapsed")

        if "include_quotes" in user_options:
            opt_cfg = user_options["include_quotes"]
            with st.container(border=True):
                 st.markdown(f"**{opt_cfg.get('display_name', 'Include Quotes')}**")
                 st.caption(opt_cfg.get("description", ""))
                 options_values["include_quotes"] = st.checkbox("Include key quotes", value=opt_cfg.get("default", False),
                                                               key="include_quotes_check")

    # Focus Areas (full width below columns)
    if "focus_areas" in user_options:
         opt_cfg = user_options["focus_areas"]
         with st.container(border=True):
              st.markdown(f"**{opt_cfg.get('display_name', 'Focus Areas')}**")
              st.caption(opt_cfg.get("description", ""))
              options_values["focus_areas"] = st.multiselect("Select focus areas", options=opt_cfg.get("options", []),
                                                           default=opt_cfg.get("default", []), key="focus_areas_multi")

    return options_values


# --- (Keep display_summary_result function as it was, but ensure keys match actual results) ---
def display_summary_result(result, format_type):
    """Display the summary result, adapting to potential structure variations."""
    # ... (function content remains mostly the same, ensure keys like 'overview',
    # 'summary', 'topics', 'key_points', 'decisions', 'action_items' are checked robustly) ...
    if not result:
        st.error("No summary result available to display.")
        return

    try:
        # Extract data safely using .get()
        formatted_data = result.get("result", {})
        if not formatted_data:
             st.warning("Result structure seems empty or invalid.")
             st.json(result) # Show raw result if format is weird
             return

        metadata = result.get("metadata", {})
        # Stats might be nested differently, check common places
        stats = formatted_data.get("statistics", result.get("statistics", {}))

        # Display the formatted result
        st.markdown("---") # Divider before results
        st.markdown("## Summary Generated")
        st.markdown(f"_(Format: {format_type.title()})_")

        # Main Summary Content (adapt based on typical 'distill' output)
        # Use markdown containers for better visual separation
        with st.container(border=True):
             st.markdown("### Executive Summary / Overview")
             # Check multiple potential keys for the main summary text
             summary_text = formatted_data.get("executive_summary") or \
                           formatted_data.get("overview") or \
                           formatted_data.get("summary") or \
                           "No main summary text found."
             st.markdown(summary_text)

        # Topics / Key Points
        topics = formatted_data.get("topics", [])
        key_points = formatted_data.get("key_points", []) # Standalone key points

        if topics:
             st.markdown("### Topics Covered")
             for i, topic_data in enumerate(topics):
                  topic_title = topic_data.get('topic', f'Topic {i+1}')
                  topic_importance = f" ({topic_data.get('importance', '').title()})" if topic_data.get('importance') else ""
                  with st.expander(f"{topic_title}{topic_importance}", expanded=(i < 2)): # Expand first few
                       points_in_topic = topic_data.get("key_points", [])
                       if points_in_topic:
                            for point in points_in_topic:
                                 st.markdown(f"- {point}") # Assuming point is a string here
                       elif "details" in topic_data: # Show details if no key_points list
                            st.markdown(topic_data["details"])
                       else:
                            st.write("_No specific points listed for this topic._")
        elif key_points:
             st.markdown("### Key Points")
             for point in key_points:
                  # Handle if points are dicts or strings
                  if isinstance(point, dict):
                       point_text = point.get("text", point.get("point", "Invalid point format"))
                       importance = f" _{point.get('importance', '').title()}_" if point.get('importance') else ""
                       st.markdown(f"- {point_text}{importance}")
                  elif isinstance(point, str):
                       st.markdown(f"- {point}")
        else:
             st.info("No topics or key points section found in the summary.")


        # Decisions and Action Items (if applicable to summary)
        decisions = formatted_data.get("decisions", [])
        action_items = formatted_data.get("action_items", [])

        if decisions or action_items:
             st.markdown("### Decisions & Actions")
             if decisions:
                  st.markdown("#### Decisions Made")
                  for dec in decisions:
                       if isinstance(dec, dict):
                            st.markdown(f"- {dec.get('decision', 'N/A')}")
                       elif isinstance(dec, str):
                            st.markdown(f"- {dec}")
             if action_items:
                  st.markdown("#### Action Items")
                  for act in action_items:
                       if isinstance(act, dict):
                            owner = f" (Owner: {act.get('owner', 'N/A')})" if act.get('owner') else ""
                            st.markdown(f"- {act.get('action', act.get('description', 'N/A'))}{owner}")
                       elif isinstance(act, str):
                            st.markdown(f"- {act}")

        # --- Stats Display --- (Keep as before, ensure keys are safe)
        st.markdown("---")
        st.markdown("### Statistics")
        col1, col2, col3 = st.columns(3)
        original_words = stats.get("original_word_count", metadata.get("document_info", {}).get("word_count", "N/A"))
        summary_words = stats.get("summary_word_count", "N/A")
        compression = stats.get("compression_ratio")
        col1.metric("Original Words", f"{original_words:,}" if isinstance(original_words, int) else original_words)
        col2.metric("Summary Words", f"{summary_words:,}" if isinstance(summary_words, int) else summary_words)
        col3.metric("Compression Ratio", f"{compression:.1%}" if isinstance(compression, (int, float)) else "N/A")
        # Add other relevant stats if available in `stats` dict

        # --- Download Buttons --- (Keep as before)
        st.markdown("---")
        st.markdown("### Download Summary")
        try:
            summary_md = format_assessment_report(result, "distill") # Assumes this function works
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_format_type = "".join(c if c.isalnum() else "_" for c in format_type) # Sanitize format type for filename
            md_filename = f"summary_{safe_format_type}_{timestamp}.md"
            json_filename = f"summary_{safe_format_type}_{timestamp}.json"

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download as Markdown", data=summary_md.encode('utf-8'), file_name=md_filename, mime="text/markdown"
                )
            with col2:
                 # Use ensure_ascii=False for better unicode handling in JSON output
                 json_data = json.dumps(result, indent=2, ensure_ascii=False)
                 st.download_button(
                     "Download as JSON", data=json_data.encode('utf-8'), file_name=json_filename, mime="application/json"
                 )
        except Exception as e:
            st.error(f"Error preparing downloads: {str(e)}")
            logger.error(f"Error preparing downloads: {e}", exc_info=True)


    except Exception as e:
        st.error(f"An unexpected error occurred while displaying the summary: {str(e)}")
        logger.error(f"Error displaying summary result: {e}", exc_info=True)
        st.markdown("### Raw Result Data (Error Fallback)")
        st.json(result) # Show raw result on display error


# --- (Keep async reformat_summary - Note: current impl re-runs everything) ---
async def reformat_summary(document, assessment_id, options):
    """Reprocesses the document with potentially new formatting options."""
    # ... (function content remains the same - it currently reruns the whole process) ...
    # A future improvement would be to modify Orchestrator to only run formatting.
    logger.warning("Reformat currently reruns the entire analysis pipeline.")
    return await process_document(document, assessment_id, options)


def main():
    """Main function for the summarizer page."""
    # Initialize page state once
    initialize_summarizer_page()

    # Header
    st.markdown('<div class="main-header">Smart Document Summarizer</div>', unsafe_allow_html=True)
    st.markdown("Transform your documents into concise, customizable summaries using AI-powered analysis.")
    st.divider()

    # Check API Key
    if not st.session_state.get("has_api_key", False):
        st.warning(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable "
            "or add it to your `.env` file to use this feature."
        )
        st.stop() # Stop execution if API key is missing

    # Find all distill assessments
    distill_configs = find_distill_assessment_configs()
    if not distill_configs:
        # Error already shown in function if loading failed
        return # Stop if no configs loaded

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # Assessment selection
        assessment_options = {cfg["id"]: f"{cfg['display_name']}" for cfg in distill_configs}
        # Find index of default (first base type or first overall)
        default_assessment = next((cfg for cfg in distill_configs if not cfg.get("is_template", False)), distill_configs[0])
        default_assessment_id = default_assessment["id"]
        default_index = list(assessment_options.keys()).index(default_assessment_id) if default_assessment_id in assessment_options else 0

        selected_assessment_id = st.selectbox(
            "Summarization Type",
            options=list(assessment_options.keys()),
            format_func=lambda x: assessment_options.get(x, x),
            index=default_index,
            key="assessment_id_select" # Use unique key
        )

        # Show description
        selected_config = next((cfg for cfg in distill_configs if cfg["id"] == selected_assessment_id), None)
        if selected_config:
            st.info(selected_config.get("description", "No description available."))

        st.divider()
        # Model selection
        st.markdown("### Model & Processing")
        model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"] # Added gpt-4o
        selected_model = st.selectbox("LLM Model", options=model_options, key="model_select")

        # Temperature
        temperature = st.slider("Temperature (Creativity)", min_value=0.0, max_value=1.5, value=0.3, step=0.1,
                                help="Lower values (e.g., 0.2) make output more focused; higher values (e.g., 0.9) make it more creative/random.",
                                key="temperature_slider")

        # Chunking options
        chunk_size = st.slider("Chunk Size", min_value=1000, max_value=16000, value=8000, step=1000, # Increased max range
                               help="Size of text chunks processed by the LLM. Larger chunks = more context but potentially higher cost/latency.",
                               key="chunk_size_slider")
        chunk_overlap = st.slider("Chunk Overlap", min_value=100, max_value=1000, value=300, step=50, # Smaller step
                                help="Amount of text overlap between consecutive chunks to maintain context.",
                                key="chunk_overlap_slider")
        st.divider()
        st.caption("Upload document and options below.")


    # --- Main Content Area ---
    col1, col2 = st.columns([2, 3]) # Ratio for left (config) vs right (results)

    with col1:
        st.markdown("### 1. Upload Document")
        uploaded_file = st.file_uploader("Upload a document (txt, md, pdf, docx)", type=["txt", "md", "pdf", "docx"], key="file_uploader")

        user_summary_options = {} # Initialize dict for options

        if uploaded_file:
            # Load document only if it hasn't been loaded or the file changed
            if not st.session_state.document_loaded or st.session_state.get("uploaded_filename") != uploaded_file.name:
                 with st.spinner("Loading document..."):
                     document, file_path = load_document(uploaded_file)
                 if document:
                     st.session_state.document_loaded = True
                     st.session_state.original_document = document
                     st.session_state.uploaded_filename = uploaded_file.name # Store filename to detect changes
                     # Reset processing state when new file is loaded
                     st.session_state.processing_started = False
                     st.session_state.processing_complete = False
                     st.session_state.summary_result = None
                     st.success(f"Loaded `{uploaded_file.name}` successfully.")
                     st.rerun() # Rerun to update UI with preview etc.
                 else:
                     st.session_state.document_loaded = False
                     st.error("Failed to load document.")

            # Display preview and options if document is loaded
            if st.session_state.document_loaded and st.session_state.original_document:
                display_document_preview(st.session_state.original_document)

                st.markdown("### 2. Select Format & Options")
                display_summary_formats(selected_assessment_id) # Includes rerun on selection
                # Display options and capture the returned dict
                st.session_state.user_summary_options = display_summary_options(selected_assessment_id)

                st.markdown("### 3. Generate Summary")
                # Show Process/Reset buttons only if doc is loaded and not currently processing
                if not st.session_state.processing_started:
                     process_cols = st.columns([3, 1])
                     with process_cols[0]:
                         process_btn_text = "Generate Summary" if not st.session_state.processing_complete else "🔄 Regenerate Summary"
                         if st.button(process_btn_text, type="primary", use_container_width=True):
                             # Prepare processing options dictionary *at the time of button click*
                             processing_options = {
                                 "model": selected_model,
                                 "temperature": temperature,
                                 "chunk_size": chunk_size,
                                 "chunk_overlap": chunk_overlap,
                                 "format": st.session_state.selected_format, # Use current format selection
                                 **(st.session_state.get("user_summary_options", {})) # Merge in user selections
                             }
                             # Store options for the async call to retrieve later if needed
                             st.session_state.current_processing_options = processing_options
                             st.session_state.current_assessment_id = selected_assessment_id

                             # Mark as processing started, reset complete flag and progress
                             st.session_state.processing_started = True
                             st.session_state.processing_complete = False
                             st.session_state.summary_result = None # Clear previous result
                             st.session_state.current_progress = 0.0
                             st.session_state.progress_message = "Initializing..."
                             st.session_state.stages_info = {} # Reset stages info

                             logger.info(f"User initiated processing for {selected_assessment_id}")
                             st.rerun() # Rerun to show progress section

                     with process_cols[1]:
                         if st.button("Reset", use_container_width=True):
                             # Reset relevant session state variables
                             st.session_state.document_loaded = False
                             st.session_state.processing_started = False
                             st.session_state.processing_complete = False
                             st.session_state.summary_result = None
                             st.session_state.original_document = None
                             st.session_state.uploaded_filename = None
                             st.session_state.user_summary_options = {}
                             st.session_state.current_progress = 0.0
                             st.session_state.progress_message = "Not started"
                             st.session_state.current_stage = None
                             st.session_state.stages_info = {}
                             # Clear the file uploader (requires specific key trick usually)
                             # st.query_params.clear() # Might clear too much
                             logger.info("User reset summarizer state.")
                             st.rerun() # Rerun to clear UI


    with col2:
        # Right column: Progress Indicator or Results
        if st.session_state.processing_started and not st.session_state.processing_complete:
            st.markdown("### Processing Document...")
            st.info("Please wait while the AI agents analyze and summarize your document. This may take a few moments...")
            # Render the detailed progress updates
            render_pipeline_status()

            # --- Trigger Asynchronous Processing ---
            # Retrieve necessary data stored in session state
            document_to_process = st.session_state.original_document
            assessment_id_to_process = st.session_state.current_assessment_id
            options_to_process = st.session_state.current_processing_options

            # Placeholder while running async task
            status_placeholder = st.empty()

            # Run the async function. Streamlit handles the loop.
            # This approach is simpler than managing asyncio.run directly in recent Streamlit versions.
            # The script will rerun multiple times, checking state. We only run the async call once.
            # Need a flag to ensure it only runs once per trigger. Add 'is_processing' flag.

            if 'is_currently_processing' not in st.session_state:
                 st.session_state.is_currently_processing = True # Set flag
                 try:
                      logger.info(f"Triggering asyncio.run for process_document: {assessment_id_to_process}")
                      status_placeholder.warning("🤖 AI agents are working...")
                      # This will block this specific execution path until completion
                      asyncio.run(process_document(document_to_process, assessment_id_to_process, options_to_process))

                 except Exception as async_e:
                      logger.error(f"Exception during asyncio.run(process_document): {async_e}", exc_info=True)
                      st.session_state.summary_result = None
                      st.session_state.processing_complete = True # Mark complete even on outer error
                      st.error(f"A critical error occurred during processing: {async_e}")

                 finally:
                      # Processing attempt finished, reset the flag
                      del st.session_state['is_currently_processing']
                      # Rerun one last time to display results or final error state
                      logger.info("Async processing attempt finished. Rerunning script.")
                      st.rerun()


        elif st.session_state.processing_complete:
            # Show Results Section
            st.markdown("### Summary Results")
            if st.session_state.summary_result:
                # Display the formatted summary
                display_summary_result(st.session_state.summary_result, st.session_state.selected_format)
            else:
                st.error("Processing completed, but failed to generate a summary result. Please check logs or try adjusting settings.")
                # Optionally show error details from context if captured

        else:
            # Initial state - Prompt user to upload
            st.info("⬆️ Upload a document and configure options in the sidebar to start.")


if __name__ == "__main__":
    main()