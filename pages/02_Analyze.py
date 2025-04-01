# pages/02_Analyze.py
import os
import sys
import asyncio
import streamlit as st
import logging
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import copy
from typing import Dict, Any, Optional, Tuple, List
from utils.result_accessor import get_assessment_data, get_item_count, debug_result_structure
from utils.formatting import format_assessment_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Project Setup ---
try:
    APP_ROOT = Path(__file__).parent.parent
    if str(APP_ROOT) not in sys.path:
        sys.path.insert(0, str(APP_ROOT))
    from core.models.document import Document
    from core.orchestrator import Orchestrator
    from assessments.loader import AssessmentLoader # Uses the NEW loader
    from utils.paths import AppPaths
    from utils.formatting import (
        format_assessment_report,
        display_pipeline_progress,
        save_result_to_output
    )
except ImportError as e:
     st.error(f"Failed to import necessary modules. Ensure project structure is correct and requirements are installed. Error: {e}")
     st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Analyze Document - Beyond Notes",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Async Processing Function ---
# run_analysis_pipeline function remains the same as the previous version
async def run_analysis_pipeline(document: Document, assessment_id: str, options: Dict[str, Any]):
    """
    Initializes and runs the Orchestrator for the given document and settings.

    Args:
        document: The Document object to analyze.
        assessment_id: The ID of the assessment configuration to use.
        options: Runtime options (model, chunking, user selections).

    Returns:
        The analysis result dictionary, or None if an error occurred.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        logger.error("OPENAI_API_KEY not found in environment.")
        return None

    orchestrator = None # Initialize to None
    try:
        # Store orchestrator in session state IF needed by callback immediately
        # Otherwise, initialize locally first
        orchestrator = Orchestrator(
            assessment_id=assessment_id,
            options=options,
            api_key=api_key
        )
        st.session_state.orchestrator_instance = orchestrator # Store for callback access if needed
    except ValueError as e:
         st.error(f"Error initializing assessment '{assessment_id}': {e}. Please check configuration.")
         logger.error(f"Orchestrator init failed for '{assessment_id}': {e}", exc_info=True)
         return None
    except Exception as e:
        st.error(f"An unexpected error occurred during orchestrator setup: {e}")
        logger.error(f"Unexpected orchestrator init error: {e}", exc_info=True)
        return None

    progress_container = st.container()
    with progress_container:
        status_text = st.empty()
        progress_bar = st.progress(0.0)
        pipeline_status_display = st.empty()

    start_time = time.time()

    def update_progress_display(progress: float, message: str):
        try:
            st.session_state.last_progress = progress # Store progress for potential error state
            progress_bar.progress(min(progress, 1.0))
            status_text.text(message)
            # Use the stored orchestrator instance if available
            orch = st.session_state.get('orchestrator_instance')
            if orch:
                 progress_data = orch.get_progress()
                 pipeline_status_display.markdown(
                      display_pipeline_progress(progress_data.get("stages", {})),
                      unsafe_allow_html=True
                 )
            else:
                 pipeline_status_display.markdown("Pipeline status: Initializing...")
        except Exception as cb_e:
            logger.warning(f"Error updating progress UI: {cb_e}", exc_info=False)

    result = None
    try:
        update_progress_display(0.0, "Starting analysis...")
        result = await orchestrator.process_with_progress(
            document=document,
            progress_callback=update_progress_display
        )
        if orchestrator.context:
            st.session_state.processing_context = orchestrator.context
            logger.info("Stored processing context in session state.")
    except Exception as e:
        st.error(f"An error occurred during document processing: {str(e)}")
        logger.error(f"Document processing pipeline error: {str(e)}", exc_info=True)
        update_progress_display(st.session_state.get('last_progress', 1.0), f"Error: {str(e)}")
        return None
    finally:
        # time.sleep(0.5) # Optional delay
        progress_bar.empty()
        status_text.empty()
        pipeline_status_display.empty()
        if 'orchestrator_instance' in st.session_state:
            del st.session_state.orchestrator_instance # Clean up instance

    end_time = time.time()
    logger.info(f"Document processing finished in {end_time - start_time:.2f} seconds.")

    if not isinstance(result, dict) or "metadata" not in result:
         logger.error(f"Processing completed but returned unexpected result format: {type(result)}")
         st.error("Processing finished, but the result format is unexpected. Check logs.")
         return None

    return result

# --- UI Rendering Functions ---
# render_user_options function remains the same as the previous version
def render_user_options(assessment_id: str, assessment_loader: AssessmentLoader) -> Dict[str, Any]:
    """Renders Streamlit widgets for user-configurable options based on the assessment config."""
    runtime_options = {}
    logger.debug(f"Rendering user options for assessment_id: {assessment_id}")

    try:
        config = assessment_loader.load_config(assessment_id)
        if not config:
            st.warning(f"Could not load config for '{assessment_id}' to render options.")
            return {}

        user_options_schema = config.get("user_options", {})
        if not user_options_schema:
            # No options to render for this assessment type
            return {}

        st.subheader("Assessment Options") # Keep the subheader here
        # --- RENDER WIDGETS (same logic as before) ---
        for option_key, option_data in user_options_schema.items():
            if not isinstance(option_data, dict) or "type" not in option_data:
                logger.warning(f"Skipping invalid user option schema entry: {option_key}")
                continue

            option_type = option_data.get("type")
            display_name = option_data.get("display_name", option_key.replace("_", " ").title())
            description = option_data.get("description", "")
            default = option_data.get("default")
            required = option_data.get("required", False)
            label = f"{display_name}{' *' if required else ''}"

            try:
                 if option_type == "select":
                     choices = option_data.get("options", {})
                     options_list = list(choices.keys())
                     default_index = options_list.index(default) if default in options_list else 0
                     selected_value = st.selectbox(
                         label, options=options_list, index=default_index,
                         help=description, format_func=lambda x: choices.get(x, x)
                     )
                     runtime_options[option_key] = selected_value
                 elif option_type == "multi_select":
                    choices_config = option_data.get("options", [])
                    options_list = []
                    if choices_config == "dynamic_from_dimensions" and config.get("assessment_type") == "analyze":
                        framework_def = config.get("framework_definition", {})
                        options_list = [d.get("name", f"dim_{i}") for i, d in enumerate(framework_def.get("dimensions", []))]
                    elif isinstance(choices_config, list):
                        options_list = choices_config
                    else:
                         logger.warning(f"Unsupported options format for multi-select '{option_key}': {choices_config}")
                    selected_values = st.multiselect(label, options=options_list, default=default if isinstance(default, list) else [], help=description)
                    runtime_options[option_key] = selected_values
                 elif option_type == "boolean":
                     runtime_options[option_key] = st.checkbox(label, value=bool(default), help=description)
                 elif option_type == "text":
                      runtime_options[option_key] = st.text_input(label, value=str(default) if default is not None else "", help=description)
                 elif option_type == "textarea":
                      runtime_options[option_key] = st.text_area(label, value=str(default) if default is not None else "", height=100, help=description) # Reduced height slightly
                 elif option_type == "number":
                      runtime_options[option_key] = st.number_input(label, value=default, help=description)
                 elif option_type == "form":
                     with st.container():
                          st.markdown(f"**{display_name}**{':' if description else ''}")
                          if description: st.caption(description)
                          form_data = {}
                          fields = option_data.get("fields", {})
                          for field_key, field_data in fields.items():
                                if not isinstance(field_data, dict): continue
                                field_type = field_data.get("type", "text")
                                field_required = field_data.get("required", False)
                                field_name = field_data.get("display_name", field_key.replace("_", " ").title())
                                field_desc = field_data.get("description", "")
                                field_default = field_data.get("default")
                                field_label = f"{field_name}{' *' if field_required else ''}"
                                # Use unique keys for widgets within forms
                                widget_key=f"form_{option_key}_{field_key}"
                                if field_type == "text":
                                     form_data[field_key] = st.text_input(field_label, value=str(field_default) if field_default is not None else "", help=field_desc, key=widget_key)
                                elif field_type == "textarea":
                                     form_data[field_key] = st.text_area(field_label, value=str(field_default) if field_default is not None else "", help=field_desc, key=widget_key)
                                elif field_type == "number":
                                     form_data[field_key] = st.number_input(field_label, value=field_default, help=field_desc, key=widget_key)
                          runtime_options[option_key] = form_data
                          # Removed the separator from inside the loop, place outside if needed
                     # st.markdown("---") # Optional separator after the whole form
                 else:
                      st.warning(f"Unsupported user option type '{option_type}' for key '{option_key}'")
            except Exception as widget_e:
                 st.error(f"Error rendering widget for option '{option_key}': {widget_e}")
                 logger.error(f"Widget rendering error for '{option_key}': {widget_e}", exc_info=True)

    except Exception as e:
        st.error(f"An unexpected error occurred while rendering user options: {str(e)}")
        logger.error(f"Error rendering user options: {str(e)}", exc_info=True)

    return runtime_options


# --- Main Application Logic ---
def main():
    st.title("🔍 Analyze Document")
    st.caption("Upload a document and select an assessment configuration to extract insights.")

    # --- Initialization ---
    try: AppPaths.ensure_dirs()
    except Exception as e: st.error(f"Failed to ensure application directories: {e}"); logger.error(f"Directory setup error: {e}", exc_info=True); st.stop()

    if "assessment_loader" not in st.session_state:
        try:
            st.session_state.assessment_loader = AssessmentLoader()
            logger.info("AssessmentLoader initialized and added to session state.")
        except Exception as e: st.error(f"Failed to initialize Assessment Loader: {e}. Check configuration files."); logger.error(f"AssessmentLoader init error: {e}", exc_info=True); st.stop()
    loader = st.session_state.assessment_loader

    if not os.environ.get("OPENAI_API_KEY"): st.warning("OpenAI API key not set in environment variables.")

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("Configuration")
        all_configs_list = loader.get_assessment_configs_list()
        if not all_configs_list: st.error("No assessment configurations found!"); st.stop()
        config_options = {cfg['id']: cfg['display_name'] for cfg in all_configs_list}
        available_ids = list(config_options.keys())
        default_config_id = available_ids[0]
        if 'selected_assessment_id' in st.session_state and st.session_state.selected_assessment_id in available_ids: default_config_id = st.session_state.selected_assessment_id
        try: default_index = available_ids.index(default_config_id)
        except ValueError: default_index = 0

        # Store selected ID in session state immediately upon selection
        st.session_state.selected_assessment_id = st.selectbox(
            "Select Assessment Configuration", options=available_ids, index=default_index,
            format_func=lambda id: config_options.get(id, id),
            help="Choose a base analysis type or a saved custom template.",
            key="sb_assessment_id" # Add key for stability if needed
        )
        selected_assessment_id = st.session_state.selected_assessment_id # Use the value from session state

        selected_config_info = next((cfg for cfg in all_configs_list if cfg['id'] == selected_assessment_id), None)
        if selected_config_info:
            st.caption(f"Type: {selected_config_info['assessment_type']} | {'Template' if selected_config_info['is_template'] else 'Base Type'}")
            with st.expander("Description"): st.write(selected_config_info.get('description', 'No description.'))
        st.divider()
        model = st.selectbox("LLM Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], index=0, help="Select OpenAI model.")
        st.divider()
        st.subheader("Processing Options")
        chunk_size = st.slider("Chunk Size", 1000, 16000, 8000, 500, help="Max characters per chunk.")
        chunk_overlap = st.slider("Chunk Overlap", 0, 1000, 300, 50, help="Characters overlapping.")

    # --- Main Area ---
    tab_upload, tab_results = st.tabs(["📤 Upload & Analyze", "📊 Results"])

    with tab_upload:
        col_upload, col_sample = st.columns(2)
        with col_upload: uploaded_file = st.file_uploader("Upload (.txt, .md)", type=["txt", "md"])
        with col_sample:
             sample_dir = APP_ROOT / "data" / "samples"
             sample_files = []
             if sample_dir.exists(): sample_files = sorted([f.name for f in sample_dir.glob("*.txt")]) + sorted([f.name for f in sample_dir.glob("*.md")])
             selected_sample = st.selectbox("Or select a sample", [""] + sample_files)

        document: Optional[Document] = None
        # Simplified Document Loading Logic
        try:
            if uploaded_file:
                 document = Document.from_uploaded_file(uploaded_file)
                 st.success(f"Loaded: {document.filename}")
            elif selected_sample:
                 file_path = sample_dir / selected_sample
                 with open(file_path, 'r', encoding='utf-8') as f: doc_text = f.read()
                 document = Document(text=doc_text, filename=selected_sample)
                 st.success(f"Loaded: {document.filename}")
        except Exception as e:
             st.error(f"Error loading document: {e}")
             logger.error(f"Error loading document: {e}", exc_info=True)


        if document:
            # Display Doc Info
            st.markdown("---")
            st.subheader("📄 Document Details")
            col1, col2, col3 = st.columns(3)
            col1.metric("Filename", document.filename)
            col2.metric("Word Count", f"{document.word_count:,}")
            col3.metric("Est. Tokens", f"{document.estimated_tokens:,}")
            with st.expander("Preview Content"): st.text_area("Preview", document.text[:2000] + ("..." if len(document.text) > 2000 else ""), height=200, disabled=True)

            # ================== KEY CHANGE HERE ==================
            # Render options and Analyze button *after* document is loaded and info displayed
            st.markdown("---")
            custom_runtime_options = render_user_options(selected_assessment_id, loader) # Render options now
            st.markdown("---")

            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                runtime_options = {"model": model, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap, **custom_runtime_options}
                logger.info(f"Starting analysis with ID: '{selected_assessment_id}', Options: {runtime_options}")
                # Clear previous results before starting new analysis
                st.session_state.assessment_result = None
                st.session_state.processing_context = None
                st.session_state.assessment_type = None # Clear derived type too

                # --- Trigger Analysis ---
                with st.spinner("Analyzing document... Please wait."): # Use spinner context manager
                    result_data = asyncio.run(run_analysis_pipeline(document, selected_assessment_id, runtime_options))

                if result_data:
                    st.session_state.document = document
                    st.session_state.assessment_result = result_data
                    st.session_state.assessment_type = result_data.get("metadata",{}).get("assessment_type", "unknown")
                    logger.info(f"Analysis complete. Result keys: {list(result_data.keys())}")
                    st.success("✅ Analysis Complete!")
                    st.info("View detailed results in the '📊 Results' tab.")
                    try: save_result_to_output(result_data, st.session_state.assessment_type, document.filename)
                    except Exception as save_e: st.warning(f"Could not automatically save result file: {save_e}")
                else:
                     logger.warning("Analysis process did not return a valid result.")
                     st.warning("Analysis did not complete successfully. Check status messages above and logs for details.")
            # ====================================================

        else:
            st.info("Upload a document or select a sample to begin analysis.")


    with tab_results:
        # --- Results Tab Logic (mostly unchanged, relies on session state) ---
        if "assessment_result" not in st.session_state or st.session_state.assessment_result is None:
            st.info("Analyze a document first to view results here.")
        else:
            result = st.session_state.assessment_result
            doc = st.session_state.document
            assessment_type = st.session_state.assessment_type

            st.header(f"📊 Assessment Results: {result.get('metadata',{}).get('assessment_display_name', assessment_type)}")
            st.caption(f"Document: {doc.filename}")

            meta = result.get("metadata", {})
            stats = result.get("statistics", {})
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Processing Time", f"{meta.get('total_processing_time', meta.get('processing_time_seconds', 0)):.1f}s") # Check both keys
            col2.metric("LLM Tokens Used", f"{stats.get('total_llm_tokens_used', 0):,}")
            col3.metric("LLM Calls", f"{stats.get('total_llm_calls', 0)}")
            # Dynamic Key Metric
            key_metric_value = "N/A"; key_metric_label = "Items Found"
            counts = stats.get("extraction_counts", {})
            if assessment_type == "extract": key_metric_label, key_metric_value = "Action Items", counts.get("action_items", 0)
            elif assessment_type == "assess": key_metric_label, key_metric_value = "Issues Found", counts.get("issues", 0)
            elif assessment_type == "distill": key_metric_label, key_metric_value = "Key Points", counts.get("key_points", 0)
            col4.metric(key_metric_label, key_metric_value)

            st.markdown("---")
            tab_report, tab_pipeline, tab_data, tab_debug = st.tabs(["📄 Report", "⚙️ Pipeline", "Raw Data", "🐞 Debug"])

            with tab_report:
                # (Report display and download logic remains the same)
                st.subheader("Formatted Report")
                try:
                    formatted_report_content = format_assessment_report(result.get("result", {}), assessment_type)
                    st.markdown(formatted_report_content, unsafe_allow_html=True)
                    st.markdown("---"); col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                         json_string = json.dumps(result, indent=2, default=str); fname = f"{Path(doc.filename).stem}_{assessment_type}_result.json"
                         col_dl1.download_button("💾 Download Full Result (JSON)", json_string, fname, "application/json", use_container_width=True)
                    with col_dl2:
                         fname_md = f"{Path(doc.filename).stem}_{assessment_type}_report.md"
                         col_dl2.download_button("📄 Download Report (Markdown)", formatted_report_content, fname_md, "text/markdown", use_container_width=True)
                except Exception as format_e: st.error(f"Error displaying report: {format_e}"); logger.error(f"Report display error: {format_e}", exc_info=True); st.json(result.get("result", {"Error": "No result data"}))

            with tab_pipeline:
                # (Pipeline display logic remains the same)
                st.subheader("Processing Pipeline Stages")
                stages = meta.get("stages", {})
                if stages:
                    st.markdown(display_pipeline_progress(stages), unsafe_allow_html=True)
                    durations = {name: data['duration'] for name, data in stages.items() if data.get('duration') is not None}
                    if durations:
                        try: # Chart generation
                            fig, ax = plt.subplots(); sorted_stages = sorted(stages.items(), key=lambda item: item[1].get('start_time', float('inf'))); sorted_names = [s[0] for s in sorted_stages if s[0] in durations]; sorted_durations = [durations[name] for name in sorted_names]
                            y_pos = range(len(sorted_names)); ax.barh(y_pos, sorted_durations, align='center'); ax.set_yticks(y_pos, labels=sorted_names); ax.invert_yaxis(); ax.set_xlabel("Duration (seconds)"); ax.set_title("Stage Processing Time"); st.pyplot(fig)
                        except Exception as chart_e: st.error(f"Chart error: {chart_e}"); logger.warning(f"Chart error: {chart_e}", exc_info=True)
                else: st.info("No pipeline stage info available.")

            with tab_data:
                # (Raw data display logic remains the same)
                st.subheader("Explore Processed Data")
                if result.get("result"):
                     with st.expander("Formatted Result Output", expanded=True): st.json(result["result"])
                if result.get("statistics"):
                     with st.expander("Processing Statistics"): st.json(result["statistics"])
                # Check context directly for review output as it might not be in main 'result' dict
                if "review_output" in st.session_state.get("processing_context", {}).results:
                     with st.expander("Reviewer Feedback"): st.json(st.session_state.processing_context.results["review_output"])
                if meta.get("errors"):
                     with st.expander("Processing Errors", expanded=True): st.error("Errors occurred:"); st.json(meta["errors"])
                if meta.get("warnings"):
                      with st.expander("Processing Warnings"): st.warning("Warnings generated:"); st.json(meta["warnings"])


            with tab_debug:
                # (Debug display logic remains the same)
                 st.subheader("Debugging Information")
                 st.warning("Contains potentially large amounts of raw data.")
                 with st.expander("Full Result JSON"): st.json(result)
                 if "processing_context" in st.session_state:
                      with st.expander("Processing Context State (Summary)"):
                           try: context_dict = st.session_state.processing_context.to_dict(include_document=False, include_chunks=False); st.json(context_dict)
                           except Exception as context_e: st.error(f"Could not serialize Processing Context: {context_e}")
                 else: st.info("Processing context not stored in session state.")

            st.markdown("---"); st.subheader("Next Steps")
            if "processing_context" in st.session_state:
                 st.success("✅ Analysis context available.")
                 if st.button("💬 Go to Chat Page", use_container_width=True): st.switch_page("pages/03_Chat.py")
            else: st.warning("Processing context not available for Chat.")

# --- Entry Point ---
if __name__ == "__main__":
    # Initialize session state keys needed by this page if they don't exist
    keys_to_init = ["assessment_loader", "selected_assessment_id", "assessment_result",
                    "document", "assessment_type", "processing_context", "last_progress"]
    for key in keys_to_init:
        if key not in st.session_state:
            st.session_state[key] = None
    main()