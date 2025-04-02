# app.py - Revised Version
import os
import sys
import streamlit as st
import logging
from pathlib import Path
from datetime import datetime
import openai # Keep openai import if CustomLLM relies on it directly
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("beyond-notes.log") # Ensure this file path is appropriate
    ]
)
# Use __name__ for the main script logger, a common convention
logger = logging.getLogger(__name__)

# Add project root to path to ensure imports work correctly
# Assumes app.py is in the project root directory
sys.path.insert(0, str(Path(__file__).parent))

# Import project components
try:
    from assessments.loader import AssessmentLoader
    from utils.paths import AppPaths
    from core.llm.customllm import CustomLLM
except ImportError as e:
    logger.error(f"Failed to import project components: {e}. Ensure PYTHONPATH is correct or app.py is in the project root.", exc_info=True)
    st.error(f"Application Error: Failed to load necessary components. Please check logs. Error: {e}")
    st.stop() # Stop execution if core components can't be imported


# Page configuration
st.set_page_config(
    page_title="Beyond Notes",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_app():
    """Initialize application components and session state."""
    try:
        # Ensure necessary directories exist
        AppPaths.ensure_dirs()

        # Initialize assessment loader if not already in session state
        if "assessment_loader" not in st.session_state:
            logger.info("Initializing assessment loader")
            loader = AssessmentLoader() # Assumes constructor requires no args

            # Create default types if needed (for first-time run)
            # Use Path object for checking existence
            types_dir = Path(AppPaths.get_types_dir())
            if not any(types_dir.glob("*.json")):
                logger.info("Creating default assessment types as none were found.")
                loader.create_default_types()
                loader.reload() # Reload after creating defaults

            st.session_state.assessment_loader = loader
            logger.info("Assessment loader initialized and loaded.")

        # Check for API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            st.session_state.has_api_key = False
            logger.warning("OPENAI_API_KEY environment variable not set.")
        else:
            st.session_state.has_api_key = True

            # Initialize LLM instance for shared usage if API key exists
            if "llm" not in st.session_state:
                # Keeping model hardcoded as per user feedback for now
                model = os.environ.get("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
                logger.info(f"Initializing LLM with model: {model}")
                # Add error handling for LLM initialization if needed
                st.session_state.llm = CustomLLM(api_key, model=model)
                logger.info(f"LLM instance created for model {model}")

    except Exception as e:
        logger.error(f"Error during application initialization: {e}", exc_info=True)
        st.error(f"Application Initialization Failed: {e}. Please check the logs.")
        # Depending on severity, you might call st.stop() here

def show_welcome_page():
    """Display the main landing page content."""
    st.title("📝 Beyond Notes")
    st.markdown("### *When your meeting transcripts deserve more than just a summary*")

    # Introduction
    st.markdown("""
    **Beyond Notes** transforms meeting transcripts into structured, actionable insights using multi-agent AI.
    It's what happens when specialized AI agents collaborate on understanding your meetings instead of asking
    a single model to do everything at once.
    """)

    # API key check
    if not st.session_state.get("has_api_key", False):
        st.warning(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable "
            "or add it to your `.env` file to use Beyond Notes."
        )
        # Optionally disable buttons or show limited functionality if key is missing
        # return # Or just show the warning and continue

    # Main capabilities section
    st.subheader("Core Capabilities")

    col1, col2, col3, col4 = st.columns(4)

    # --- Revised Button Logic: Set state *before* switching page ---
    with col1:
        st.markdown("### 🔍 Distill")
        st.markdown(
            "Create concise, focused summaries of meeting content with "
            "varying detail levels and formats."
        )
        if st.button("Summarize a Document", key="btn_distill", disabled=not st.session_state.get("has_api_key", False)):
            st.session_state.default_assessment = "distill" # Set state first
            st.switch_page("pages/02_Analyze.py")

    with col2:
        st.markdown("### 📋 Extract")
        st.markdown(
            "Identify and organize action items, owners, and deadlines "
            "from meeting transcripts."
        )
        if st.button("Extract Action Items", key="btn_extract", disabled=not st.session_state.get("has_api_key", False)):
            st.session_state.default_assessment = "extract" # Set state first
            st.switch_page("pages/02_Analyze.py")

    with col3:
        st.markdown("### ⚠️ Assess")
        st.markdown(
            "Uncover issues, challenges, and risks with severity ratings "
            "and recommendations."
        )
        if st.button("Identify Issues", key="btn_assess", disabled=not st.session_state.get("has_api_key", False)):
            st.session_state.default_assessment = "assess" # Set state first
            st.switch_page("pages/02_Analyze.py")

    with col4:
        st.markdown("### 📊 Analyze")
        st.markdown(
            "Evaluate content against structured frameworks like "
            "readiness assessments."
        )
        if st.button("Analyze Framework", key="btn_analyze", disabled=not st.session_state.get("has_api_key", False)):
            st.session_state.default_assessment = "analyze" # Set state first
            st.switch_page("pages/02_Analyze.py")

    # How it works section
    st.subheader("How It Works")

    st.markdown("""
    Beyond Notes uses a multi-agent architecture where specialized AI agents collaborate:

    1.  **🧭 Planner Agent** - Creates a tailored analysis strategy.
    2.  **🔍 Extractor Agent** - Identifies relevant information from text chunks.
    3.  **🧩 Aggregator Agent** - Combines findings and removes duplicates intelligently.
    4.  **⚖️ Evaluator Agent** - Assesses importance, assigns ratings/scores based on evidence.
    5.  **📊 Formatter Agent** - Transforms insights into structured, readable reports.

    The shared `ProcessingContext` enables agents to build upon each other's work, tracking metrics and evidence throughout the pipeline.
    """) # Updated agent names based on our discussion

    # Recent analyses section
    st.subheader("Recent Analyses")

    # --- Revised Recent Analyses Logic: Dynamic assessment types ---
    recent_files = []
    if "assessment_loader" in st.session_state:
        assessment_loader = st.session_state.assessment_loader
        try:
            # --- !!! Assumption: Replace 'get_available_type_names' with your actual method name !!! ---
            available_types = assessment_loader.get_available_type_names()
            logger.debug(f"Dynamically loaded assessment types for recent files: {available_types}")

            for assessment_type in available_types: # Use dynamic list
                output_dir = AppPaths.get_assessment_output_dir(assessment_type)
                if output_dir.exists():
                    # Scan for markdown reports, adjust glob pattern if needed
                    for file_path in output_dir.glob(f"{assessment_type}_report_*.md"):
                        try:
                            stat = file_path.stat()
                            recent_files.append({
                                "path": file_path,
                                "type": assessment_type,
                                "filename": file_path.name,
                                "modified": datetime.fromtimestamp(stat.st_mtime)
                            })
                        except OSError as e:
                             logger.warning(f"Could not stat file {file_path}: {e}")


        except AttributeError:
            logger.error("Failed to get assessment types dynamically. AssessmentLoader might be missing the expected method ('get_available_type_names'). Falling back to hardcoded list.")
            # Fallback to hardcoded list if dynamic loading fails
            available_types = ["distill", "extract", "assess", "analyze"]
             # Duplicate the scanning logic here for the fallback or refactor into a helper function
            for assessment_type in available_types:
                output_dir = AppPaths.get_assessment_output_dir(assessment_type)
                if output_dir.exists():
                    for file_path in output_dir.glob(f"{assessment_type}_report_*.md"):
                         try:
                            stat = file_path.stat()
                            recent_files.append({
                                "path": file_path, "type": assessment_type, "filename": file_path.name,
                                "modified": datetime.fromtimestamp(stat.st_mtime)
                            })
                         except OSError as e:
                             logger.warning(f"Could not stat file {file_path} (fallback): {e}")

        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching recent analysis files: {e}", exc_info=True)
            st.error("Could not load recent analyses due to an error.")

    else:
        st.info("Assessment loader not available in session state.")


    # Sort by modification time (most recent first)
    recent_files.sort(key=lambda x: x["modified"], reverse=True)

    if recent_files:
        st.markdown("Click on a past analysis to view the report (functionality to be added).") # Placeholder action
        # Display recent files (up to 5)
        for i, file_info in enumerate(recent_files[:5]):
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                # Make filename clickable (if view page exists)
                 # st.page_link("pages/04_ViewReport.py", label=f"{file_info['filename']}", icon="📄", query_params={"report_path": str(file_info['path'])})
                 st.write(f"📄 {file_info['filename']}") # Placeholder if view page doesn't exist yet
            with col2:
                st.caption(f"Type: {file_info['type'].title()}")
            with col3:
                st.caption(f"{file_info['modified'].strftime('%Y-%m-%d %H:%M')}")

            if i < 4 and i < len(recent_files) -1 : # Ensure divider doesn't appear after the last item shown
                st.divider()
    else:
        st.info("No analyses have been performed yet. Get started by analyzing a document!")

# Main app execution
def main():
    # Initialize app components (runs on first load and potentially reruns)
    initialize_app()

    # Display the welcome page content
    show_welcome_page()

if __name__ == "__main__":
    main()