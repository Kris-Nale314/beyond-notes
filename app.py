# app.py - Main entry point for Beyond Notes application
import os
import sys
import streamlit as st
import logging
from pathlib import Path
from datetime import datetime
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("beyond-notes.log")
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path to ensure imports work correctly
sys.path.insert(0, str(Path(__file__).parent))

# Import project components
try:
    from assessments.loader import AssessmentLoader
    from utils.paths import AppPaths
    from core.llm.customllm import CustomLLM
    
    # Try to import the new summary renderer
    try:
        from core.models.summary_renderer import render_summary
        logger.info("Successfully imported summary renderer module")
    except ImportError as e:
        logger.warning(f"Could not import summary renderer: {e}. Enhanced summary visualization may not be available.")
except ImportError as e:
    logger.error(f"Failed to import project components: {e}. Ensure PYTHONPATH is correct or app.py is in the project root.", exc_info=True)
    st.error(f"Application Error: Failed to load necessary components. Please check logs. Error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Beyond Notes",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for the app
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .card-container {
        background-color: rgba(0, 0, 0, 0.1);
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .card-header {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.7rem;
        color: white;
    }
    .card-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .card-desc {
        opacity: 0.85;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .section-header {
        font-size: 1.7rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: white;
    }
    .agent-container {
        background-color: rgba(0, 0, 0, 0.05);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 3px solid #2196F3;
    }
    .agent-name {
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .recent-file {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 0.8rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    .recent-file-name {
        font-weight: 500;
    }
    .file-meta {
        opacity: 0.7;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_app():
    """Initialize application components and session state."""
    try:
        # Ensure necessary directories exist
        AppPaths.ensure_dirs()

        # Initialize assessment loader if not already in session state
        if "assessment_loader" not in st.session_state:
            logger.info("Initializing assessment loader")
            loader = AssessmentLoader()

            # Create default types if needed (for first-time run)
            types_dir = Path(AppPaths.get_types_dir())
            if not any(types_dir.glob("*.json")):
                logger.info("Creating default assessment types as none were found.")
                loader.create_default_types()
                loader.reload()

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
                model = os.environ.get("OPENAI_MODEL_NAME", "gpt-4")  # Default to GPT-4 for better results
                logger.info(f"Initializing LLM with model: {model}")
                st.session_state.llm = CustomLLM(api_key, model=model)
                logger.info(f"LLM instance created for model {model}")
                
        # Initialize summary format preference if not set
        if "preferred_summary_format" not in st.session_state:
            st.session_state.preferred_summary_format = "executive"
            logger.info("Set default summary format preference to 'executive'")

    except Exception as e:
        logger.error(f"Error during application initialization: {e}", exc_info=True)
        st.error(f"Application Initialization Failed: {e}. Please check the logs.")
        st.stop()

def show_welcome_page():
    """Display the main landing page content."""
    st.markdown('<div class="main-header">📝 Beyond Notes</div>', unsafe_allow_html=True)
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

    # Main capabilities section
    st.markdown('<div class="section-header">Core Capabilities</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown('<div class="card-icon">🔍</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Distill</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-desc">Create concise, focused summaries of meeting content with '
            'varying detail levels and formats.</div>', 
            unsafe_allow_html=True
        )
        if st.button("Summarize a Document", key="btn_distill", disabled=not st.session_state.get("has_api_key", False)):
            st.session_state.default_assessment = "distill"
            # Switch to our new specialized summarizer page
            st.switch_page("pages/01_Summarizer.py")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown('<div class="card-icon">📋</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Extract</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-desc">Identify and organize action items, owners, and deadlines '
            'from meeting transcripts.</div>', 
            unsafe_allow_html=True
        )
        if st.button("Extract Action Items", key="btn_extract", disabled=not st.session_state.get("has_api_key", False)):
            st.session_state.default_assessment = "extract"
            st.switch_page("pages/02_Analyze.py")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown('<div class="card-icon">⚠️</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Assess</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-desc">Uncover issues, challenges, and risks with severity ratings '
            'and recommendations.</div>', 
            unsafe_allow_html=True
        )
        if st.button("Identify Issues", key="btn_assess", disabled=not st.session_state.get("has_api_key", False)):
            st.session_state.default_assessment = "assess"
            st.switch_page("pages/02_Analyze.py")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown('<div class="card-icon">📊</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Analyze</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-desc">Evaluate content against structured frameworks like '
            'readiness assessments.</div>', 
            unsafe_allow_html=True
        )
        if st.button("Analyze Framework", key="btn_analyze", disabled=not st.session_state.get("has_api_key", False)):
            st.session_state.default_assessment = "analyze"
            st.switch_page("pages/02_Analyze.py")
        st.markdown('</div>', unsafe_allow_html=True)

    # How it works section
    st.markdown('<div class="section-header">How It Works</div>', unsafe_allow_html=True)

    st.markdown("""
    Beyond Notes uses a multi-agent architecture where specialized AI agents collaborate:
    """)
    
    agents = [
        ("🧭 Planner Agent", "Creates a tailored analysis strategy based on document type and content"),
        ("🔍 Extractor Agent", "Identifies relevant information from text chunks with precision"),
        ("🧩 Aggregator Agent", "Combines findings and removes duplicates intelligently"),
        ("⚖️ Evaluator Agent", "Assesses importance, assigns ratings/scores based on evidence"),
        ("📊 Formatter Agent", "Transforms insights into structured, readable reports")
    ]
    
    for icon_name, description in agents:
        st.markdown(f'<div class="agent-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="agent-name">{icon_name}</div>', unsafe_allow_html=True)
        st.markdown(f'{description}', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    The shared `ProcessingContext` enables agents to build upon each other's work, tracking metrics 
    and evidence throughout the pipeline.
    """)

    # Recent analyses section
    st.markdown('<div class="section-header">Recent Analyses</div>', unsafe_allow_html=True)

    # Get recent files
    recent_files = []
    if "assessment_loader" in st.session_state:
        assessment_loader = st.session_state.assessment_loader
        try:
            standard_types = assessment_loader.get_standard_assessment_type_names()
            logger.debug(f"Dynamically loaded standard assessment types for recent files: {standard_types}")

            for assessment_type in standard_types:
                output_dir = AppPaths.get_assessment_output_dir(assessment_type)
                if output_dir.exists():
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
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching recent analysis files: {e}", exc_info=True)
            # Fallback to hardcoded list if dynamic loading fails
            standard_types = ["distill", "extract", "assess", "analyze"]
            for assessment_type in standard_types:
                output_dir = AppPaths.get_assessment_output_dir(assessment_type)
                if output_dir.exists():
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
                             logger.warning(f"Could not stat file {file_path} (fallback): {e}")
    else:
        st.error("Assessment loader not available in session state. Application may not have initialized correctly.")

    # Sort by modification time (most recent first)
    recent_files.sort(key=lambda x: x["modified"], reverse=True)

    if recent_files:
        st.markdown("Previously analyzed documents:")
        
        # Display recent files (up to 5) with improved formatting
        for i, file_info in enumerate(recent_files[:5]):
            st.markdown(f'<div class="recent-file">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f'<div class="recent-file-name">📄 {file_info["filename"]}</div>', unsafe_allow_html=True)
                file_type = file_info["type"].capitalize()
                st.markdown(f'<div class="file-meta">Type: {file_type}</div>', unsafe_allow_html=True)
            
            with col2:
                date_str = file_info["modified"].strftime("%b %d, %H:%M")
                st.markdown(f'<div class="file-meta" style="text-align: right;">{date_str}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
    elif "assessment_loader" in st.session_state:
        st.info("No analyses have been performed yet. Get started by analyzing a document!")

# Main app execution
def main():
    # Initialize app components
    initialize_app()

    # Display the welcome page content
    if "assessment_loader" in st.session_state:
        show_welcome_page()
    else:
        st.error("Application failed to initialize properly. Please check the logs or environment configuration.")

if __name__ == "__main__":
    main()