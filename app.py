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
logger = logging.getLogger("beyond-notes")

# Add project root to path to ensure imports work correctly
sys.path.insert(0, str(Path(__file__).parent))

# Import project components
from assessments.loader import AssessmentLoader
from utils.paths import AppPaths
from core.llm.customllm import CustomLLM


# Page configuration
st.set_page_config(
    page_title="Beyond Notes",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_app():
    """Initialize application components and session state."""
    # Ensure necessary directories exist
    AppPaths.ensure_dirs()
    
    # Initialize assessment loader if not already in session state
    if "assessment_loader" not in st.session_state:
        logger.info("Initializing assessment loader")
        loader = AssessmentLoader()
        
        # Create default types if needed (for first-time run)
        if not any(Path(AppPaths.get_types_dir()).glob("*.json")):
            logger.info("Creating default assessment types")
            loader.create_default_types()
            loader.reload()
        
        st.session_state.assessment_loader = loader
    
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.session_state.has_api_key = False
    else:
        st.session_state.has_api_key = True
        
        # Initialize LLM instance for shared usage if needed
        if "llm" not in st.session_state:
            model = "gpt-3.5-turbo"  # Default model
            st.session_state.llm = CustomLLM(api_key, model=model)
            logger.info(f"Initialized LLM with model {model}")

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
            "to use Beyond Notes."
        )
    
    # Main capabilities section
    st.subheader("Core Capabilities")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 🔍 Distill")
        st.markdown(
            "Create concise, focused summaries of meeting content with "
            "varying detail levels and formats."
        )
        if st.button("Summarize a Document", key="btn_distill"):
            st.switch_page("pages/02_Analyze.py")
            # Will need to set default assessment type
            st.session_state.default_assessment = "distill"
    
    with col2:
        st.markdown("### 📋 Extract")
        st.markdown(
            "Identify and organize action items, owners, and deadlines "
            "from meeting transcripts."
        )
        if st.button("Extract Action Items", key="btn_extract"):
            st.switch_page("pages/02_Analyze.py")
            st.session_state.default_assessment = "extract"
    
    with col3:
        st.markdown("### ⚠️ Assess")
        st.markdown(
            "Uncover issues, challenges, and risks with severity ratings "
            "and recommendations."
        )
        if st.button("Identify Issues", key="btn_assess"):
            st.switch_page("pages/02_Analyze.py")
            st.session_state.default_assessment = "assess"
    
    with col4:
        st.markdown("### 📊 Analyze")
        st.markdown(
            "Evaluate content against structured frameworks like "
            "readiness assessments."
        )
        if st.button("Analyze Framework", key="btn_analyze"):
            st.switch_page("pages/02_Analyze.py")
            st.session_state.default_assessment = "analyze"
    
    # How it works section
    st.subheader("How It Works")
    
    st.markdown("""
    Beyond Notes uses a multi-agent architecture where specialized AI agents collaborate:
    
    1. **🧠 Planner Agent** - Creates a tailored analysis strategy based on document content
    2. **🔍 Extractor Agent** - Identifies relevant information from document chunks
    3. **🧩 Aggregator Agent** - Combines findings and removes duplicates
    4. **⚖️ Evaluator Agent** - Assesses importance and assigns ratings
    5. **📊 Formatter Agent** - Transforms insights into structured reports
    
    The shared Processing Context enables agents to build upon each other's work, tracking metrics throughout processing.
    """)
    
    # Recent analyses section
    st.subheader("Recent Analyses")
    
    # Get recent output files across all assessment types
    recent_files = []
    for assessment_type in ["distill", "extract", "assess", "analyze"]:
        output_dir = AppPaths.get_assessment_output_dir(assessment_type)
        if output_dir.exists():
            for file in output_dir.glob(f"{assessment_type}_report_*.md"):
                stat = file.stat()
                recent_files.append({
                    "path": file,
                    "type": assessment_type,
                    "filename": file.name,
                    "modified": datetime.fromtimestamp(stat.st_mtime)
                })
    
    # Sort by modification time (most recent first)
    recent_files.sort(key=lambda x: x["modified"], reverse=True)
    
    if recent_files:
        # Display recent files
        for i, file in enumerate(recent_files[:5]):  # Show up to 5 recent files
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(f"{file['filename']}")
            with col2:
                st.write(f"Type: {file['type'].title()}")
            with col3:
                st.write(f"{file['modified'].strftime('%Y-%m-%d %H:%M')}")
                
            if i < len(recent_files) - 1:
                st.divider()
    else:
        st.info("No analyses have been performed yet. Get started by analyzing a document!")

# Main app execution
def main():
    # Initialize app components
    initialize_app()
    
    # Display the welcome page
    show_welcome_page()

if __name__ == "__main__":
    main()