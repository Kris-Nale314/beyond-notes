# app.py - Main entry point for Beyond Notes application
import streamlit as st
import os
import logging
from pathlib import Path
from datetime import datetime

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

# Import components with clean error handling
try:
    logger.info("Importing core components...")
    from assessments.loader import AssessmentLoader
    from utils.paths import AppPaths
    from utils.ui.styles import get_base_styles
    logger.info("Successfully imported core components")
except ImportError as e:
    logger.error(f"Failed to import core components: {e}", exc_info=True)
    st.error(f"Application Error: Failed to load necessary components. Please check logs.")
    st.stop()

# Ensure directories exist
AppPaths.ensure_dirs()

# Page configuration
st.set_page_config(
    page_title="Beyond Notes",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply shared styles
st.markdown(get_base_styles(), unsafe_allow_html=True)

# Add home page specific styles
st.markdown("""
<style>
    /* Hero section styling */
    .hero-container {
        background-color: rgba(0, 0, 0, 0.1);
        border-radius: 10px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #2196F3, #673AB7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        display: inline-block;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 1.5rem;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Feature card styling */
    .feature-card {
        background-color: rgba(0, 0, 0, 0.05);
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.7rem;
    }
    
    .feature-description {
        opacity: 0.8;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Architecture section */
    .architecture-heading {
        font-size: 1.8rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 0.5rem;
    }
    
    .agent-container {
        background-color: rgba(0, 0, 0, 0.05);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
        border-left: 3px solid #2196F3;
        transition: background-color 0.2s ease;
    }
    
    .agent-container:hover {
        background-color: rgba(33, 150, 243, 0.1);
    }
    
    .agent-name {
        font-weight: 600;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
    }
    
    .agent-icon {
        margin-right: 0.5rem;
    }
    
    .agent-description {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 0.9rem;
        opacity: 0.7;
    }
</style>
""", unsafe_allow_html=True)

def initialize_app():
    """Initialize application components and session state."""
    try:
        # Load assessment configurations
        if "assessment_loader" not in st.session_state:
            logger.info("Initializing assessment loader")
            loader = AssessmentLoader()
            
            # Create default assessment types if needed
            types_dir = Path(AppPaths.get_types_dir())
            if not any(types_dir.glob("*.json")):
                logger.info("Creating default assessment types")
                loader.create_default_types()
                loader.reload()
                
            st.session_state.assessment_loader = loader
            
        # Check for API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            st.session_state.has_api_key = False
            logger.warning("OPENAI_API_KEY not set")
        else:
            st.session_state.has_api_key = True
            
        # Initialize preferred format if not set
        if "preferred_summary_format" not in st.session_state:
            st.session_state.preferred_summary_format = "executive"
            
    except Exception as e:
        logger.error(f"Error during app initialization: {e}", exc_info=True)
        st.error(f"Application Initialization Failed: {e}")

def main():
    """Main function to render the home page."""
    # Initialize app
    initialize_app()
    
    # API key warning if needed
    if not st.session_state.get("has_api_key", False):
        st.warning(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable "
            "to use Beyond Notes. Check the documentation for setup instructions."
        )
    
    # Hero section
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">Beyond Notes</div>
            <div class="hero-subtitle">
                Transform documents into structured insights using multi-agent AI orchestration.
                Beyond Notes extracts deeper meaning through specialized agent collaboration.
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Core capabilities section
    st.markdown("<h2>Core Capabilities</h2>", unsafe_allow_html=True)
    
    # Use Streamlit columns for feature cards
    col1, col2 = st.columns(2)
    
    with col1:
        # Distill feature
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <div class="feature-title">Distill</div>
                <div class="feature-description">
                    Create concise, focused summaries of meeting content with varying detail levels and formats. 
                    Transform lengthy transcripts into actionable insights.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Extract feature  
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📋</div>
                <div class="feature-title">Extract</div>
                <div class="feature-description">
                    Identify and organize action items, owners, and deadlines from meeting transcripts.
                    Never miss a follow-up item again.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        # Assess feature
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">⚠️</div>
                <div class="feature-title">Assess</div>
                <div class="feature-description">
                    Uncover issues, challenges, and risks with severity ratings and recommendations. 
                    Identify potential problems before they become critical.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Analyze feature
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Analyze</div>
                <div class="feature-description">
                    Evaluate content against structured frameworks like readiness assessments.
                    Get objective measurement of document quality.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # How it works section
    with st.expander("How It Works: Multi-Agent Architecture", expanded=False):
        st.markdown('<div class="architecture-heading">Multi-Agent Collaboration</div>', unsafe_allow_html=True)
        
        st.markdown("Beyond Notes uses a specialized pipeline of AI agents, each focused on a specific task:")
        
        # Use individual markdown calls for each agent
        st.markdown(
            """
            <div class="agent-container">
                <div class="agent-name"><span class="agent-icon">🧭</span> Planner Agent</div>
                <div class="agent-description">Analyzes the document structure and creates a tailored extraction strategy</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        st.markdown(
            """
            <div class="agent-container">
                <div class="agent-name"><span class="agent-icon">🔍</span> Extractor Agent</div>
                <div class="agent-description">Identifies relevant information from document chunks with precision</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        st.markdown(
            """
            <div class="agent-container">
                <div class="agent-name"><span class="agent-icon">🧩</span> Aggregator Agent</div>
                <div class="agent-description">Combines findings and removes duplicates intelligently</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        st.markdown(
            """
            <div class="agent-container">
                <div class="agent-name"><span class="agent-icon">⚖️</span> Evaluator Agent</div>
                <div class="agent-description">Assesses information quality and importance</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        st.markdown(
            """
            <div class="agent-container">
                <div class="agent-name"><span class="agent-icon">📊</span> Formatter Agent</div>
                <div class="agent-description">Transforms insights into structured, readable reports</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
            
        st.markdown("""
        The shared `ProcessingContext` enables these agents to build upon each other's work,
        creating a more coherent and insightful analysis than any single model could achieve.
        """)
    
    # Getting started guidance
    st.markdown("## Getting Started")
    st.markdown("""
    To begin using Beyond Notes, select one of the tools from the sidebar navigation:

    1. Use the **Document Summarizer** to create concise summaries of your documents
    2. Try the **Issue Assessment** tool to identify potential problems and challenges
    3. Extract action items with the **Action Item Extraction** tool
    4. Chat with your documents using the **Document Chat** feature
    """)
    
    # Footer
    st.markdown(
        """
        <div class="footer">
            Beyond Notes © 2025 | An educational project demonstrating multi-agent AI orchestration
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    try:
        logger.info("Starting app.py main execution")
        main()
        logger.info("app.py execution completed successfully")
    except Exception as e:
        logger.error(f"Unhandled exception in main execution: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {e}")