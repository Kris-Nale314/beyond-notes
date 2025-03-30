# app.py
import os
import streamlit as st
from pathlib import Path

from dotenv import load_dotenv
import openai
import logging
import sys
import warnings
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Beyond Notes",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #6c5ce7; margin-bottom: 1rem;}
    .subheader {font-size: 1.5rem; font-weight: 500; color: #a29bfe; margin-bottom: 2rem;}
    .feature-card {
        padding: 1.5rem; 
        border-radius: 0.5rem; 
        background-color: rgba(108, 92, 231, 0.05);
        border-left: 5px solid #6c5ce7;
        margin: 1rem 0;
    }
    .feature-title {font-size: 1.2rem; font-weight: 600; color: #6c5ce7; margin-bottom: 0.5rem;}
</style>
""", unsafe_allow_html=True)

# Ensure directories exist
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/samples", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("assessments", exist_ok=True)

def main():
    # App title
    st.markdown('<div class="main-header">Beyond Notes</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Transform Meeting Transcripts into Structured Insights</div>', unsafe_allow_html=True)
    
    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.warning("⚠️ OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    
    # Main content
    st.write("Beyond Notes applies multi-stage reasoning and specialized agent frameworks to transform meeting transcripts into structured, actionable insights.")
    
    # Feature cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<div class="feature-title">Multiple Assessment Types</div>', unsafe_allow_html=True)
        st.write("Choose from different assessment types to extract relevant insights:")
        st.write("• **Issues Analysis**: Identify problems and challenges")
        st.write("• **Action Items**: Extract follow-up tasks with ownership")
        st.write("• **SWOT Analysis**: Discover strengths, weaknesses, opportunities, and threats")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<div class="feature-title">Multi-Agent Architecture</div>', unsafe_allow_html=True)
        st.write("Leverages specialized AI agents for better analysis:")
        st.write("• **Planner**: Creates analysis strategy")
        st.write("• **Extractor**: Identifies relevant information")
        st.write("• **Aggregator**: Combines findings across sections")
        st.write("• **Evaluator**: Assesses importance and relevance")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<div class="feature-title">Enhanced Understanding</div>', unsafe_allow_html=True)
        st.write("Beyond basic summarization with:")
        st.write("• **Progressive Enhancement**: Multi-stage processing")
        st.write("• **Contextual Analysis**: Client-specific insights")
        st.write("• **Structured Output**: Organized, actionable reports")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<div class="feature-title">Getting Started</div>', unsafe_allow_html=True)
        st.write("Use the pages in the sidebar to:")
        st.write("• **Process Documents**: Extract insights from transcripts")
        st.write("• **Test Agents**: Evaluate individual agent performance")
        st.write("• **Manage Assessments**: Configure assessment types")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.divider()
    st.caption("Beyond Notes v1.0.0 | Use the sidebar to navigate between pages")

if __name__ == "__main__":
    main()