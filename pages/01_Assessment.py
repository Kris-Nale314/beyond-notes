# pages/01_Assessment.py
import os
import sys
import asyncio
import streamlit as st
import logging
import json
import time
from pathlib import Path
from datetime import datetime
import importlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import project components
from core.models.document import Document
from core.llm.customllm import CustomLLM
from orchestrator import Orchestrator
from assessments.loader import AssessmentLoader

# Page config
st.set_page_config(
    page_title="Document Assessment - Beyond Notes",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .assessment-header {font-size: 2rem; font-weight: 700; color: #6c5ce7; margin-bottom: 1rem;}
    .report-container {padding: 1.5rem; background-color: rgba(108, 92, 231, 0.05); 
                     border-radius: 0.5rem; border-left: 5px solid #6c5ce7; margin: 1rem 0;}
    .metric-container {text-align: center; padding: 1rem; background-color: #f8f9fa; 
                      border-radius: 0.5rem; margin-bottom: 1rem;}
    .metric-value {font-size: 2rem; font-weight: 600; color: #6c5ce7;}
    .metric-label {font-size: 0.9rem; color: #666; margin-top: 0.25rem;}
    .progress-container {padding: 1rem; background-color: rgba(108, 92, 231, 0.1);
                        border-radius: 0.5rem; margin: 1rem 0;}
    
    /* HTML Report Styling */
    .report-html h1 {color: #6c5ce7; font-size: 2rem; margin-bottom: 1rem;}
    .report-html h2 {color: #6c5ce7; font-size: 1.5rem; margin-top: 2rem; margin-bottom: 1rem;}
    .report-html h3 {color: #555; font-size: 1.25rem; margin-top: 1.5rem; margin-bottom: 0.5rem;}
    .report-html .exec-summary {background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;}
    .report-html .critical {border-left: 5px solid #ff6b6b; padding-left: 1rem;}
    .report-html .high {border-left: 5px solid #ff9f43; padding-left: 1rem;}
    .report-html .medium {border-left: 5px solid #feca57; padding-left: 1rem;}
    .report-html .low {border-left: 5px solid #1dd1a1; padding-left: 1rem;}
    .report-html .item-card {background-color: #fff; padding: 1rem; border-radius: 0.5rem; 
                          margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);}
    .report-html .item-header {display: flex; justify-content: space-between; margin-bottom: 0.5rem;}
    .report-html .item-title {font-weight: 600;}
    .report-html .item-meta {color: #666; font-size: 0.9rem;}
    .report-html .item-badge {display: inline-block; padding: 0.25rem 0.5rem; border-radius: 0.25rem; 
                          font-size: 0.8rem; margin-right: 0.5rem;}
    .report-html .badge-critical {background-color: #ff6b6b; color: white;}
    .report-html .badge-high {background-color: #ff9f43; color: white;}
    .report-html .badge-medium {background-color: #feca57; color: black;}
    .report-html .badge-low {background-color: #1dd1a1; color: white;}
</style>
""", unsafe_allow_html=True)

def generate_html_report(result, assessment_type):
    """Generate an HTML report from the processing result."""
    
    # Get formatted result
    formatted_result = result.get("result", {})
    document_info = result.get("metadata", {}).get("document_info", {})
    
    # Base HTML structure
    html = f"""
    <div class="report-html">
        <h1>{formatted_result.get('title', f"{assessment_type.title()} Assessment")}</h1>
        
        <div class="exec-summary">
            <h2>Executive Summary</h2>
            <p>{formatted_result.get('executive_summary', '')}</p>
        </div>
    """
    
    # Handle different assessment types
    if assessment_type == "issues":
        # Add sections for issues by severity
        for section in formatted_result.get("sections", []):
            severity_class = "critical" if "Critical" in section.get("title", "") else \
                            "high" if "High" in section.get("title", "") else \
                            "medium" if "Medium" in section.get("title", "") else \
                            "low" if "Low" in section.get("title", "") else ""
            
            html += f"""
            <div class="{severity_class}">
                <h2>{section.get('title', 'Issues')}</h2>
                <p>{section.get('content', '')}</p>
                
                <div class="items-container">
            """
            
            # Add issues in this section
            for issue in section.get("issues", []):
                severity = issue.get("severity", "medium")
                html += f"""
                <div class="item-card">
                    <div class="item-header">
                        <span class="item-title">{issue.get('title', 'Issue')}</span>
                        <span class="item-meta">
                            <span class="item-badge badge-{severity}">{severity.upper()}</span>
                            <span class="item-category">{issue.get('category', '')}</span>
                        </span>
                    </div>
                    <p>{issue.get('description', '')}</p>
                    
                    {f"<p><strong>Impact:</strong> {issue.get('impact', '')}</p>" if issue.get('impact') else ""}
                    
                    {f"<div><strong>Recommendations:</strong><ul>" + "".join([f"<li>{r}</li>" for r in issue.get('recommendations', [])]) + "</ul></div>" if issue.get('recommendations') else ""}
                </div>
                """
            
            html += """
                </div>
            </div>
            """
    
    elif assessment_type == "action_items":
        # Group action items by owner
        items_by_owner = {}
        action_items = result.get("extracted_info", {}).get("action_items", [])
        
        for item in action_items:
            owner = item.get("owner", "Unassigned")
            if owner not in items_by_owner:
                items_by_owner[owner] = []
            items_by_owner[owner].append(item)
        
        # Add each owner section
        html += f"""
        <h2>Action Items by Owner</h2>
        """
        
        for owner, items in items_by_owner.items():
            html += f"""
            <h3>{owner}</h3>
            <div class="items-container">
            """
            
            for item in items:
                priority = item.get("priority", "medium")
                html += f"""
                <div class="item-card">
                    <div class="item-header">
                        <span class="item-title">{item.get('description', 'Action needed')}</span>
                        <span class="item-meta">
                            <span class="item-badge badge-{priority}">{priority.upper()}</span>
                            {f"<span>Due: {item.get('due_date', '')}</span>" if item.get('due_date') else ""}
                        </span>
                    </div>
                    {f"<p>{item.get('context', '')}</p>" if item.get('context') else ""}
                </div>
                """
            
            html += """
            </div>
            """
    
    elif assessment_type == "insights":
        # SWOT analysis
        swot = formatted_result.get("swot_analysis", {})
        
        # Strengths
        html += f"""
        <h2>Strengths</h2>
        <div class="items-container">
        """
        
        for strength in swot.get("strengths", []):
            impact = strength.get("impact", "medium")
            html += f"""
            <div class="item-card">
                <div class="item-header">
                    <span class="item-title">{strength.get('description', '')}</span>
                    <span class="item-meta">
                        <span class="item-badge badge-{impact}">{impact.upper()} IMPACT</span>
                    </span>
                </div>
                {f"<p><strong>Strategic Implication:</strong> {strength.get('strategic_implication', '')}</p>" if strength.get('strategic_implication') else ""}
            </div>
            """
        
        html += """
        </div>
        """
        
        # Weaknesses
        html += f"""
        <h2>Weaknesses</h2>
        <div class="items-container">
        """
        
        for weakness in swot.get("weaknesses", []):
            impact = weakness.get("impact", "medium")
            html += f"""
            <div class="item-card">
                <div class="item-header">
                    <span class="item-title">{weakness.get('description', '')}</span>
                    <span class="item-meta">
                        <span class="item-badge badge-{impact}">{impact.upper()} IMPACT</span>
                    </span>
                </div>
                {f"<p><strong>Strategic Implication:</strong> {weakness.get('strategic_implication', '')}</p>" if weakness.get('strategic_implication') else ""}
            </div>
            """
        
        html += """
        </div>
        """
        
        # Opportunities
        html += f"""
        <h2>Opportunities</h2>
        <div class="items-container">
        """
        
        for opportunity in swot.get("opportunities", []):
            impact = opportunity.get("impact", "medium")
            html += f"""
            <div class="item-card">
                <div class="item-header">
                    <span class="item-title">{opportunity.get('description', '')}</span>
                    <span class="item-meta">
                        <span class="item-badge badge-{impact}">{impact.upper()} IMPACT</span>
                    </span>
                </div>
                {f"<p><strong>Strategic Implication:</strong> {opportunity.get('strategic_implication', '')}</p>" if opportunity.get('strategic_implication') else ""}
            </div>
            """
        
        html += """
        </div>
        """
        
        # Threats
        html += f"""
        <h2>Threats</h2>
        <div class="items-container">
        """
        
        for threat in swot.get("threats", []):
            impact = threat.get("impact", "medium")
            html += f"""
            <div class="item-card">
                <div class="item-header">
                    <span class="item-title">{threat.get('description', '')}</span>
                    <span class="item-meta">
                        <span class="item-badge badge-{impact}">{impact.upper()} IMPACT</span>
                    </span>
                </div>
                {f"<p><strong>Strategic Implication:</strong> {threat.get('strategic_implication', '')}</p>" if threat.get('strategic_implication') else ""}
            </div>
            """
        
        html += """
        </div>
        """
        
        # Strategic recommendations
        html += f"""
        <h2>Strategic Recommendations</h2>
        <div class="items-container">
        """
        
        for rec in formatted_result.get("strategic_recommendations", []):
            priority = rec.get("priority", "medium")
            html += f"""
            <div class="item-card">
                <div class="item-header">
                    <span class="item-title">{rec.get('recommendation', '')}</span>
                    <span class="item-meta">
                        <span class="item-badge badge-{priority}">{priority.upper()}</span>
                    </span>
                </div>
                <p><strong>Rationale:</strong> {rec.get('rationale', '')}</p>
            </div>
            """
        
        html += """
        </div>
        """
    
    # Close HTML
    html += """
    </div>
    """
    
    return html

def generate_report_file(result, assessment_type, file_format="html"):
    """Generate a downloadable report file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a clean copy of the result without any API keys
    clean_result = result.copy()
    if "metadata" in clean_result and "options" in clean_result["metadata"]:
        if "api_key" in clean_result["metadata"]["options"]:
            clean_result["metadata"]["options"].pop("api_key")
    
    if file_format == "html":
        # No changes needed for HTML as it uses the template
        html_content = generate_html_report(clean_result, assessment_type)
        return html_content, f"{assessment_type}_report_{timestamp}.html", "text/html"
    
    elif file_format == "markdown":
        # Generate markdown from the result
        md_content = f"# {assessment_type.title()} Assessment\n\n"
        
        # Add executive summary
        formatted_result = clean_result.get("result", {})
        md_content += f"## Executive Summary\n\n{formatted_result.get('executive_summary', '')}\n\n"
        
        # Add sections
        for section in formatted_result.get("sections", []):
            md_content += f"## {section.get('title', 'Section')}\n\n{section.get('content', '')}\n\n"
            
            # Add items in this section
            for item in section.get("issues", []):
                md_content += f"### {item.get('title', 'Item')}\n\n"
                md_content += f"*Severity: {item.get('severity', 'Unknown')} | Category: {item.get('category', 'Unknown')}*\n\n"
                md_content += f"{item.get('description', '')}\n\n"
                
                if "impact" in item:
                    md_content += f"**Impact:** {item['impact']}\n\n"
                
                if "recommendations" in item and item["recommendations"]:
                    md_content += "**Recommendations:**\n\n"
                    for rec in item["recommendations"]:
                        md_content += f"- {rec}\n"
                    md_content += "\n"
        
        return md_content, f"{assessment_type}_report_{timestamp}.md", "text/markdown"
    
    elif file_format == "json":
        return json.dumps(clean_result, indent=2), f"{assessment_type}_report_{timestamp}.json", "application/json"
    
    return None, None, None

async def process_document(document, assessment_type, model, options):
    """Process a document with the orchestrator."""
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return None
    
    # Initialize orchestrator
    orchestrator = Orchestrator(
        assessment_type=assessment_type,
        options={
            "api_key": api_key,
            "model": model,
            **options
        }
    )
    
    # Setup progress tracking
    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0)
    status_text = st.empty()
    
    def update_progress(progress, message):
        progress_bar.progress(progress)
        status_text.text(message)
    
    # Process document
    result = await orchestrator.process_with_progress(
        document=document,
        progress_callback=update_progress
    )
    
    # Clear progress indicators
    progress_placeholder.empty()
    status_text.empty()
    
    return result

def get_client_context_inputs(assessment_type, assessment_loader):
    """Get user inputs for client context if needed."""
    client_context = {}
    
    if assessment_type == "insights":
        user_options = assessment_loader.get_user_options("insights")
        if "client_context" in user_options and user_options["client_context"].get("required", False):
            st.subheader("Client Context")
            st.write("Please provide background information to ground the analysis:")
            
            fields = user_options["client_context"].get("fields", {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                client_context["client_name"] = st.text_input("Client Name", help=fields.get("client_name", ""))
                client_context["industry"] = st.text_input("Industry", help=fields.get("industry", ""))
                client_context["size"] = st.text_input("Organization Size", help=fields.get("size", ""))
            
            with col2:
                client_context["key_challenges"] = st.text_area("Key Challenges", help=fields.get("key_challenges", ""))
                client_context["objectives"] = st.text_area("Objectives", help=fields.get("objectives", ""))
            
            client_context["current_systems"] = st.text_area("Current Systems/Technology", help=fields.get("current_systems", ""))
            client_context["additional_context"] = st.text_area("Additional Context", help=fields.get("additional_context", ""))
    
    return client_context

def main():
    st.markdown('<div class="assessment-header">Document Assessment</div>', unsafe_allow_html=True)
    
    # Initialize assessment loader
    assessment_loader = AssessmentLoader()
    
    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("Assessment Settings")
        
        # Assessment type selection
        assessment_types = assessment_loader.get_assessment_types()
        if not assessment_types:
            st.warning("No assessment types found. Creating default assessments...")
            assessment_loader.create_assessment_directories()
            st.success("Default assessments created. Please refresh the page.")
            assessment_types = assessment_loader.get_assessment_types()
        
        assessment_type = st.selectbox(
            "Assessment Type",
            assessment_types,
            index=0 if "issues" in assessment_types else 0,
            help="Choose the type of analysis to perform on the document"
        )
        
        # Get assessment config
        assessment_config = assessment_loader.load_assessment(assessment_type)
        if assessment_config:
            st.caption(assessment_config.get("description", ""))
        
        st.divider()
        
        # Model settings
        model = st.selectbox(
            "Model",
            ["gpt-3.5-turbo", "gpt-4"],
            index=0,
            help="Select the model to use (gpt-3.5-turbo is faster, gpt-4 is more accurate)"
        )
        
        detail_level = st.select_slider(
            "Detail Level",
            options=["essential", "standard", "comprehensive"],
            value="standard",
            help="Control the level of detail in the analysis"
        )
        
        st.divider()
        
        # Processing options
        st.subheader("Processing Options")
        
        chunk_size = st.slider(
            "Chunk Size (tokens)",
            min_value=1000,
            max_value=15000,
            value=8000,
            step=1000,
            help="Size of document chunks for processing"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap (tokens)",
            min_value=0,
            max_value=1000,
            value=300,
            step=100,
            help="Overlap between chunks to maintain context"
        )
    
    # Main area - tabs
    tab1, tab2 = st.tabs(["Upload Document", "Assessment Results"])
    
    with tab1:
        st.subheader("Upload Document")
        
        # File upload
        uploaded_file = st.file_uploader("Choose a text file", type=["txt"])
        
        # Or select sample
        sample_files = list(Path("data/samples").glob("*.txt"))
        if sample_files:
            sample_option = st.selectbox(
                "Or select a sample document",
                [""] + [f.name for f in sample_files]
            )
        
        # Document display
        document = None
        
        if uploaded_file:
            try:
                # Reset the file pointer
                uploaded_file.seek(0)
                
                # Create document using Document class
                document = Document.from_uploaded_file(uploaded_file)
                st.success(f"Loaded document: {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"Error loading document: {str(e)}")
        
        elif sample_option:
            try:
                # Load sample file
                with open(Path("data/samples") / sample_option, 'r') as f:
                    doc_text = f.read()
                
                document = Document(text=doc_text, filename=sample_option)
                st.success(f"Loaded sample: {sample_option}")
                
            except Exception as e:
                st.error(f"Error loading sample: {str(e)}")
        
        # Show document details if loaded
        if document:
            st.subheader("Document Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Filename: {document.filename}")
                st.write(f"Word count: {document.word_count:,}")
            with col2:
                st.write(f"Character count: {document.character_count:,}")
                st.write(f"Estimated tokens: {document.estimated_tokens:,}")
            
            # Document preview
            with st.expander("Document Preview", expanded=False):
                st.text_area(
                    "Content Preview", 
                    document.text[:2000] + "..." if len(document.text) > 2000 else document.text, 
                    height=200
                )
            
            # Get client context if needed
            client_context = get_client_context_inputs(assessment_type, assessment_loader)
            
            # Process button
            process_button = st.button("Process Document")
            
            if process_button:
                # Processing options
                options = {
                    "detail_level": detail_level,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "client_context": client_context if client_context else None
                }
                
                # Process document
                with st.spinner("Processing document..."):
                    result = asyncio.run(process_document(document, assessment_type, model, options))
                    
                    if result:
                        # Remove API key from result before storing
                        if "metadata" in result and "options" in result["metadata"]:
                            if "api_key" in result["metadata"]["options"]:
                                result["metadata"]["options"].pop("api_key")
                        
                        # Store result in session state
                        st.session_state.document = document
                        st.session_state.assessment_result = result
                        st.session_state.assessment_type = assessment_type
                        
                        # Save result to file with API key removed
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = f"output/{assessment_type}_result_{timestamp}.json"
                        
                        # Create a clean copy of the result without any API keys
                        clean_result = result.copy()
                        if "metadata" in clean_result and "options" in clean_result["metadata"]:
                            if "api_key" in clean_result["metadata"]["options"]:
                                clean_result["metadata"]["options"].pop("api_key")
                        
                        with open(output_path, 'w') as f:
                            json.dump(clean_result, f, indent=2)
                        
                        st.success(f"Processing complete! Results saved to {output_path}")
                        
                        # Prompt to view results
                        st.info("Switch to the 'Assessment Results' tab to view the analysis")
    
    with tab2:
        st.subheader("Assessment Results")
        
        if "assessment_result" not in st.session_state:
            st.info("Process a document to see results here")
        else:
            result = st.session_state.assessment_result
            document = st.session_state.document
            assessment_type = st.session_state.assessment_type
            
            # Document info
            st.write(f"Document: {document.filename}")
            
            # Metrics
            st.markdown("### Key Metrics")
            metrics_cols = st.columns(4)
            
            # Extract metrics from result
            processing_time = result.get("metadata", {}).get("processing_time", 0)
            
            extraction_counts = {}
            if "statistics" in result and "extraction_counts" in result["statistics"]:
                extraction_counts = result["statistics"]["extraction_counts"]
            elif "statistics" in result:
                # Special handling based on assessment type
                if assessment_type == "issues":
                    extraction_counts["issues"] = result["statistics"].get("total_issues", 0)
                elif assessment_type == "action_items":
                    extraction_counts["action_items"] = result["statistics"].get("total_action_items", 0)
                elif assessment_type == "insights":
                    extraction_counts["strengths"] = result["statistics"].get("strengths_count", 0)
                    extraction_counts["weaknesses"] = result["statistics"].get("weaknesses_count", 0)
                    extraction_counts["opportunities"] = result["statistics"].get("opportunities_count", 0)
                    extraction_counts["threats"] = result["statistics"].get("threats_count", 0)
            
            total_extractions = sum(extraction_counts.values())
            
            # Display metrics
            with metrics_cols[0]:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{processing_time:.1f}s</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Processing Time</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with metrics_cols[1]:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{document.word_count:,}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Document Words</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with metrics_cols[2]:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{total_extractions}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Total Extractions</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with metrics_cols[3]:
                chunks = result.get("metadata", {}).get("stages", {}).get("chunking", {}).get("result", {}).get("total_chunks", 0)
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{chunks}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Document Chunks</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display extraction counts if available
            if extraction_counts:
                st.markdown("#### Extraction Counts")
                count_cols = st.columns(len(extraction_counts))
                
                for i, (key, value) in enumerate(extraction_counts.items()):
                    with count_cols[i]:
                        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-value">{value}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="metric-label">{key.replace("_", " ").title()}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # Display HTML report
            st.markdown("### Assessment Report")
            html_report = generate_html_report(result, assessment_type)
            st.markdown('<div class="report-container">', unsafe_allow_html=True)
            st.markdown(html_report, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download options
            st.markdown("### Download Report")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                html_content, html_filename, html_mime = generate_report_file(result, assessment_type, "html")
                st.download_button(
                    label="Download HTML Report",
                    data=html_content,
                    file_name=html_filename,
                    mime=html_mime
                )
            
            with col2:
                md_content, md_filename, md_mime = generate_report_file(result, assessment_type, "markdown") 
                st.download_button(
                    label="Download Markdown",
                    data=md_content,
                    file_name=md_filename,
                    mime=md_mime
                )
            
            with col3:
                json_content, json_filename, json_mime = generate_report_file(result, assessment_type, "json")
                st.download_button(
                    label="Download Raw JSON",
                    data=json_content,
                    file_name=json_filename,
                    mime=json_mime
                )

if __name__ == "__main__":
    main()