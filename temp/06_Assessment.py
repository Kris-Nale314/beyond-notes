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
import matplotlib.pyplot as plt
import io
import base64

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
from utils.paths import AppPaths

# Page config
st.set_page_config(
    page_title="Document Assessment - Beyond Notes",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

def display_pipeline_progress(stages):
    """Display the processing pipeline stages in a simple format"""
    if not stages:
        return "No pipeline data available"
    
    # Process stages in order
    stage_order = [
        "document_analysis", "chunking", "planning", 
        "extraction", "aggregation", "evaluation", 
        "formatting", "review"
    ]
    
    # Map stage names to more readable names
    stage_names = {
        "document_analysis": "Document Analysis",
        "chunking": "Document Chunking",
        "planning": "Planning",
        "extraction": "Information Extraction",
        "aggregation": "Aggregation",
        "evaluation": "Evaluation",
        "formatting": "Report Formatting",
        "review": "Quality Review"
    }
    
    pipeline_text = ""
    
    for stage_key in stage_order:
        if stage_key not in stages:
            continue
            
        stage = stages[stage_key]
        status = stage.get("status", "waiting")
        
        # Format status and progress
        if status == "completed":
            status_text = "✅ Completed"
            progress = "100%"
        elif status == "running":
            status_text = "🔄 Running"
            progress = f"{stage.get('progress', 0)*100:.0f}%"
        elif status == "failed":
            status_text = "❌ Failed"
            progress = "0%"
        else:
            status_text = "⏳ Waiting"
            progress = "0%"
        
        # Calculate duration if available
        duration = ""
        if "duration" in stage:
            duration = f"{stage['duration']:.1f}s"
        
        # Add stage to pipeline text
        pipeline_text += f"**{stage_names.get(stage_key, stage_key)}**: {status_text} {progress} {duration}\n\n"
    
    return pipeline_text

def format_simple_issues_report(result):
    """Format issues in a simple markdown report"""
    report_text = "# Issues Assessment\n\n"
    
    # Add executive summary
    formatted_result = result.get("result", {}).get("report", {})
    report_text += f"## Executive Summary\n\n{formatted_result.get('executive_summary', '')}\n\n"
    
    # Get the properly evaluated issues
    evaluated_issues = result.get("evaluations", {}).get("issues", {}).get("evaluated_issues", [])
    
    # Group issues by severity
    issues_by_severity = {}
    for issue in evaluated_issues:
        severity = issue.get("severity", "Unknown").lower()
        if severity not in issues_by_severity:
            issues_by_severity[severity] = []
        issues_by_severity[severity].append(issue)
    
    # Display issues by severity
    for severity in ["critical", "high", "medium", "low"]:
        if severity in issues_by_severity and issues_by_severity[severity]:
            report_text += f"## {severity.title()} Severity Issues\n\n"
            
            for issue in issues_by_severity[severity]:
                title = issue.get("title", "Untitled Issue")
                category = issue.get("category", "")
                description = issue.get("description", "")
                impact = issue.get("impact", "")
                
                report_text += f"### {title}\n\n"
                report_text += f"**Category:** {category}\n\n"
                report_text += f"{description}\n\n"
                
                if impact:
                    report_text += f"**Impact:** {impact}\n\n"
                
                if "recommendations" in issue and issue["recommendations"]:
                    report_text += "**Recommendations:**\n\n"
                    for rec in issue["recommendations"]:
                        report_text += f"- {rec}\n"
                    report_text += "\n"
    
    return report_text
    
def format_simple_action_items_report(result):
    """Format action items in a simple markdown report"""
    report_text = "# Action Items Assessment\n\n"
    
    # Add summary
    formatted_result = result.get("result", {})
    report_text += f"## Summary\n\n{formatted_result.get('summary', '')}\n\n"
    
    # Group action items by owner
    items_by_owner = {}
    action_items = result.get("extracted_info", {}).get("action_items", [])
    
    for item in action_items:
        owner = item.get("owner", "Unassigned")
        if owner not in items_by_owner:
            items_by_owner[owner] = []
        items_by_owner[owner].append(item)
    
    # Add action items by owner
    report_text += "## Action Items by Owner\n\n"
    
    for owner, items in items_by_owner.items():
        report_text += f"### {owner}\n\n"
        
        for item in items:
            priority = item.get("priority", "medium")
            due_date = f"Due: {item.get('due_date', 'Unspecified')}" if item.get("due_date") else ""
            
            report_text += f"**{item.get('description', 'Action needed')}**\n\n"
            report_text += f"Priority: {priority.upper()}  |  {due_date}\n\n"
            
            if item.get("context"):
                report_text += f"{item['context']}\n\n"
    
    return report_text

def format_simple_insights_report(result):
    """Format SWOT insights in a simple markdown report"""
    report_text = "# SWOT Analysis\n\n"
    
    # Add executive summary
    formatted_result = result.get("result", {})
    report_text += f"## Executive Summary\n\n{formatted_result.get('executive_summary', '')}\n\n"
    
    # SWOT analysis
    swot = formatted_result.get("swot_analysis", {})
    
    # Strengths
    report_text += "## Strengths\n\n"
    for strength in swot.get("strengths", []):
        impact = strength.get("impact", "medium")
        report_text += f"**{strength.get('description', '')}** *(Impact: {impact.upper()})*\n\n"
        if strength.get("strategic_implication"):
            report_text += f"*Strategic Implication:* {strength['strategic_implication']}\n\n"
    
    # Weaknesses
    report_text += "## Weaknesses\n\n"
    for weakness in swot.get("weaknesses", []):
        impact = weakness.get("impact", "medium")
        report_text += f"**{weakness.get('description', '')}** *(Impact: {impact.upper()})*\n\n"
        if weakness.get("strategic_implication"):
            report_text += f"*Strategic Implication:* {weakness['strategic_implication']}\n\n"
    
    # Opportunities
    report_text += "## Opportunities\n\n"
    for opportunity in swot.get("opportunities", []):
        impact = opportunity.get("impact", "medium")
        report_text += f"**{opportunity.get('description', '')}** *(Impact: {impact.upper()})*\n\n"
        if opportunity.get("strategic_implication"):
            report_text += f"*Strategic Implication:* {opportunity['strategic_implication']}\n\n"
    
    # Threats
    report_text += "## Threats\n\n"
    for threat in swot.get("threats", []):
        impact = threat.get("impact", "medium")
        report_text += f"**{threat.get('description', '')}** *(Impact: {impact.upper()})*\n\n"
        if threat.get("strategic_implication"):
            report_text += f"*Strategic Implication:* {threat['strategic_implication']}\n\n"
    
    # Strategic recommendations
    report_text += "## Strategic Recommendations\n\n"
    for rec in formatted_result.get("strategic_recommendations", []):
        priority = rec.get("priority", "medium")
        report_text += f"**{rec.get('recommendation', '')}** *(Priority: {priority.upper()})*\n\n"
        report_text += f"*Rationale:* {rec.get('rationale', '')}\n\n"
    
    return report_text

def format_assessment_report(result, assessment_type):
    """Format the assessment report based on the type"""
    if assessment_type == "issues":
        return format_simple_issues_report(result)
    elif assessment_type == "action_items":
        return format_simple_action_items_report(result)
    elif assessment_type == "insights":
        return format_simple_insights_report(result)
    else:
        # Generic report format
        return f"# {assessment_type.title()} Assessment\n\nNo specific formatter available for this assessment type."

def save_result_to_output(result, assessment_type):
    """Save the result to the output folder with proper organization"""
    # Ensure output directories exist
    output_dir = AppPaths.get_assessment_output_dir(assessment_type)
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a clean copy of the result without sensitive data
    clean_result = result.copy()
    if "metadata" in clean_result and "options" in clean_result["metadata"]:
        options = clean_result["metadata"]["options"]
        if "api_key" in options:
            options.pop("api_key")
    
    # Save JSON result
    output_file = output_dir / f"{assessment_type}_result_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(clean_result, f, indent=2)
    
    # Generate and save markdown report
    md_content = format_assessment_report(clean_result, assessment_type)
    md_file = output_dir / f"{assessment_type}_report_{timestamp}.md"
    with open(md_file, 'w') as f:
        f.write(md_content)
    
    # Return paths
    return {
        "json_path": str(output_file),
        "md_path": str(md_file)
    }

async def process_document(document, assessment_type, model, options):
    """Process a document with the orchestrator and track progress."""
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
    progress_bar = st.progress(0)
    status_text = st.empty()
    pipeline_status = st.empty()
    
    # Process data and timer
    processing_data = {
        "stages": {},
        "current_stage": None,
        "start_time": time.time()
    }
    
    def update_progress(progress, message):
        progress_bar.progress(progress)
        status_text.text(message)
        
        # Get the current pipeline status
        progress_data = orchestrator.get_progress()
        
        # Update processing data
        processing_data["current_stage"] = progress_data.get("current_stage")
        processing_data["stages"] = progress_data.get("stages", {})
        
        # Display pipeline status as simple text
        pipeline_status.markdown(display_pipeline_progress(processing_data["stages"]))
    
    # Process document
    result = await orchestrator.process_with_progress(
        document=document,
        progress_callback=update_progress
    )
    
    # Clear progress indicators
    progress_bar.empty()
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
    st.title("Document Assessment")
    
    # Ensure necessary directories exist
    AppPaths.ensure_dirs()
    
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
            help="Choose the type of analysis to perform"
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
            help="Select the model to use"
        )
        
        detail_level = st.select_slider(
            "Detail Level",
            options=["essential", "standard", "comprehensive"],
            value="standard",
            help="Control the level of detail"
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
            help="Size of document chunks"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap (tokens)",
            min_value=0,
            max_value=1000,
            value=300,
            step=100,
            help="Overlap between chunks"
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
                        # Store result in session state
                        st.session_state.document = document
                        st.session_state.assessment_result = result
                        st.session_state.assessment_type = assessment_type
                        
                        # Save result to output folder
                        output_paths = save_result_to_output(result, assessment_type)
                        
                        st.success(f"Processing complete! Results saved to output/{assessment_type}/")
                        
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
            
            # Processing metrics
            col1, col2, col3 = st.columns(3)
            
            processing_time = result.get("metadata", {}).get("processing_time", 0)
            
            with col1:
                st.metric("Processing Time", f"{processing_time:.1f}s")
            
            with col2:
                st.metric("Document Words", f"{document.word_count:,}")
            
            with col3:
                chunks = result.get("metadata", {}).get("stages", {}).get("chunking", {}).get("result", {}).get("total_chunks", 0)
                st.metric("Document Chunks", chunks)
            
            # Create tabs for different views
            report_tab, pipeline_tab, data_tab = st.tabs(["Report", "Pipeline", "Raw Data"])
            
            with report_tab:
                # Format and display report
                st.markdown(format_assessment_report(result, assessment_type))
                
                # Download options
                col1, col2 = st.columns(2)
                
                with col1:
                    # Prepare markdown for download
                    md_content = format_assessment_report(result, assessment_type)
                    st.download_button(
                        label="Download as Markdown",
                        data=md_content,
                        file_name=f"{assessment_type}_report.md",
                        mime="text/markdown"
                    )
                
                with col2:
                    # Prepare JSON for download
                    json_content = json.dumps(result, indent=2)
                    st.download_button(
                        label="Download Raw JSON",
                        data=json_content,
                        file_name=f"{assessment_type}_result.json",
                        mime="application/json"
                    )
            
            with pipeline_tab:
                # Display pipeline stages
                st.subheader("Processing Pipeline")
                stages = result.get("metadata", {}).get("stages", {})
                st.markdown(display_pipeline_progress(stages))
                
                # Extract and display stage durations
                durations = {}
                for stage_name, stage_data in stages.items():
                    if "duration" in stage_data:
                        durations[stage_name] = stage_data["duration"]
                
                if durations:
                    # Create a simple bar chart
                    fig, ax = plt.subplots()
                    y_pos = range(len(durations))
                    ax.barh(y_pos, durations.values())
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(durations.keys())
                    ax.set_xlabel('Duration (seconds)')
                    ax.set_title('Processing Time by Stage')
                    
                    st.pyplot(fig)
            
            with data_tab:
                # Show raw data in expandable sections
                st.subheader("Raw Data")
                
                with st.expander("Extracted Information"):
                    st.json(result.get("extracted_info", {}))
                
                with st.expander("Statistics"):
                    st.json(result.get("statistics", {}))
                
                with st.expander("Full Result"):
                    st.json(result)

if __name__ == "__main__":
    main()