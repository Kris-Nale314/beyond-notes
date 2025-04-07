"""
Rendering components for the Beyond Notes application.
Handles formatting and display of assessment results.
"""

import streamlit as st
import json
from datetime import datetime
import logging


def render_assessment_result(result, assessment_type):
    """Render assessment result based on type."""
    if assessment_type == "distill":
        return render_summary_result(result)
    elif assessment_type == "assess":
        return render_issues_result(result)
    elif assessment_type == "extract":
        return render_actions_result(result)
    elif assessment_type == "analyze":
        return render_analysis_result(result)
    else:
        st.warning(f"Unknown assessment type: {assessment_type}")
        st.json(result)

def render_summary_result(result, format_type=None):
    """Render summary results (reuses existing code from summary_renderer)."""
    # Get the format type from the result if not specified
    if not format_type:
        format_type = result.get("metadata", {}).get("user_options", {}).get("format", "executive")
    
    # Import summary_renderer function from core modules
    from core.models.summary_renderer import render_summary
    
    # Generate HTML with the renderer
    html_content = render_summary(result, format_type)
    
    # Display the summary
    st.markdown(html_content, unsafe_allow_html=True)
    
    # Return the generated HTML for potential additional use
    return html_content

def render_issues_result(result):
    """Render issues assessment results using native Streamlit components."""
    # First, log structure for debugging
    logger = logging.getLogger("beyond-notes-issues")
    logger.info(f"Result keys: {list(result.keys())}")
    if "result" in result:
        logger.info(f"Result.result keys: {list(result['result'].keys())}")
    if "formatted" in result:
        logger.info(f"Result.formatted keys: {list(result['formatted'].keys())}")
    
    # Check for formatter errors
    formatter_errors = result.get("error") or result.get("formatter_warnings")
    if formatter_errors:
        st.error("The formatter encountered errors:")
        st.code(formatter_errors)
        
        # Show raw data for debugging
        with st.expander("Show Raw Result Data", expanded=True):
            st.json(result)
        return
        
    # Extract data
    metadata = result.get("metadata", {})
    
    # Find issues in various possible locations in the result
    issues = result.get("issues", [])
    if not issues and "result" in result and "issues" in result["result"]:
        issues = result["result"]["issues"]
    if not issues and "formatted" in result and "issues" in result["formatted"]:  
        issues = result["formatted"]["issues"]
    # If still no issues found, try to extract structured content from raw text
    if not issues and "result" in result:
        result_content = result["result"]
        if isinstance(result_content, str) and len(result_content) > 0:
            st.markdown("## Results Text")
            st.markdown(result_content)
            return
    
    # Find executive summary
    executive_summary = result.get("executive_summary", "")
    if not executive_summary and "result" in result and "executive_summary" in result["result"]:
        executive_summary = result["result"]["executive_summary"]
    
    # Display executive summary
    if executive_summary:
        st.markdown("## Executive Summary")
        st.markdown(executive_summary)
    
    # Display issues
    if issues:
        # Add issue count and copy button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"## Issues Found: {len(issues)}")
        with col2:
            # Add copy button logic
            if st.button("📋 Copy All Issues", type="secondary"):
                # Create text for clipboard
                issues_text = create_text_for_copy(issues)
                # Use workaround for clipboard
                st.code(issues_text, language="text")
                st.caption("👆 Copy the text above to your clipboard")
        
        # Group issues by severity
        issues_by_severity = {}
        for issue in issues:
            severity = issue.get("severity", "medium")
            if severity not in issues_by_severity:
                issues_by_severity[severity] = []
            issues_by_severity[severity].append(issue)
        
        # Display issues grouped by severity
        for severity in ["critical", "high", "medium", "low"]:
            if severity in issues_by_severity and issues_by_severity[severity]:
                severity_issues = issues_by_severity[severity]
                st.markdown(f"### {severity.upper()} Severity Issues ({len(severity_issues)})")
                
                for issue in severity_issues:
                    render_issue_card_simple(issue)
    else:
        st.info("No issues were identified in this document.")
        
        # Show raw result for debugging
        with st.expander("Debug: Raw Result Data", expanded=False):
            st.json(result)

def render_issue_card_simple(issue):
    """Render a single issue card using only native Streamlit components."""
    # Get issue properties
    title = issue.get("title", "Untitled Issue")
    description = issue.get("description", "")
    severity = issue.get("severity", "medium")
    category = issue.get("category", "process")
    impact = issue.get("impact", "")
    
    # Set severity color
    severity_colors = {
        "critical": "#F44336",  # Red
        "high": "#FF9800",      # Orange
        "medium": "#2196F3",    # Blue
        "low": "#4CAF50"        # Green
    }
    
    color = severity_colors.get(severity, "#2196F3")
    
    # Create a container with a border
    container = st.container()
    with container:
        # Add a horizontal rule with colored styling
        st.markdown(f"<hr style='height:3px;border:none;color:{color};background-color:{color};margin-bottom:10px;'/>", unsafe_allow_html=True)
        
        # Header row with title and badges
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### {title}")
        
        with col2:
            st.markdown(f"**Severity:** {severity.upper()}")
            st.markdown(f"**Category:** {category.upper()}")
        
        # Description
        st.markdown(description)
        
        # Impact (optional)
        if impact:
            st.markdown("**Potential Impact:**")
            st.info(impact)

def create_text_for_copy(issues):
    """
    Create formatted text content of all issues for copying.
    
    Args:
        issues: List of issue dictionaries
        
    Returns:
        Formatted text string
    """
    text_content = "ISSUES ASSESSMENT REPORT\n"
    text_content += "=" * 50 + "\n\n"
    
    # Group issues by severity
    issues_by_severity = {}
    for issue in issues:
        severity = issue.get("severity", "medium")
        if severity not in issues_by_severity:
            issues_by_severity[severity] = []
        issues_by_severity[severity].append(issue)
    
    # Add issues by severity
    for severity in ["critical", "high", "medium", "low"]:
        if severity in issues_by_severity and issues_by_severity[severity]:
            severity_issues = issues_by_severity[severity]
            text_content += f"{severity.upper()} SEVERITY ISSUES ({len(severity_issues)})\n"
            text_content += "-" * 50 + "\n\n"
            
            for idx, issue in enumerate(severity_issues, 1):
                title = issue.get("title", "Untitled Issue")
                description = issue.get("description", "")
                category = issue.get("category", "process")
                impact = issue.get("impact", "")
                
                text_content += f"{idx}. {title}\n"
                text_content += f"   Category: {category.upper()}\n"
                text_content += f"   Description: {description}\n"
                
                if impact:
                    text_content += f"   Potential Impact: {impact}\n"
                
                text_content += "\n"
    
    return text_content

def render_actions_result(result):
    """Render action item extraction results."""
    # Placeholder for now - will implement when needed
    st.info("Action item rendering will be implemented in a future update.")
    st.json(result)

def render_analysis_result(result):
    """Render framework analysis results."""
    # Placeholder for now - will implement when needed
    st.info("Framework analysis rendering will be implemented in a future update.")
    st.json(result)