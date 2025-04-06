"""
Reusable UI components for the Beyond Notes application.
Provides consistent interface elements across pages.
"""

import streamlit as st
import os
from pathlib import Path

def page_header(title, subtitle=None):
    """Render a consistent page header."""
    st.markdown(f'<div class="main-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="subheader">{subtitle}</div>', unsafe_allow_html=True)

def section_header(title, number=None):
    """Render a consistent section header with optional number."""
    if number:
        st.markdown(f'<div class="section-header"><span class="section-number">{number}</span>{title}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def display_document_preview(document):
    """Display a preview of the loaded document with enhanced styling."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Document Preview")
        
        preview_length = min(1000, len(document.text))
        preview_text = document.text[:preview_length]
        if len(document.text) > preview_length:
            preview_text += "..."
        
        with st.expander("Document Content Preview", expanded=False):
            st.text_area("", preview_text, height=200, disabled=True)
    
    with col2:
        st.markdown('<div class="document-meta">', unsafe_allow_html=True)
        
        st.markdown('<div class="document-meta-label">Filename</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="document-meta-value">{document.filename}</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="document-meta-label">Word Count</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="document-meta-value">{document.word_count:,}</div>', unsafe_allow_html=True)
        
        file_size = os.path.getsize(str(document.file_path)) if hasattr(document, 'file_path') and document.file_path else 0
        if file_size > 0:
            size_kb = file_size / 1024
            size_display = f"{size_kb:.1f} KB"
            st.markdown('<div class="document-meta-label">File Size</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="document-meta-value">{size_display}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def select_severity_filter(default="all"):
    """Create a severity filter selector with proper styling."""
    severities = [
        {"value": "all", "label": "All Severities", "desc": "Show issues of all severity levels"},
        {"value": "critical", "label": "Critical Only", "desc": "Show only critical severity issues"},
        {"value": "high", "label": "High & Critical", "desc": "Show high and critical severity issues"},
        {"value": "medium", "label": "Medium & Above", "desc": "Show medium, high, and critical issues"}
    ]
    
    # Create a more visually appealing radio selector
    selected = st.radio(
        "Filter by Severity",
        options=[s["value"] for s in severities],
        format_func=lambda x: next((s["label"] for s in severities if s["value"] == x), x),
        index=[s["value"] for s in severities].index(default),
        horizontal=True
    )
    
    # Display description of selected option
    selected_desc = next((s["desc"] for s in severities if s["value"] == selected), "")
    st.caption(selected_desc)
    
    return selected

# utils/ui/components.py

def display_statistics_bar(current_value, total_value, label, color="#2196F3"):
    """
    Display a statistics bar with percentage.
    
    Args:
        current_value: Current value to display
        total_value: Maximum value for the bar
        label: Label to display next to the bar
        color: Bar color (hex or name)
    """
    # Calculate percentage
    if total_value <= 0:
        percentage = 0
    else:
        percentage = min(100, round((current_value / total_value) * 100))
    
    # Create the progress bar HTML
    html = f"""
    <div style="margin: 10px 0;">
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="flex-grow: 1; font-size: 0.9rem;">{label}</div>
            <div style="font-weight: 500; font-size: 0.9rem;">{percentage}%</div>
        </div>
        <div style="background-color: rgba(0, 0, 0, 0.1); height: 8px; border-radius: 4px; overflow: hidden;">
            <div style="background-color: {color}; width: {percentage}%; height: 100%;"></div>
        </div>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)

def display_metric_grid(metrics, columns=3):
    """
    Display a grid of metrics.
    
    Args:
        metrics: List of dicts with 'label' and 'value' keys
        columns: Number of columns in the grid
    """
    # Create columns
    cols = st.columns(columns)
    
    # Place metrics in columns
    for i, metric in enumerate(metrics):
        col_index = i % columns
        with cols[col_index]:
            st.metric(
                label=metric.get("label", ""),
                value=metric.get("value", ""),
                delta=metric.get("delta", None),
                delta_color=metric.get("delta_color", "normal")
            )

def create_copy_button(content, button_text="Copy", success_text="Copied!"):
    """
    Create a button that copies content to clipboard when clicked.
    
    Args:
        content: Text content to copy
        button_text: Text to display on the button
        success_text: Text to display after successful copy
        
    Returns:
        A Streamlit button element
    """
    import base64
    
    # Encode the content to avoid JS injection and formatting issues
    encoded_content = base64.b64encode(content.encode()).decode()
    
    # Create a unique ID for this button
    import uuid
    button_id = f"copy_button_{uuid.uuid4().hex[:8]}"
    
    # Create the JavaScript function
    copy_script = f"""
    <script>
    function copyToClipboard_{button_id}() {{
        const content = atob("{encoded_content}");
        navigator.clipboard.writeText(content).then(
            function() {{
                // Success
                const button = document.getElementById('{button_id}');
                const originalText = button.innerText;
                button.innerText = '{success_text}';
                button.style.backgroundColor = '#4CAF50';
                setTimeout(function() {{
                    button.innerText = originalText;
                    button.style.backgroundColor = '';
                }}, 2000);
            }},
            function() {{
                // Failure
                alert('Failed to copy content to clipboard');
            }}
        );
    }}
    </script>
    """
    
    # Create the button HTML
    button_html = f"""
    {copy_script}
    <button 
        id="{button_id}" 
        onclick="copyToClipboard_{button_id}()" 
        style="background-color: #2196F3; color: white; border: none; 
               padding: 8px 16px; border-radius: 4px; cursor: pointer;
               display: inline-flex; align-items: center; font-weight: 500;">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" 
             fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" 
             stroke-linejoin="round" style="margin-right: 6px;">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
        {button_text}
    </button>
    """
    
    return st.markdown(button_html, unsafe_allow_html=True)

def render_issues_statistics(result):
    """Render statistics for issue assessment results."""
    statistics = result.get("statistics", {})
    metadata = result.get("metadata", {})
    issues = result.get("issues", [])
    
    if not issues:
        return
    
    st.markdown("## Assessment Overview")
    
    # Create a card-like container
    st.markdown(
        """
        <div style="background-color: rgba(0, 0, 0, 0.1); 
                    border-radius: 10px; 
                    padding: 20px; 
                    margin-bottom: 20px;">
        """, 
        unsafe_allow_html=True
    )
    
    # Key metrics
    total_issues = len(issues)
    word_count = metadata.get("word_count", 0)
    processing_time = metadata.get("processing_time", 0)
    
    # Group issues by severity
    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for issue in issues:
        severity = issue.get("severity", "medium")
        if severity in severity_counts:
            severity_counts[severity] += 1
    
    # Display metrics grid
    metrics = [
        {"label": "Total Issues", "value": total_issues},
        {"label": "Document Size", "value": f"{word_count:,} words"},
        {"label": "Processing Time", "value": f"{processing_time:.1f}s"}
    ]
    display_metric_grid(metrics, columns=3)
    
    # Display severity distribution
    st.markdown("### Issue Severity Distribution")
    
    # Calculate percentages for bars
    for severity in ["critical", "high", "medium", "low"]:
        count = severity_counts.get(severity, 0)
        display_statistics_bar(
            count, 
            total_issues, 
            f"{severity.title()} ({count})",
            color={"critical": "#F44336", "high": "#FF9800", "medium": "#2196F3", "low": "#4CAF50"}[severity]
        )
    
    # Close the container div
    st.markdown("</div>", unsafe_allow_html=True)