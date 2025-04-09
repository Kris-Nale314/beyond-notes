"""
Improved rendering components for the Beyond Notes application.
Handles flexible formatting and display of assessment results with robust data extraction.
"""

import streamlit as st
import json
from datetime import datetime
import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger("beyond-notes-renderers")

def render_assessment_result(result: Dict[str, Any], assessment_type: str) -> str:
    """
    Render assessment result based on type with improved data extraction.
    Works with different output structures from improved agents.
    
    Args:
        result: The assessment result dictionary
        assessment_type: The type of assessment ("distill", "assess", "extract", "analyze")
        
    Returns:
        Rendered HTML or text content
    """
    logger.info(f"Rendering assessment type: {assessment_type}")
    
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
        return "Unknown assessment type"

def render_summary_result(result: Dict[str, Any], format_type: Optional[str] = None) -> str:
    """
    Render summary results with improved data extraction to handle
    different output structures from the improved agents.
    
    Args:
        result: The assessment result dictionary
        format_type: Optional format type (executive, comprehensive, bullet_points, narrative)
        
    Returns:
        Rendered HTML content
    """
    # Log structure for debugging
    logger.info(f"Rendering summary with format: {format_type}")
    logger.info(f"Result top-level keys: {list(result.keys())}")
    
    # Check for formatter errors
    if "error" in result and result["error"]:
        st.error(f"The formatter encountered errors: {result['error']}")
        
        # Show raw data for debugging
        with st.expander("Show Raw Result Data", expanded=True):
            st.json(result)
        return f"<div class='error-message'>Error: {result['error']}</div>"
    
    # Get the format type from the result if not specified
    if not format_type:
        format_type = result.get("metadata", {}).get("user_options", {}).get("format", "executive")
    
    # Find summary content using flexible approach
    summary_content = extract_summary_content(result)
    
    if not summary_content:
        error_msg = "Could not find summary content in the result structure"
        logger.error(error_msg)
        
        # Show raw data for debugging
        with st.expander("Show Raw Result Structure", expanded=True):
            st.json(result)
            
        return f"<div class='error-message'>{error_msg}</div>"
    
    # Get topics if available
    topics = extract_topics(result)
    
    # Create HTML representation
    html_content = format_summary_as_html(summary_content, topics, format_type)
    
    # Display the summary
    st.markdown(html_content, unsafe_allow_html=True)
    
    # Add statistics if available
    display_summary_statistics(result)
    
    # Return the generated HTML for potential additional use
    return html_content

def render_issues_result(result: Dict[str, Any]) -> str:
    """
    Render issues assessment results with improved data extraction.
    Finds issues in different locations in complex output structures.
    
    Args:
        result: The assessment result dictionary
        
    Returns:
        Rendered HTML content
    """
    # Log structure for debugging
    logger = logging.getLogger("beyond-notes-issues")
    logger.info(f"Result top-level keys: {list(result.keys())}")
    if "result" in result:
        logger.info(f"Result.result keys: {list(result['result'].keys())}")
    if "formatted" in result:
        logger.info(f"Result.formatted keys: {list(result['formatted'].keys())}")
    
    # Check for formatter errors
    if "error" in result and result["error"]:
        st.error(f"The formatter encountered errors: {result['error']}")
        
        # Show raw data for debugging
        with st.expander("Show Raw Result Data", expanded=True):
            st.json(result)
        return
    
    # Extract data using flexible approach
    metadata = result.get("metadata", {})
    
    # Find issues with comprehensive search
    issues = extract_issues(result)
    
    # Find executive summary
    executive_summary = extract_executive_summary(result)
    
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
            # Use evaluated_severity if available, otherwise use severity
            severity = issue.get("evaluated_severity", issue.get("severity", "medium")).lower()
            if severity not in issues_by_severity:
                issues_by_severity[severity] = []
            issues_by_severity[severity].append(issue)
        
        # Display issues grouped by severity
        for severity in ["critical", "high", "medium", "low"]:
            if severity in issues_by_severity and issues_by_severity[severity]:
                severity_issues = issues_by_severity[severity]
                st.markdown(f"### {severity.upper()} Severity Issues ({len(severity_issues)})")
                
                for issue in severity_issues:
                    render_issue_card(issue)
                    
        # Display statistics if available
        display_issues_statistics(result, issues)
    else:
        st.info("No issues were identified in this document.")
        
        # Show raw result for debugging
        with st.expander("Debug: Raw Result Data", expanded=False):
            st.json(result)
            
    return "Issues rendered successfully"

def render_actions_result(result: Dict[str, Any]) -> str:
    """
    Render action item extraction results with improved data extraction.
    Finds action items in different locations in complex output structures.
    
    Args:
        result: The assessment result dictionary
        
    Returns:
        Rendered HTML content
    """
    # Extract action items
    action_items = extract_action_items(result)
    
    if action_items:
        # Display action items count
        st.markdown(f"## Action Items Found: {len(action_items)}")
        
        # Group action items by priority
        items_by_priority = {}
        for item in action_items:
            # Use evaluated_priority if available, otherwise use priority
            priority = item.get("evaluated_priority", item.get("priority", "medium")).lower()
            if priority not in items_by_priority:
                items_by_priority[priority] = []
            items_by_priority[priority].append(item)
        
        # Display action items grouped by priority
        for priority in ["high", "medium", "low"]:
            if priority in items_by_priority and items_by_priority[priority]:
                priority_items = items_by_priority[priority]
                st.markdown(f"### {priority.upper()} Priority Actions ({len(priority_items)})")
                
                for item in priority_items:
                    render_action_item_card(item)
    else:
        st.info("No action items were identified in this document.")
        
        # Show raw result for debugging
        with st.expander("Debug: Raw Result Data", expanded=False):
            st.json(result)
            
    return "Action items rendered successfully"

def render_analysis_result(result: Dict[str, Any]) -> str:
    """
    Render framework analysis results.
    
    Args:
        result: The assessment result dictionary
        
    Returns:
        Rendered HTML content
    """
    # Placeholder implementation - will enhance when needed
    st.info("Framework analysis rendering will be implemented in a future update.")
    st.json(result)
    return "Analysis rendering not yet implemented"

# ==========================================
# Extraction Helper Functions
# ==========================================

def extract_summary_content(result: Dict[str, Any]) -> Optional[str]:
    """
    Extract summary content from various possible locations in the result.
    Uses a flexible approach to find the summary.
    
    Args:
        result: The assessment result
        
    Returns:
        The summary text if found, otherwise None
    """
    # Try common paths for summary content
    potential_paths = [
        result,                               # Look directly in result
        result.get("result", {}),             # Look in result.result
        result.get("formatted", {}),          # Look in result.formatted
        result.get("extracted_info", {})      # Look in result.extracted_info 
    ]
    
    # Look for common summary fields
    summary_fields = ["summary", "overview", "content", "text"]
    
    # Try each potential path and field combination
    for path in potential_paths:
        if isinstance(path, dict):
            for field in summary_fields:
                if field in path and path[field] and isinstance(path[field], str):
                    logger.info(f"Found summary content in field: {field}")
                    return path[field]
            
            # Also check for 'key_points' which might contain the summary
            if "key_points" in path and isinstance(path["key_points"], list):
                key_points = path["key_points"]
                if key_points:
                    # Check if key_points contains text points
                    points_with_text = []
                    for point in key_points:
                        if isinstance(point, dict):
                            text = point.get("text") or point.get("point", "")
                            if text:
                                points_with_text.append(text)
                    
                    if points_with_text:
                        logger.info(f"Created summary from key_points array with {len(points_with_text)} items")
                        return "\n\n".join(points_with_text)
    
    # If no summary content found, try deep search
    logger.warning("Could not find summary content through standard paths, trying recursive search")
    return find_summary_content_recursively(result)

def find_summary_content_recursively(obj: Any, max_depth: int = 3) -> Optional[str]:
    """
    Recursively search for summary content in nested structure.
    
    Args:
        obj: The object to search in
        max_depth: Maximum recursion depth
        
    Returns:
        The summary text if found, otherwise None
    """
    if max_depth <= 0:
        return None
        
    if isinstance(obj, dict):
        # Check for common summary fields
        for field in ["summary", "overview", "content", "text"]:
            if field in obj and isinstance(obj[field], str) and len(obj[field]) > 100:
                return obj[field]
                
        # Check all nested objects
        for key, value in obj.items():
            result = find_summary_content_recursively(value, max_depth - 1)
            if result:
                return result
                
    elif isinstance(obj, list) and len(obj) > 0:
        # Check if this is a list of text items (key points)
        if all(isinstance(item, dict) for item in obj):
            if all(any(text_field in item for text_field in ["text", "point", "content"]) for item in obj):
                # This looks like a list of text points
                points = []
                for item in obj:
                    for field in ["text", "point", "content"]:
                        if field in item and isinstance(item[field], str):
                            points.append(item[field])
                            break
                if points:
                    return "\n\n".join(points)
    
    return None

def extract_topics(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract topics from various possible locations in the result.
    
    Args:
        result: The assessment result
        
    Returns:
        List of topic dictionaries
    """
    # Try common paths for topics
    potential_paths = [
        result,                               # Look directly in result
        result.get("result", {}),             # Look in result.result
        result.get("formatted", {}),          # Look in result.formatted
        result.get("extracted_info", {})      # Look in result.extracted_info 
    ]
    
    # Try each potential path
    for path in potential_paths:
        if isinstance(path, dict) and "topics" in path and isinstance(path["topics"], list):
            logger.info(f"Found topics array with {len(path['topics'])} items")
            return path["topics"]
    
    # If not found, return empty list
    return []

def extract_issues(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract issues from various possible locations in the result.
    
    Args:
        result: The assessment result
        
    Returns:
        List of issue dictionaries
    """
    # Try common paths for issues
    potential_paths = [
        result,                                       # issues directly in result
        result.get("result", {}),                     # issues in result.result
        result.get("formatted", {}),                  # issues in result.formatted
        result.get("extracted_info", {})              # issues in result.extracted_info
    ]
    
    # Log what we find along each path
    for i, path in enumerate(potential_paths):
        if isinstance(path, dict) and "issues" in path and isinstance(path["issues"], list):
            logger.info(f"Found issues in path {i}: {len(path['issues'])} items")
            return path["issues"]
    
    # If still no issues, try a deep search
    logger.info("Performing deep search for issues array")
    issues = find_issues_recursively(result)
    logger.info(f"Deep search found {len(issues)} issues")
    return issues

def find_issues_recursively(obj: Any, max_depth: int = 4) -> List[Dict[str, Any]]:
    """
    Recursively search for 'issues' array in nested dictionaries.
    
    Args:
        obj: Object to search in
        max_depth: Maximum recursion depth
        
    Returns:
        List of issue dictionaries
    """
    if max_depth <= 0:
        return []
        
    if isinstance(obj, dict):
        # Direct check for issues
        if "issues" in obj and isinstance(obj["issues"], list):
            # Verify these look like issues (have title, description, or severity)
            if all(isinstance(item, dict) for item in obj["issues"]):
                if any(any(key in item for key in ["title", "description", "severity"]) for item in obj["issues"]):
                    return obj["issues"]
            
        # Recursive check in all dictionary values
        for key, value in obj.items():
            result = find_issues_recursively(value, max_depth - 1)
            if result:
                return result
                
    # Check in lists of dictionaries
    elif isinstance(obj, list) and len(obj) > 0 and all(isinstance(item, dict) for item in obj):
        # This might be the issues array itself if it has expected keys
        if all(any(key in item for key in ["title", "description", "severity"]) for item in obj):
            return obj
            
        # Otherwise search in each list item
        for item in obj:
            result = find_issues_recursively(item, max_depth - 1)
            if result:
                return result
                
    return []

def extract_executive_summary(result: Dict[str, Any]) -> Optional[str]:
    """
    Extract executive summary from various possible locations in the result.
    
    Args:
        result: The assessment result
        
    Returns:
        Executive summary text if found, otherwise None
    """
    # Try common paths for executive summary
    executive_summary = None
    
    # Check direct paths first
    potential_paths = [
        result,                              # executive_summary directly in result
        result.get("result", {}),            # executive_summary in result.result
        result.get("formatted", {})          # executive_summary in result.formatted
    ]
    
    for path in potential_paths:
        if isinstance(path, dict) and "executive_summary" in path:
            executive_summary = path["executive_summary"]
            if executive_summary:
                return executive_summary
    
    # If not found, check in review if available
    if "review" in result and isinstance(result["review"], dict):
        review = result["review"]
        if "executive_summary" in review:
            return review["executive_summary"]
        if "review_summary" in review:
            return review["review_summary"]
    
    # If still not found, check for overall_assessment
    if "overall_assessment" in result and isinstance(result["overall_assessment"], dict):
        assessment = result["overall_assessment"]
        if "executive_summary" in assessment:
            return assessment["executive_summary"]
        if "summary" in assessment:
            return assessment["summary"]
    
    return None

def extract_action_items(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract action items from various possible locations in the result.
    
    Args:
        result: The assessment result
        
    Returns:
        List of action item dictionaries
    """
    # Try common paths for action items
    potential_paths = [
        result,                                       # action_items directly in result
        result.get("result", {}),                     # action_items in result.result
        result.get("formatted", {}),                  # action_items in result.formatted
        result.get("extracted_info", {})              # action_items in result.extracted_info
    ]
    
    # Log what we find along each path
    for i, path in enumerate(potential_paths):
        if isinstance(path, dict) and "action_items" in path and isinstance(path["action_items"], list):
            logger.info(f"Found action_items in path {i}: {len(path['action_items'])} items")
            return path["action_items"]
    
    # If still no action items, try a deep search
    logger.info("Performing deep search for action_items array")
    action_items = find_action_items_recursively(result)
    logger.info(f"Deep search found {len(action_items)} action items")
    return action_items

def find_action_items_recursively(obj: Any, max_depth: int = 4) -> List[Dict[str, Any]]:
    """
    Recursively search for 'action_items' array in nested dictionaries.
    
    Args:
        obj: Object to search in
        max_depth: Maximum recursion depth
        
    Returns:
        List of action item dictionaries
    """
    if max_depth <= 0:
        return []
        
    if isinstance(obj, dict):
        # Direct check for action_items
        if "action_items" in obj and isinstance(obj["action_items"], list):
            # Verify these look like action items (have description, owner, or due_date)
            if all(isinstance(item, dict) for item in obj["action_items"]):
                if any(any(key in item for key in ["description", "owner", "due_date"]) for item in obj["action_items"]):
                    return obj["action_items"]
            
        # Recursive check in all dictionary values
        for key, value in obj.items():
            result = find_action_items_recursively(value, max_depth - 1)
            if result:
                return result
                
    # Check in lists of dictionaries
    elif isinstance(obj, list) and len(obj) > 0 and all(isinstance(item, dict) for item in obj):
        # This might be the action items array itself if it has expected keys
        if all(any(key in item for key in ["description", "owner", "due_date"]) for item in obj):
            return obj
            
        # Otherwise search in each list item
        for item in obj:
            result = find_action_items_recursively(item, max_depth - 1)
            if result:
                return result
                
    return []

# ==========================================
# Formatting Helper Functions
# ==========================================

def format_summary_as_html(content: str, topics: List[Dict[str, Any]], format_type: str) -> str:
    """
    Format summary content as HTML based on format type.
    
    Args:
        content: The summary text content
        topics: List of topic dictionaries
        format_type: The format type (executive, comprehensive, bullet_points, narrative)
        
    Returns:
        Formatted HTML content
    """
    # Default CSS classes
    css_classes = f"summary-content {format_type}-format"
    
    # Format based on type
    if format_type == "bullet_points":
        # Convert to bullet points if not already
        if not any(line.strip().startswith(("- ", "* ", "• ")) for line in content.split("\n") if line.strip()):
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            content = "\n".join([f"- {p}" for p in paragraphs])
        
        # Wrap in div with bullet point styling
        html = f"""
        <div class="{css_classes}">
            <div class="bullet-list">
                {content}
            </div>
        </div>
        """
    elif format_type == "executive":
        # Keep it concise, wrap in executive summary styling
        html = f"""
        <div class="{css_classes}">
            <div class="executive-summary">
                {content}
            </div>
        </div>
        """
    elif format_type == "comprehensive":
        # Include topics if available
        topics_html = ""
        if topics:
            topics_html = "<h3>Topics</h3><ul>"
            for topic in topics:
                topic_name = topic.get("topic", "")
                key_points = topic.get("key_points", [])
                
                if topic_name:
                    topics_html += f"<li><strong>{topic_name}</strong>"
                    
                    if key_points:
                        topics_html += "<ul>"
                        for point in key_points:
                            if isinstance(point, str):
                                topics_html += f"<li>{point}</li>"
                            elif isinstance(point, dict) and "point" in point:
                                topics_html += f"<li>{point['point']}</li>"
                        topics_html += "</ul>"
                        
                    topics_html += "</li>"
            topics_html += "</ul>"
        
        # Wrap in comprehensive summary styling
        html = f"""
        <div class="{css_classes}">
            <div class="comprehensive-summary">
                {content}
            </div>
            {topics_html}
        </div>
        """
    else:  # narrative or default
        # Format as flowing narrative
        html = f"""
        <div class="{css_classes}">
            <div class="narrative-summary">
                {content}
            </div>
        </div>
        """
    
    # Add CSS styles to make it look nice
    html += """
    <style>
        .summary-content {
            margin: 1.5rem 0;
            line-height: 1.6;
            animation: fadeIn 0.5s ease;
        }
        
        .executive-format {
            font-size: 1.1rem;
        }
        
        .comprehensive-format {
            font-size: 1rem;
        }
        
        .bullet-list {
            padding-left: 1.5rem;
        }
        
        .narrative-summary {
            font-size: 1.05rem;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """
    
    return html

def render_issue_card(issue: Dict[str, Any]) -> None:
    """
    Render a single issue card using only native Streamlit components.
    
    Args:
        issue: Issue dictionary with properties like title, description, severity, etc.
    """
    # Get issue properties
    title = issue.get("title", "Untitled Issue")
    description = issue.get("description", "")
    
    # Use evaluated_severity if available, otherwise use severity
    severity = issue.get("evaluated_severity", issue.get("severity", "medium")).lower()
    category = issue.get("category", "process")
    impact = issue.get("potential_impact", issue.get("impact", ""))
    
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
        
        # Impact and recommendations (optional)
        if impact:
            st.markdown("**Potential Impact:**")
            st.info(impact)
            
        # Recommendations if available
        recommendations = issue.get("recommendations", issue.get("suggested_recommendations", []))
        if recommendations:
            st.markdown("**Recommendations:**")
            for rec in recommendations:
                st.markdown(f"- {rec}")

def render_action_item_card(item: Dict[str, Any]) -> None:
    """
    Render a single action item card using Streamlit components.
    
    Args:
        item: Action item dictionary with properties like description, owner, due_date, etc.
    """
    # Get action item properties
    description = item.get("description", "Untitled Action Item")
    owner = item.get("owner", "Unassigned")
    due_date = item.get("due_date", "No deadline")
    
    # Use evaluated_priority if available, otherwise use priority
    priority = item.get("evaluated_priority", item.get("priority", "medium")).lower()
    is_actionable = item.get("is_actionable", True)
    
    # Set priority color
    priority_colors = {
        "high": "#FF9800",      # Orange
        "medium": "#2196F3",    # Blue
        "low": "#4CAF50"        # Green
    }
    
    color = priority_colors.get(priority, "#2196F3")
    
    # Create a container with a border
    container = st.container()
    with container:
        # Add a horizontal rule with colored styling
        st.markdown(f"<hr style='height:3px;border:none;color:{color};background-color:{color};margin-bottom:10px;'/>", unsafe_allow_html=True)
        
        # Header row with description
        st.markdown(f"### {description}")
        
        # Details in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**Owner:** {owner}")
        
        with col2:
            st.markdown(f"**Due Date:** {due_date}")
        
        with col3:
            st.markdown(f"**Priority:** {priority.upper()}")
            
        # Actionability
        if not is_actionable:
            st.warning("⚠️ This item may need more clarification to be actionable.")

def create_text_for_copy(issues: List[Dict[str, Any]]) -> str:
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
        # Use evaluated_severity if available, otherwise use severity
        severity = issue.get("evaluated_severity", issue.get("severity", "medium")).lower()
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
                impact = issue.get("potential_impact", issue.get("impact", ""))
                
                text_content += f"{idx}. {title}\n"
                text_content += f"   Category: {category.upper()}\n"
                text_content += f"   Description: {description}\n"
                
                if impact:
                    text_content += f"   Potential Impact: {impact}\n"
                
                text_content += "\n"
    
    return text_content

# ==========================================
# Statistics Display Functions
# ==========================================

def display_summary_statistics(result: Dict[str, Any]) -> None:
    """
    Display statistics for summary results.
    
    Args:
        result: The assessment result
    """
    # Try to find statistics in different locations
    statistics = None
    potential_paths = [
        result.get("statistics", {}),
        result.get("result", {}).get("statistics", {}),
        result.get("formatted", {}).get("statistics", {})
    ]
    
    for path in potential_paths:
        if path and isinstance(path, dict) and any(key in path for key in ["original_word_count", "summary_word_count"]):
            statistics = path
            break
            
    if not statistics:
        return
        
    # Get key statistics
    original_words = statistics.get("original_word_count", 0)
    summary_words = statistics.get("summary_word_count", 0)
    topics_count = statistics.get("topics_covered", 0)
    
    # Calculate compression ratio
    compression_ratio = 0
    if original_words > 0 and summary_words > 0:
        compression_ratio = (summary_words / original_words) * 100
    
    # Display in a nicely formatted grid
    if original_words > 0:
        st.markdown("### Summary Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Original Word Count", f"{original_words:,}")
            
        with col2:
            st.metric("Summary Word Count", f"{summary_words:,}")
            
        with col3:
            st.metric("Compression Ratio", f"{compression_ratio:.1f}%")
            
        # Add topics count if available
        if topics_count > 0:
            st.metric("Topics Covered", topics_count)

def display_issues_statistics(result: Dict[str, Any], issues: List[Dict[str, Any]]) -> None:
    """
    Display statistics for issue assessment results.
    
    Args:
        result: The assessment result
        issues: The extracted issues list
    """
    if not issues:
        return
        
    # Try to find statistics in different locations
    statistics = None
    potential_paths = [
        result.get("statistics", {}),
        result.get("result", {}).get("statistics", {}),
        result.get("formatted", {}).get("statistics", {})
    ]
    
    for path in potential_paths:
        if path and isinstance(path, dict):
            statistics = path
            break
    
    # Count issues by severity
    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for issue in issues:
        severity = issue.get("evaluated_severity", issue.get("severity", "medium")).lower()
        if severity in severity_counts:
            severity_counts[severity] += 1
    
    # Display statistics
    st.markdown("### Assessment Overview")
    
    # Display metrics grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Issues", len(issues))
        
    with col2:
        critical_high = severity_counts["critical"] + severity_counts["high"]
        st.metric("Critical/High Issues", critical_high)
        
    with col3:
        # Try to get processing time from metadata
        processing_time = result.get("metadata", {}).get("processing_time", 0)
        st.metric("Processing Time", f"{processing_time:.1f}s")
    
    # Display severity distribution
    st.markdown("#### Severity Distribution")
    
    # Create a horizontal bar chart using HTML/CSS
    for severity in ["critical", "high", "medium", "low"]:
        count = severity_counts.get(severity, 0)
        percentage = (count / len(issues) * 100) if issues else 0
        
        # Colors based on severity
        colors = {
            "critical": "#F44336",
            "high": "#FF9800",
            "medium": "#2196F3",
            "low": "#4CAF50"
        }
        
        # Create bar
        st.markdown(f"""
        <div style="margin-bottom: 8px;">
            <div style="display: flex; align-items: center; margin-bottom: 4px;">
                <div style="flex-grow: 1; font-size: 0.9rem; text-transform: capitalize;">{severity}</div>
                <div style="font-weight: 500; font-size: 0.9rem;">{count} ({percentage:.1f}%)</div>
            </div>
            <div style="height: 8px; background-color: rgba(0,0,0,0.1); border-radius: 4px; overflow: hidden;">
                <div style="width: {percentage}%; height: 100%; background-color: {colors[severity]};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)