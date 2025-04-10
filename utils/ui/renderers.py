# utils/ui/renderers.py

"""
UI Rendering Module

Provides standardized rendering functions for different assessment types.
Uses the DataAccessor to ensure consistent data access regardless of source.
"""

import streamlit as st
import json
from typing import Dict, Any, List, Optional, Union, Tuple
import logging

from utils.accessor import DataAccessor

logger = logging.getLogger("beyond-notes.renderer")

# =====================================================================
# Main Assessment Renderers
# =====================================================================

def render_assessment_result(result: Dict[str, Any], assessment_type: str, context=None) -> None:
    """
    Main entry point for rendering assessment results based on type.
    
    Args:
        result: Assessment result dictionary
        assessment_type: The type of assessment ("distill", "assess", "extract", "analyze")
        context: Optional ProcessingContext object for enhanced rendering
    """
    logger.info(f"Rendering assessment type: {assessment_type}")
    
    # Apply shared styles for all renderers
    apply_common_styles()
    
    # Route to appropriate renderer based on assessment type
    if assessment_type == "assess":
        render_issues_result(result, context)
    elif assessment_type == "distill":
        render_summary_result(result, context)
    elif assessment_type == "extract":
        render_action_items_result(result, context)
    elif assessment_type == "analyze":
        render_analysis_result(result, context)
    else:
        st.warning(f"Unknown assessment type: {assessment_type}")
        st.json(result)

# =====================================================================
# Summary Rendering (Distill Assessment)
# =====================================================================

def render_summary_result(result: Dict[str, Any], context=None) -> None:
    """
    Render summary results with a clean, document-like layout.
    Uses DataAccessor for standardized data access.
    
    Args:
        result: Summary result dictionary
        context: Optional ProcessingContext object for enhanced rendering with evidence
    """
    # Get standardized data
    data = DataAccessor.get_summary_data(result, context)
    
    # Validate the data
    is_valid, error_message = DataAccessor.validate_data(data, "distill")
    if not is_valid:
        st.warning(f"Cannot render summary: {error_message}")
        if st.checkbox("Show raw data for debugging"):
            st.json(result)
        return
    
    # Get key data elements
    format_type = data["format_type"]
    summary_content = data["summary_content"]
    key_points = data["key_points"]
    topics = data["topics"]
    executive_summary = data["executive_summary"]
    statistics = data["statistics"]
    
    # Apply summary styling
    apply_summary_styles()
    
    # Add a title and brief description based on format type
    format_display_names = {
        "executive": "Executive Summary",
        "comprehensive": "Comprehensive Summary",
        "bullet_points": "Summary - Key Points",
        "narrative": "Detailed Summary"
    }
    summary_title = format_display_names.get(format_type, "Document Summary")
    
    # Add a title
    st.markdown(f"<div class='bn-summary'><h1>{summary_title}</h1>", unsafe_allow_html=True)
    
    # Format description
    format_descriptions = {
        "executive": "A concise overview highlighting the most essential information",
        "comprehensive": "A detailed summary covering all significant aspects of the document",
        "bullet_points": "Key information organized into easily scannable points",
        "narrative": "A flowing narrative that captures the document's content"
    }
    st.caption(format_descriptions.get(format_type, ""))
    
    # Render content based on format type
    if format_type == "bullet_points":
        render_bullet_point_summary(data)
    else:
        # Executive, comprehensive, and narrative all use prose format
        render_prose_summary(data)
    
    # Display key statistics in a clean, subtle way
    display_summary_statistics(statistics)
    
    # Close the main div
    st.markdown("</div>", unsafe_allow_html=True)

def render_prose_summary(data: Dict[str, Any]) -> None:
    """
    Render a prose-style summary (executive, comprehensive, or narrative).
    
    Args:
        data: Standardized summary data dictionary
    """
    # Extract data
    summary_content = data["summary_content"]
    executive_summary = data["executive_summary"]
    topics = data["topics"]
    format_type = data["format_type"]
    
    # Start with executive summary if available
    if executive_summary:
        st.markdown("<div class='bn-overview'>", unsafe_allow_html=True)
        st.markdown(executive_summary)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add a separator if we have a main summary following
        if summary_content:
            st.markdown("<hr />", unsafe_allow_html=True)
    
    # Main summary content
    if summary_content:
        st.markdown("<div class='summary-content'>", unsafe_allow_html=True)
        
        # Split into paragraphs for better rendering
        paragraphs = summary_content.split('\n\n')
        for paragraph in paragraphs:
            if paragraph.strip():
                st.markdown(f"<p>{paragraph}</p>", unsafe_allow_html=True)
                
        st.markdown("</div>", unsafe_allow_html=True)
    elif not executive_summary:
        st.info("No summary content available")
    
    # Render topics if available and format is comprehensive
    if topics and format_type == "comprehensive":
        st.markdown("<hr />", unsafe_allow_html=True)
        st.markdown("<h2>Key Topics</h2>", unsafe_allow_html=True)
        
        for topic in topics:
            if isinstance(topic, dict):
                topic_name = topic.get("topic", "")
                details = topic.get("details", "")
                
                if topic_name:
                    st.markdown(f"<div class='topic-heading'>{topic_name}</div>", unsafe_allow_html=True)
                    if details:
                        st.markdown(details)
                        
                    # Render topic key points if available
                    topic_points = topic.get("key_points", [])
                    if topic_points and isinstance(topic_points, list):
                        for point in topic_points:
                            if isinstance(point, str):
                                st.markdown(f"- {point}")
                            elif isinstance(point, dict) and "text" in point:
                                st.markdown(f"- {point['text']}")

def render_bullet_point_summary(data: Dict[str, Any]) -> None:
    """
    Render a bullet point summary with a clean, professional layout.
    
    Args:
        data: Standardized summary data dictionary
    """
    # Extract data
    key_points = data["key_points"]
    topics = data["topics"]
    executive_summary = data["executive_summary"]
    
    # Start with executive summary if available
    if executive_summary:
        st.markdown("<div class='bn-overview'>", unsafe_allow_html=True)
        st.markdown(executive_summary)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<hr />", unsafe_allow_html=True)
    
    # If we have topics, organize by topic
    if topics and isinstance(topics, list) and len(topics) > 0:
        for topic in topics:
            if isinstance(topic, dict):
                topic_name = topic.get("topic", "")
                
                if topic_name:
                    st.markdown(f"<div class='topic-heading'>{topic_name}</div>", unsafe_allow_html=True)
                    
                    # Render topic key points
                    topic_points = topic.get("key_points", [])
                    if topic_points and isinstance(topic_points, list):
                        for point in topic_points:
                            if isinstance(point, str):
                                st.markdown(f"<div class='bullet-point'>{point}</div>", unsafe_allow_html=True)
                            elif isinstance(point, dict) and "text" in point:
                                st.markdown(f"<div class='bullet-point'>{point['text']}</div>", unsafe_allow_html=True)
    
    # Otherwise render as a flat list
    elif key_points and isinstance(key_points, list):
        st.markdown("<div class='bn-section'>", unsafe_allow_html=True)
        
        # Group points by topic if available
        points_by_topic = {}
        for point in key_points:
            if isinstance(point, dict):
                topic = point.get("topic", "General")
                if topic not in points_by_topic:
                    points_by_topic[topic] = []
                points_by_topic[topic].append(point)
        
        # If points are grouped by topic
        if len(points_by_topic) > 1:
            for topic, points in sorted(points_by_topic.items()):
                st.markdown(f"<div class='topic-heading'>{topic}</div>", unsafe_allow_html=True)
                for point in points:
                    text = point.get("text", "")
                    if text:
                        st.markdown(f"<div class='bullet-point'>{text}</div>", unsafe_allow_html=True)
                        
                        # Show evidence if available
                        if "_evidence" in point and point["_evidence"]:
                            with st.expander("View Evidence", expanded=False):
                                for evidence in point["_evidence"]:
                                    render_evidence_item(evidence)
        else:
            # Just render all points in order
            for point in key_points:
                if isinstance(point, dict):
                    text = point.get("text", "")
                    if text:
                        st.markdown(f"<div class='bullet-point'>{text}</div>", unsafe_allow_html=True)
                        
                        # Show evidence if available
                        if "_evidence" in point and point["_evidence"]:
                            with st.expander("View Evidence", expanded=False):
                                for evidence in point["_evidence"]:
                                    render_evidence_item(evidence)
                elif isinstance(point, str):
                    st.markdown(f"<div class='bullet-point'>{point}</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    elif not executive_summary:
        st.info("No key points available")

def display_summary_statistics(statistics: Dict[str, Any]) -> None:
    """Display clean, minimal statistics for summary results."""
    # Extract key statistics
    original_words = statistics.get("original_word_count", 0)
    summary_words = statistics.get("summary_word_count", 0)
    
    # Only display if we have meaningful stats
    if original_words > 0 or summary_words > 0:
        st.markdown("<hr />", unsafe_allow_html=True)
        st.markdown("<h3>Summary Statistics</h3>", unsafe_allow_html=True)
        
        # Create a clean metrics row
        cols = st.columns(3)
        
        with cols[0]:
            st.metric("Original Document", f"{original_words:,} words")
            
        with cols[1]:
            st.metric("Summary Length", f"{summary_words:,} words")
            
        with cols[2]:
            if original_words and summary_words:
                compression = (summary_words / original_words) * 100
                st.metric("Compression Ratio", f"{compression:.1f}%")
            else:
                st.metric("Compression Ratio", "N/A")

# =====================================================================
# Issues Rendering (Assess Assessment)
# =====================================================================

def render_issues_result(result: Dict[str, Any], context=None) -> None:
    """
    Render issues assessment results with enhanced context when available.
    
    Args:
        result: Issues assessment result dictionary
        context: Optional ProcessingContext object
    """
    # Get standardized data
    data = DataAccessor.get_issues_data(result, context)
    
    # Validate the data
    is_valid, error_message = DataAccessor.validate_data(data, "assess")
    if not is_valid:
        st.warning(f"Cannot render issues: {error_message}")
        if st.checkbox("Show raw data for debugging"):
            st.json(result)
        return
    
    # Apply issues styling
    apply_issues_styles()
    
    # Extract key data elements
    issues = data["issues"]
    executive_summary = data.get("executive_summary")
    statistics = data.get("statistics", {})
    
    # Define severity colors for consistent use
    severity_colors = {
        "critical": "#F44336",  # Red
        "high": "#FF9800",      # Orange
        "medium": "#2196F3",    # Blue
        "low": "#4CAF50"        # Green
    }
    
    # Display executive summary if available
    if executive_summary:
        st.markdown("## Executive Summary")
        st.markdown(executive_summary)
    
    # Display issues count and copy button in a single row
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"## Issues Found: {len(issues)}")
    with col2:
        # Add copy button
        if st.button("📋 Copy All Issues", type="secondary"):
            issues_text = create_text_for_copy(issues, "issues")
            st.code(issues_text, language="text")
            st.caption("👆 Copy the text above to your clipboard")
    
    # Group issues by severity
    issues_by_severity = {}
    for issue in issues:
        if not isinstance(issue, dict):
            continue
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
            
            # Draw a separator line
            st.markdown(f"<hr style='height:2px;border:none;color:#333;background-color:{severity_colors.get(severity, '#2196F3')};margin:0.5rem 0 1rem 0;'/>", unsafe_allow_html=True)
            
            # Render each issue with flat layout
            for issue in severity_issues:
                render_issue_card_flat(issue)
                
    # Display statistics
    display_issues_statistics(issues)

def render_issue_card_flat(issue: Dict[str, Any]) -> None:
    """
    Render a single issue card with a flat layout (no expanders).
    
    Args:
        issue: Issue dictionary with details and optional evidence
    """
    # Define severity colors for consistent use
    severity_colors = {
        "critical": "#F44336",  # Red
        "high": "#FF9800",      # Orange
        "medium": "#2196F3",    # Blue
        "low": "#4CAF50"        # Green
    }
    
    # Get issue properties
    title = issue.get("title", "Untitled Issue")
    description = issue.get("description", "")
    
    # Use evaluated_severity if available, otherwise use severity
    severity = issue.get("evaluated_severity", issue.get("severity", "medium")).lower()
    category = issue.get("category", "process")
    impact = issue.get("potential_impact", issue.get("impact", ""))
    
    # Get color for severity
    color = severity_colors.get(severity, "#2196F3")
    
    # Create a card-like container with styled HTML
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem; padding: 1rem; background-color: rgba(0, 0, 0, 0.05); 
             border-radius: 8px; border-left: 4px solid {color};">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
            <h4 style="margin: 0; font-size: 1.1rem;">{title}</h4>
            <div>
                <span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 4px; 
                     font-size: 0.7rem; font-weight: 600; margin-left: 8px;">{severity.upper()}</span>
                <span style="background-color: rgba(0,0,0,0.1); padding: 2px 8px; border-radius: 4px; 
                     font-size: 0.7rem; font-weight: 600; margin-left: 8px;">{category.upper()}</span>
            </div>
        </div>
        <div style="margin: 0.5rem 0;">
            <div style="font-size: 0.95rem;">{description}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add impact and recommendations in a collapsible section if available
    has_details = impact or issue.get("recommendations") or issue.get("suggested_recommendations") or issue.get("_evidence")
    
    if has_details:
        with st.expander("📌 View additional details", expanded=False):
            if impact:
                st.markdown("**Potential Impact:**")
                st.info(impact)
                
            # Recommendations if available
            recommendations = issue.get("recommendations", issue.get("suggested_recommendations", []))
            if recommendations:
                st.markdown("**Recommendations:**")
                for rec in recommendations:
                    st.markdown(f"- {rec}")
            
            # Show evidence if available
            evidence = issue.get("_evidence", [])
            if evidence:
                st.markdown("**Supporting Evidence:**")
                for evidence_item in evidence:
                    render_evidence_item(evidence_item, color)

def display_issues_statistics(issues: List[Dict[str, Any]]) -> None:
    """Display statistics for issue assessment based on issues list."""
    if not issues:
        return
        
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
        # Get categories distribution
        categories = {}
        for issue in issues:
            category = issue.get("category", "unknown").lower()
            categories[category] = categories.get(category, 0) + 1
        
        most_common_category = max(categories.items(), key=lambda x: x[1])[0] if categories else "None"
        st.metric("Most Common Category", most_common_category.title())
    
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

# =====================================================================
# Action Items Rendering (Extract Assessment)
# =====================================================================

def render_action_items_result(result: Dict[str, Any], context=None) -> None:
    """
    Render action items with enhanced formatting and evidence integration.
    
    Args:
        result: Action items result dictionary
        context: Optional ProcessingContext object for enhanced rendering
    """
    # Get standardized data
    data = DataAccessor.get_action_items_data(result, context)
    
    # Validate the data
    is_valid, error_message = DataAccessor.validate_data(data, "extract")
    if not is_valid:
        st.warning(f"Cannot render action items: {error_message}")
        if st.checkbox("Show raw data for debugging"):
            st.json(result)
        return
    
    # Apply action items styling
    apply_action_items_styles()
    
    # Extract key data elements
    action_items = data["action_items"]
    summary = data.get("summary")
    statistics = data.get("statistics", {})
    
    # Display summary if available
    if summary:
        st.markdown("## Summary")
        st.markdown(summary)
    
    # Display action items count and copy button in a single row
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"## Action Items Found: {len(action_items)}")
    with col2:
        # Add copy button
        if st.button("📋 Copy All Action Items", type="secondary"):
            actions_text = create_text_for_copy(action_items, "action_items")
            st.code(actions_text, language="text")
            st.caption("👆 Copy the text above to your clipboard")
    
    # Group action items by priority
    items_by_priority = {}
    for item in action_items:
        if not isinstance(item, dict):
            continue
        # Use evaluated_priority if available, otherwise use priority
        priority = item.get("evaluated_priority", item.get("priority", "medium")).lower()
        if priority not in items_by_priority:
            items_by_priority[priority] = []
        items_by_priority[priority].append(item)
    
    # Display action items grouped by priority
    for priority in ["high", "medium", "low"]:
        if priority in items_by_priority and items_by_priority[priority]:
            priority_items = items_by_priority[priority]
            st.markdown(f"### {priority.upper()} Priority Action Items ({len(priority_items)})")
            
            # Priority colors
            priority_colors = {
                "high": "#FF9800",    # Orange
                "medium": "#2196F3",  # Blue
                "low": "#4CAF50"      # Green
            }
            color = priority_colors.get(priority, "#2196F3")
            
            # Draw a separator line
            st.markdown(f"<hr style='height:2px;border:none;color:#333;background-color:{color};margin:0.5rem 0 1rem 0;'/>", unsafe_allow_html=True)
            
            # Render each action item
            for item in priority_items:
                render_action_item_card(item, color)
                
    # Display statistics if we have items
    if action_items:
        display_action_items_statistics(action_items, statistics)

def render_action_item_card(item: Dict[str, Any], color: str = "#2196F3") -> None:
    """
    Render a single action item card.
    
    Args:
        item: Action item dictionary
        color: Color for styling accents
    """
    # Get item properties
    description = item.get("description", "Untitled Action Item")
    owner = item.get("owner", "Unassigned")
    due_date = item.get("due_date", "No deadline")
    priority = item.get("evaluated_priority", item.get("priority", "medium")).title()
    context_info = item.get("context", "")
    
    # Create a card-like container with styled HTML
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem; padding: 1rem; background-color: rgba(0, 0, 0, 0.05); 
             border-radius: 8px; border-left: 4px solid {color};">
        <div style="margin-bottom: 0.7rem;">
            <div style="font-size: 1.1rem; font-weight: 500;">{description}</div>
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 0.5rem;">
            <div>
                <span style="font-weight: 500;">Owner:</span> {owner}
            </div>
            <div>
                <span style="font-weight: 500;">Due:</span> {due_date}
            </div>
            <div>
                <span style="font-weight: 500;">Priority:</span> {priority}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add context and evidence in a collapsible section if available
    has_details = context_info or item.get("_evidence")
    
    if has_details:
        with st.expander("📌 View additional details", expanded=False):
            if context_info:
                st.markdown("**Context:**")
                st.info(context_info)
            
            # Show evidence if available
            evidence = item.get("_evidence", [])
            if evidence:
                st.markdown("**Supporting Evidence:**")
                for evidence_item in evidence:
                    render_evidence_item(evidence_item, color)

def display_action_items_statistics(action_items: List[Dict[str, Any]], statistics: Dict[str, Any]) -> None:
    """Display statistics for action items assessment."""
    if not action_items:
        return
        
    # Get statistics directly if available
    by_owner = statistics.get("by_owner", {})
    by_priority = statistics.get("by_priority", {})
    
    # Calculate statistics if not available in stats dictionary
    if not by_owner:
        by_owner = {}
        for item in action_items:
            owner = item.get("owner", "Unassigned")
            by_owner[owner] = by_owner.get(owner, 0) + 1
    
    if not by_priority:
        by_priority = {"high": 0, "medium": 0, "low": 0}
        for item in action_items:
            priority = item.get("evaluated_priority", item.get("priority", "medium")).lower()
            if priority in by_priority:
                by_priority[priority] += 1
    
    # Display statistics
    st.markdown("### Action Items Overview")
    
    # Display metrics grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Action Items", len(action_items))
        
    with col2:
        high_count = by_priority.get("high", 0)
        st.metric("High Priority Items", high_count)
        
    with col3:
        # Show the owner with most items
        if by_owner:
            most_assigned = max(by_owner.items(), key=lambda x: x[1])[0]
            most_assigned_count = by_owner.get(most_assigned, 0)
            st.metric("Most Assigned", most_assigned, f"{most_assigned_count} items")
        else:
            st.metric("Most Assigned", "None", "0 items")
    
    # Display owner distribution if we have multiple owners
    if len(by_owner) > 1:
        st.markdown("#### Action Items by Owner")
        
        # Sort owners by item count
        sorted_owners = sorted(by_owner.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate max value for percentage
        max_value = max(by_owner.values()) if by_owner else 1
        
        # Create a horizontal bar chart for each owner
        for owner, count in sorted_owners:
            percentage = (count / max_value) * 100
            
            st.markdown(f"""
            <div style="margin-bottom: 8px;">
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <div style="flex-grow: 1; font-size: 0.9rem;">{owner}</div>
                    <div style="font-weight: 500; font-size: 0.9rem;">{count} item{'s' if count != 1 else ''}</div>
                </div>
                <div style="height: 8px; background-color: rgba(0,0,0,0.1); border-radius: 4px; overflow: hidden;">
                    <div style="width: {percentage}%; height: 100%; background-color: #2196F3;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# =====================================================================
# Analysis Rendering (Analyze Assessment)
# =====================================================================

def render_analysis_result(result: Dict[str, Any], context=None) -> None:
    """
    Render framework analysis results.
    
    Args:
        result: Analysis result dictionary
        context: Optional ProcessingContext object
    """
    # Get standardized data
    data = DataAccessor.get_analysis_data(result, context)
    
    # Validate the data
    is_valid, error_message = DataAccessor.validate_data(data, "analyze")
    if not is_valid:
        st.warning(f"Cannot render analysis: {error_message}")
        if st.checkbox("Show raw data for debugging"):
            st.json(result)
        return
    
    # Apply analysis styling
    apply_analysis_styles()
    
    # Extract key data elements
    dimension_assessments = data["dimension_assessments"]
    overall_rating = data.get("overall_rating")
    strategic_recommendations = data.get("strategic_recommendations", [])
    strengths = data.get("strengths", [])
    weaknesses = data.get("weaknesses", [])
    summary = data.get("summary")
    
    # Display summary if available
    if summary:
        st.markdown("## Executive Summary")
        st.markdown(summary)
    
    # Display overall rating if available
    if overall_rating:
        if isinstance(overall_rating, dict):
            rating_value = overall_rating.get("rating", "N/A")
            rating_description = overall_rating.get("description", "")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"""
                <div style="background-color: #2196F3; color: white; border-radius: 50%; width: 80px; height: 80px; 
                          display: flex; align-items: center; justify-content: center; font-size: 2rem; font-weight: bold;
                          margin: 0 auto;">
                    {rating_value}
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("## Overall Rating")
                st.markdown(rating_description)
        else:
            st.markdown("## Overall Rating")
            st.markdown(f"**{overall_rating}**")
    
    # Display dimension assessments
    st.markdown("## Dimension Assessments")
    
    # Use expanders for each dimension
    for dimension in dimension_assessments:
        if not isinstance(dimension, dict):
            continue
            
        dimension_name = dimension.get("dimension_name", "Unnamed Dimension")
        dimension_rating = dimension.get("dimension_rating", "")
        dimension_summary = dimension.get("dimension_summary", "")
        
        # Rating display
        rating_display = f" - Rating: {dimension_rating}" if dimension_rating else ""
        
        with st.expander(f"{dimension_name}{rating_display}", expanded=False):
            if dimension_summary:
                st.markdown(dimension_summary)
            
            # Display criteria assessments if available
            criteria = dimension.get("criteria_assessments", [])
            if criteria and isinstance(criteria, list):
                st.markdown("### Criteria Assessments")
                
                for criterion in criteria:
                    if not isinstance(criterion, dict):
                        continue
                        
                    criteria_name = criterion.get("criteria_name", "Unnamed Criterion")
                    criteria_rating = criterion.get("rating", "")
                    rationale = criterion.get("rationale", "")
                    
                    st.markdown(f"**{criteria_name}** - Rating: {criteria_rating}")
                    if rationale:
                        st.markdown(rationale)
            
            # Display strengths and weaknesses specific to this dimension
            dimension_strengths = dimension.get("strengths", [])
            if dimension_strengths and isinstance(dimension_strengths, list):
                st.markdown("**Strengths:**")
                for strength in dimension_strengths:
                    st.markdown(f"- {strength}")
            
            dimension_weaknesses = dimension.get("weaknesses", [])
            if dimension_weaknesses and isinstance(dimension_weaknesses, list):
                st.markdown("**Weaknesses:**")
                for weakness in dimension_weaknesses:
                    st.markdown(f"- {weakness}")
            
            # Display evidence if available
            evidence = dimension.get("_evidence", [])
            if evidence:
                st.markdown("**Supporting Evidence:**")
                for evidence_item in evidence:
                    render_evidence_item(evidence_item)
    
    # Display overall strengths and weaknesses
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("## Strengths")
        if strengths and isinstance(strengths, list):
            for strength in strengths:
                st.markdown(f"- {strength}")
        else:
            st.info("No overall strengths identified")
    
    with col2:
        st.markdown("## Weaknesses")
        if weaknesses and isinstance(weaknesses, list):
            for weakness in weaknesses:
                st.markdown(f"- {weakness}")
        else:
            st.info("No overall weaknesses identified")
    
    # Display strategic recommendations
    if strategic_recommendations:
        st.markdown("## Strategic Recommendations")
        
        for recommendation in strategic_recommendations:
            if not isinstance(recommendation, dict):
                continue
                
            rec_text = recommendation.get("recommendation", "")
            priority = recommendation.get("priority", "").upper()
            rationale = recommendation.get("rationale", "")
            
            # Priority colors
            priority_colors = {
                "HIGH": "#FF9800",    # Orange
                "MEDIUM": "#2196F3",  # Blue
                "LOW": "#4CAF50"      # Green
            }
            color = priority_colors.get(priority, "#2196F3")
            
            # Create styled recommendation card
            st.markdown(f"""
            <div style="margin-bottom: 1.5rem; padding: 1rem; background-color: rgba(0, 0, 0, 0.05); 
                     border-radius: 8px; border-left: 4px solid {color};">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
                    <h4 style="margin: 0; font-size: 1.1rem;">{rec_text}</h4>
                    <div>
                        <span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 4px; 
                             font-size: 0.7rem; font-weight: 600;">{priority}</span>
                    </div>
                </div>
                <div style="margin: 0.5rem 0;">
                    <div style="font-size: 0.95rem;">{rationale}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# =====================================================================
# Progress Rendering
# =====================================================================

def render_simple_progress(session_state):
    """
    Display a clean progress indicator with animated components.
    
    Args:
        session_state: Streamlit session state containing progress information
    """
    progress_value = float(session_state.get("current_progress", 0.0))
    progress_message = session_state.get("progress_message", "Processing...")
    current_stage = session_state.get("current_stage", "")
    
    # Add styling for progress elements
    st.markdown("""
    <style>
        /* Progress container */
        .progress-container {
            background-color: rgba(33, 150, 243, 0.05);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            border: 1px solid rgba(33, 150, 243, 0.2);
        }
        
        /* Stage name styling */
        .current-stage {
            font-weight: 600;
            margin: 0.75rem 0 0.25rem 0;
            color: #2196F3;
        }
        
        /* Message styling */
        .progress-message {
            margin-bottom: 0.75rem;
            opacity: 0.9;
        }
        
        /* Processing animation */
        @keyframes pulse {
            0% { opacity: 0.4; }
            50% { opacity: 1; }
            100% { opacity: 0.4; }
        }
        
        .processing-indicator {
            animation: pulse 1.5s infinite;
            color: #2196F3;
            display: flex;
            align-items: center;
            padding: 0.5rem;
            border-radius: 4px;
            margin-top: 0.7rem;
            background-color: rgba(33, 150, 243, 0.1);
            font-weight: 500;
            width: fit-content;
        }
        
        .processing-icon {
            margin-right: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Create progress container
    st.markdown("<div class='progress-container'>", unsafe_allow_html=True)
    
    # Create a clean progress bar
    st.progress(progress_value)
    
    # Show current stage with proper formatting
    if current_stage:
        stage_display = current_stage.replace('_', ' ').title()
        st.markdown(f"<div class='current-stage'>{stage_display}</div>", unsafe_allow_html=True)
    
    # Show progress message
    st.markdown(f"<div class='progress-message'>{progress_message}</div>", unsafe_allow_html=True)
    
    # Add animated processing indicator
    st.markdown(
        """<div class="processing-indicator">
            <span class="processing-icon">⚙️</span> Processing document...
        </div>""", 
        unsafe_allow_html=True
    )
    
    # Close container
    st.markdown("</div>", unsafe_allow_html=True)

def render_detailed_progress(session_state):
    """
    Display detailed progress information with terminal-like output.
    
    Args:
        session_state: Streamlit session state containing progress information
    """
    progress_value = float(session_state.get("current_progress", 0.0))
    
    # Create main progress bar
    st.progress(progress_value)
    
    # Show current stage and message
    current_stage = session_state.get("current_stage", "")
    progress_message = session_state.get("progress_message", "Processing...")
    
    if current_stage:
        st.markdown(f"**Current Stage:** {current_stage.replace('_', ' ').title()}")
    st.caption(progress_message)
    
    # Show detailed stage information
    stages_info = session_state.get("stages_info", {})
    if stages_info:
        with st.expander("View Processing Details", expanded=True):
            # Expected stage order
            stage_order = [
                "document_analysis", "chunking", "planning", 
                "extraction", "aggregation", "evaluation", 
                "formatting", "review"
            ]
            
            # Display stages in order
            for stage_name in stage_order:
                if stage_name in stages_info:
                    stage_info = stages_info[stage_name]
                    status = stage_info.get("status", "not_started")
                    progress = stage_info.get("progress", 0)
                    message = stage_info.get("message", "")
                    
                    # Determine status icon and CSS class
                    if status == "completed":
                        icon = "✅"
                        css_class = "completed-stage"
                    elif status == "running":
                        icon = "⏳"
                        css_class = "running-stage"
                    elif status == "failed":
                        icon = "❌"
                        css_class = "failed-stage"
                    else:
                        icon = "⏱️"
                        css_class = ""
                    
                    # Display stage status
                    display_name = stage_name.replace("_", " ").title()
                    progress_pct = f"{int(progress * 100)}%" if progress > 0 else ""
                    
                    st.markdown(f"<div class='stage-progress {css_class}'>", unsafe_allow_html=True)
                    st.markdown(f"{icon} **{display_name}** {progress_pct} `{status.upper()}`")
                    if message:
                        st.markdown(f"<div class='stage-message'>{message}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================================
# Shared Helper Functions
# =====================================================================

def render_evidence_item(evidence, color="#2196F3"):
    """
    Render a single evidence item with consistent styling.
    
    Args:
        evidence: Evidence dictionary with text and metadata
        color: Accent color for styling
    """
    chunk_index = evidence.get("chunk_index", "unknown")
    text = evidence.get("text", "No text available")
    
    st.markdown(f"""
    <div style="margin-bottom: 12px; padding: 8px 12px; 
             background-color: rgba(0, 0, 0, 0.05); 
             border-left: 3px solid {color}; border-radius: 4px;">
        <div style="font-size: 0.9rem; opacity: 0.7; margin-bottom: 4px;">
            Evidence (Chunk {chunk_index})
        </div>
        <div style="font-size: 0.95rem;">
            {text}
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_text_for_copy(items: List[Dict[str, Any]], item_type: str = "items") -> str:
    """
    Create formatted text content for copying to clipboard.
    Works for both issues and action items.
    
    Args:
        items: List of item dictionaries (issues or action items)
        item_type: Type of items ("issues" or "action_items")
        
    Returns:
        Formatted text string
    """
    # Determine item type based on content
    type_title = "ISSUES"
    if item_type == "action_items":
        type_title = "ACTION ITEMS"
    
    text_content = f"{type_title} ASSESSMENT REPORT\n"
    text_content += "=" * 50 + "\n\n"
    
    if item_type == "issues":
        # Group issues by severity
        items_by_severity = {}
        for item in items:
            # Use evaluated_severity if available, otherwise use severity
            severity = item.get("evaluated_severity", item.get("severity", "medium")).lower()
            if severity not in items_by_severity:
                items_by_severity[severity] = []
            items_by_severity[severity].append(item)
        
        # Add issues by severity
        for severity in ["critical", "high", "medium", "low"]:
            if severity in items_by_severity and items_by_severity[severity]:
                severity_items = items_by_severity[severity]
                text_content += f"{severity.upper()} SEVERITY ISSUES ({len(severity_items)})\n"
                text_content += "-" * 50 + "\n\n"
                
                for idx, item in enumerate(severity_items, 1):
                    title = item.get("title", "Untitled Issue")
                    description = item.get("description", "")
                    category = item.get("category", "process")
                    impact = item.get("potential_impact", item.get("impact", ""))
                    
                    text_content += f"{idx}. {title}\n"
                    text_content += f"   Category: {category.upper()}\n"
                    text_content += f"   Description: {description}\n"
                    
                    if impact:
                        text_content += f"   Potential Impact: {impact}\n"
                    
                    # Add recommendations if available
                    recommendations = item.get("recommendations", item.get("suggested_recommendations", []))
                    if recommendations:
                        text_content += "   Recommendations:\n"
                        for rec in recommendations:
                            text_content += f"   - {rec}\n"
                    
                    text_content += "\n"
    else:
        # Group action items by priority
        items_by_priority = {}
        for item in items:
            # Use evaluated_priority if available, otherwise use priority
            priority = item.get("evaluated_priority", item.get("priority", "medium")).lower()
            if priority not in items_by_priority:
                items_by_priority[priority] = []
            items_by_priority[priority].append(item)
        
        # Add action items by priority
        for priority in ["high", "medium", "low"]:
            if priority in items_by_priority and items_by_priority[priority]:
                priority_items = items_by_priority[priority]
                text_content += f"{priority.upper()} PRIORITY ACTION ITEMS ({len(priority_items)})\n"
                text_content += "-" * 50 + "\n\n"
                
                for idx, item in enumerate(priority_items, 1):
                    description = item.get("description", "Untitled Action Item")
                    owner = item.get("owner", "Unassigned")
                    due_date = item.get("due_date", "No deadline")
                    
                    text_content += f"{idx}. {description}\n"
                    text_content += f"   Owner: {owner}\n"
                    text_content += f"   Due Date: {due_date}\n"
                    text_content += "\n"
    
    return text_content

# =====================================================================
# Styling Functions
# =====================================================================

def apply_common_styles():
    """Apply common CSS styles for all renderers."""
    st.markdown("""
    <style>
        /* Base typography and spacing */
        h1, h2, h3, h4, h5, h6 {
            margin-top: 1.2em;
            margin-bottom: 0.8em;
        }
        
        /* Card styling for all assessment types */
        .bn-card {
            background-color: rgba(0, 0, 0, 0.05);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            transition: transform 0.15s ease;
        }
        
        .bn-card:hover {
            transform: translateY(-2px);
        }
        
        /* Evidence box styling */
        .evidence-box {
            background-color: rgba(0, 0, 0, 0.05);
            border-left: 3px solid #2196F3;
            border-radius: 4px;
            padding: 8px 12px;
            margin-bottom: 12px;
        }
        
        .evidence-source {
            font-size: 0.9rem;
            opacity: 0.7;
            margin-bottom: 4px;
        }
        
        .evidence-text {
            font-size: 0.95rem;
        }
        
        /* Copy button styling */
        .copy-button {
            background-color: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            font-size: 0.9rem;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        
        .copy-button:hover {
            background-color: #1976D2;
        }
        
        /* Badge styling for various types */
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 500;
            text-transform: uppercase;
        }
    </style>
    """, unsafe_allow_html=True)

def apply_summary_styles():
    """Apply specific CSS styles for summary rendering."""
    st.markdown("""
    <style>
        /* Base typography */
        .bn-summary {
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
        }
        
        /* Overview section */
        .bn-overview {
            font-size: 1.1rem;
            line-height: 1.5;
            background-color: rgba(33, 150, 243, 0.15);
            border-left: 4px solid #2196F3;
            padding: 1.25rem;
            margin-bottom: 1.5rem;
            border-radius: 0.25rem;
        }
        
        /* Content sections */
        .bn-section {
            margin-bottom: 1.5rem;
        }
        
        /* Summary content */
        .summary-content {
            line-height: 1.6;
            margin-top: 1rem;
        }
        
        /* Topic styling */
        .topic-heading {
            font-size: 1.2rem;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
            color: #2196F3;
        }
        
        /* Bullet points */
        .bullet-point {
            margin-bottom: 0.8rem;
            padding-left: 1.5rem;
            position: relative;
        }
        
        .bullet-point:before {
            content: "•";
            position: absolute;
            left: 0.5rem;
            color: #2196F3;
        }
    </style>
    """, unsafe_allow_html=True)

def apply_issues_styles():
    """Apply specific CSS styles for issues rendering."""
    st.markdown("""
    <style>
        /* Issue card styling */
        .issue-card {
            margin-bottom: 1.5rem;
            padding: 1rem;
            background-color: rgba(0, 0, 0, 0.05);
            border-radius: 8px;
            transition: transform 0.15s ease;
        }
        
        .issue-card:hover {
            transform: translateY(-2px);
        }
        
        /* Severity badges */
        .severity-badge {
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
            margin-left: 8px;
        }
        
        .severity-critical {
            background-color: #F44336;
        }
        
        .severity-high {
            background-color: #FF9800;
        }
        
        .severity-medium {
            background-color: #2196F3;
        }
        
        .severity-low {
            background-color: #4CAF50;
        }
        
        /* Category badges */
        .category-badge {
            background-color: rgba(0, 0, 0, 0.1);
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
            margin-left: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

def apply_action_items_styles():
    """Apply specific CSS styles for action items rendering."""
    st.markdown("""
    <style>
        /* Action item card styling */
        .action-item-card {
            margin-bottom: 1.5rem;
            padding: 1rem;
            background-color: rgba(0, 0, 0, 0.05);
            border-radius: 8px;
            transition: transform 0.15s ease;
        }
        
        .action-item-card:hover {
            transform: translateY(-2px);
        }
        
        /* Action item description */
        .action-item-description {
            font-size: 1.1rem;
            font-weight: 500;
            margin-bottom: 0.7rem;
        }
        
        /* Action item meta info */
        .action-item-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-bottom: 0.5rem;
        }
        
        .meta-label {
            font-weight: 500;
        }
        
        /* Priority badges */
        .priority-badge {
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
        }
        
        .priority-high {
            background-color: #FF9800;
        }
        
        .priority-medium {
            background-color: #2196F3;
        }
        
        .priority-low {
            background-color: #4CAF50;
        }
    </style>
    """, unsafe_allow_html=True)

def apply_analysis_styles():
    """Apply specific CSS styles for analysis rendering."""
    st.markdown("""
    <style>
        /* Rating display */
        .rating-circle {
            background-color: #2196F3;
            color: white;
            border-radius: 50%;
            width: 80px;
            height: 80px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            font-weight: bold;
            margin: 0 auto;
        }
        
        /* Dimension card */
        .dimension-card {
            margin-bottom: 1.5rem;
            padding: 1rem;
            background-color: rgba(0, 0, 0, 0.05);
            border-radius: 8px;
        }
        
        .dimension-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.8rem;
        }
        
        .dimension-name {
            font-size: 1.2rem;
            font-weight: 600;
        }
        
        .dimension-rating {
            font-weight: 600;
            color: #2196F3;
        }
        
        /* Criteria styling */
        .criteria-item {
            margin-bottom: 1rem;
            padding-left: 1rem;
            border-left: 2px solid rgba(33, 150, 243, 0.5);
        }
        
        .criteria-name {
            font-weight: 600;
        }
        
        .criteria-rating {
            font-weight: 500;
            color: #2196F3;
        }
        
        /* Recommendation card */
        .recommendation-card {
            margin-bottom: 1.5rem;
            padding: 1rem;
            background-color: rgba(0, 0, 0, 0.05);
            border-radius: 8px;
            border-left: 4px solid #2196F3;
        }
    </style>
    """, unsafe_allow_html=True)

def show_data_inspector(result, context=None):
    """
    Display an inspector that shows raw data structure.
    Only visible when debug mode is enabled.
    """
    if not st.session_state.get("debug_mode", False):
        return
        
    with st.expander("🔍 Data Inspector", expanded=False):
        st.write("### Result Data Structure")
        if isinstance(result, dict):
            st.json({k: type(v).__name__ for k, v in result.items()})
            
            # Show first level of nested dictionaries
            for key, value in result.items():
                if isinstance(value, dict):
                    st.write(f"**{key}** contains:")
                    st.json({k: type(v).__name__ for k, v in value.items()})
                elif isinstance(value, list) and value:
                    st.write(f"**{key}** is a list with {len(value)} items")
                    if value and isinstance(value[0], dict):
                        st.write("First item keys:")
                        st.json({k: type(v).__name__ for k, v in value[0].items()})
        
        # Show context data if available
        if context:
            st.write("### Context Structure")
            if hasattr(context, "data"):
                st.json({k: type(v).__name__ for k, v in context.data.items()})