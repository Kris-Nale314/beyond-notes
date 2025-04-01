"""
Formatting Utilities

Handles formatting assessment results into various output formats (markdown, etc.)
and provides helper functions for displaying progress and saving outputs.
"""

import os
import json
import logging
import copy
from pathlib import Path
from datetime import datetime

from utils.paths import AppPaths
from utils.result_accessor import get_assessment_data, get_item_count

# Configure logging
logger = logging.getLogger(__name__)

def format_assessment_report(result, assessment_type):
    """
    Format the assessment result into a markdown report using the accessor utility.
    
    Args:
        result: The raw assessment result dictionary
        assessment_type: Type of assessment (distill, extract, assess, analyze)
        
    Returns:
        Formatted markdown report as a string
    """
    # Get normalized data view using the accessor
    data = get_assessment_data(result, assessment_type)
    
    # Format the report based on available data
    report_text = f"# {assessment_type.title()} Assessment\n\n"
    
    # Add executive summary or overview if available
    if "executive_summary" in data:
        report_text += f"## Executive Summary\n\n{data['executive_summary']}\n\n"
    elif "overview" in data:
        report_text += f"## Overview\n\n{data['overview']}\n\n"
    elif "summary" in data:
        report_text += f"## Summary\n\n{data['summary']}\n\n"
    
    # Add specific sections based on assessment type
    if assessment_type == "distill":
        # Add summary content if not already included above
        if "summary" in data and "Summary" not in report_text:
            report_text += f"{data['summary']}\n\n"
        
        # Add topics if available
        if "topics" in data and isinstance(data["topics"], list):
            report_text += "## Topics\n\n"
            for topic in data["topics"]:
                report_text += f"### {topic.get('topic', 'Topic')}\n\n"
                
                # Handle key points
                if "key_points" in topic and isinstance(topic["key_points"], list):
                    for point in topic["key_points"]:
                        report_text += f"- {point}\n"
                    report_text += "\n"
                
                # Add details if available
                if "details" in topic and topic["details"]:
                    report_text += f"{topic['details']}\n\n"
    
    elif assessment_type == "extract":
        # Add action items
        if "action_items" in data and isinstance(data["action_items"], list):
            # Group by owner if available
            owners = {}
            for item in data["action_items"]:
                owner = item.get("owner", "Unassigned")
                if owner not in owners:
                    owners[owner] = []
                owners[owner].append(item)
            
            # Display by owner
            report_text += "## Action Items by Owner\n\n"
            for owner, items in owners.items():
                report_text += f"### {owner}\n\n"
                
                for item in items:
                    # Get priority from regular or evaluated field
                    priority = item.get("evaluated_priority", item.get("priority", "medium"))
                    due_date = f"Due: {item.get('due_date', 'Unspecified')}" if item.get("due_date") else ""
                    
                    report_text += f"**{item.get('description', 'Action needed')}**\n\n"
                    report_text += f"Priority: {priority.upper()}  |  {due_date}\n\n"
                    
                    if item.get("context"):
                        report_text += f"{item['context']}\n\n"
    
    elif assessment_type == "assess":
        # Add issues
        if "issues" in data and isinstance(data["issues"], list):
            # Group by severity
            issues_by_severity = {}
            for issue in data["issues"]:
                # Use evaluated_severity if available, fallback to severity
                severity = issue.get("evaluated_severity", issue.get("severity", "medium"))
                if severity not in issues_by_severity:
                    issues_by_severity[severity] = []
                issues_by_severity[severity].append(issue)
            
            # Display issues by severity (in order from most to least severe)
            for severity in ["critical", "high", "medium", "low"]:
                if severity in issues_by_severity and issues_by_severity[severity]:
                    report_text += f"## {severity.title()} Severity Issues\n\n"
                    
                    for issue in issues_by_severity[severity]:
                        title = issue.get("title", "Untitled Issue")
                        # Use evaluated_category if available
                        category = issue.get("evaluated_category", issue.get("category", ""))
                        description = issue.get("description", "")
                        impact = issue.get("impact", issue.get("potential_impact", ""))
                        
                        report_text += f"### {title}\n\n"
                        report_text += f"**Category:** {category}\n\n"
                        report_text += f"{description}\n\n"
                        
                        if impact:
                            report_text += f"**Impact:** {impact}\n\n"
                        
                        # Try to find recommendations in different places
                        recommendations = []
                        if "recommendations" in issue and issue["recommendations"]:
                            recommendations = issue["recommendations"]
                        elif "suggested_recommendations" in issue and issue["suggested_recommendations"]:
                            recommendations = issue["suggested_recommendations"]
                            
                        if recommendations:
                            report_text += "**Recommendations:**\n\n"
                            for rec in recommendations:
                                report_text += f"- {rec}\n"
                            report_text += "\n"
    
    elif assessment_type == "analyze":
        # Add dimension assessments
        if "dimension_assessments" in data:
            dimensions = data["dimension_assessments"]
            
            # Handle either dictionary or list format
            if isinstance(dimensions, dict):
                for dimension_name, dimension in dimensions.items():
                    rating = dimension.get("dimension_rating", "")
                    rating_text = f" (Rating: {rating})" if rating else ""
                    
                    report_text += f"## {dimension.get('dimension_name', dimension_name)}{rating_text}\n\n"
                    report_text += f"{dimension.get('dimension_summary', '')}\n\n"
                    
                    # Add criteria assessments
                    if "criteria_assessments" in dimension and isinstance(dimension["criteria_assessments"], list):
                        for criteria in dimension["criteria_assessments"]:
                            criteria_rating = criteria.get("rating", "")
                            criteria_rating_text = f" (Rating: {criteria_rating})" if criteria_rating else ""
                            
                            report_text += f"### {criteria.get('criteria_name', 'Criteria')}{criteria_rating_text}\n\n"
                            report_text += f"{criteria.get('rationale', '')}\n\n"
                    
                    # Add strengths and weaknesses
                    if "strengths" in dimension and dimension["strengths"]:
                        report_text += "**Strengths:**\n\n"
                        for strength in dimension["strengths"]:
                            report_text += f"- {strength}\n"
                        report_text += "\n"
                    
                    if "weaknesses" in dimension and dimension["weaknesses"]:
                        report_text += "**Weaknesses:**\n\n"
                        for weakness in dimension["weaknesses"]:
                            report_text += f"- {weakness}\n"
                        report_text += "\n"
            elif isinstance(dimensions, list):
                # Handle list format if used
                for dimension in dimensions:
                    rating = dimension.get("dimension_rating", "")
                    rating_text = f" (Rating: {rating})" if rating else ""
                    
                    report_text += f"## {dimension.get('dimension_name', 'Dimension')}{rating_text}\n\n"
                    report_text += f"{dimension.get('dimension_summary', '')}\n\n"
                    
                    # Similar processing for criteria, strengths, weaknesses as above
                    # ...
            
            # Add strategic recommendations
            recommendations = data.get("strategic_recommendations", [])
            if recommendations and isinstance(recommendations, list):
                report_text += "## Strategic Recommendations\n\n"
                
                for rec in recommendations:
                    priority = rec.get("priority", "")
                    priority_text = f" (Priority: {priority.upper()})" if priority else ""
                    
                    report_text += f"### {rec.get('recommendation', 'Recommendation')}{priority_text}\n\n"
                    report_text += f"{rec.get('rationale', '')}\n\n"
    
    # Add statistics if available
    stats = data.get("statistics", {})
    if stats:
        report_text += "## Processing Statistics\n\n"
        
        # Add processing time if available
        processing_time = data.get("metadata", {}).get("processing_time") or stats.get("processing_time_seconds")
        if processing_time:
            report_text += f"- Processing Time: {processing_time:.2f} seconds\n"
        
        # Add item count
        items_count = get_item_count(result, assessment_type)
        if items_count > 0:
            item_type = "topics" if assessment_type == "distill" else \
                       "action items" if assessment_type == "extract" else \
                       "issues" if assessment_type == "assess" else \
                       "dimensions" if assessment_type == "analyze" else "items"
            report_text += f"- Total {item_type.capitalize()}: {items_count}\n"
        
        # Add other statistics available in the data
        if assessment_type == "distill":
            if "original_word_count" in stats:
                report_text += f"- Original Word Count: {stats['original_word_count']}\n"
            if "summary_word_count" in stats:
                report_text += f"- Summary Word Count: {stats['summary_word_count']}\n"
            if "compression_ratio" in stats:
                report_text += f"- Compression Ratio: {stats['compression_ratio']:.2f}\n"
                
        elif assessment_type == "extract":
            # Show counts by owner if available
            owner_counts = stats.get("by_owner", {})
            if owner_counts:
                report_text += "- Items by Owner:\n"
                for owner, count in owner_counts.items():
                    report_text += f"  - {owner}: {count}\n"
                    
        elif assessment_type == "assess":
            # Show counts by severity if available
            severity_counts = stats.get("by_severity", {})
            if severity_counts:
                report_text += "- Issues by Severity:\n"
                for severity, count in severity_counts.items():
                    report_text += f"  - {severity.capitalize()}: {count}\n"
                    
        elif assessment_type == "analyze":
            # Add analyze-specific statistics
            pass
            
    return report_text

def display_pipeline_progress(stages):
    """
    Display the processing pipeline stages in a simple format.
    
    Args:
        stages: Dictionary of pipeline stages and their status
        
    Returns:
        Formatted markdown text showing pipeline status
    """
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
            # Add error message if available
            if stage.get("error"):
                status_text += f" - {stage.get('error')}"
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

def save_result_to_output(result, assessment_type):
    """
    Save the result to the output folder with proper organization.
    
    Args:
        result: Assessment result dictionary
        assessment_type: Type of assessment (distill, extract, assess, analyze)
        
    Returns:
        Dictionary with paths to saved files
    """
    output_dir = AppPaths.get_assessment_output_dir(assessment_type)
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a clean copy of the result without sensitive data
    clean_result = copy.deepcopy(result)
    if "metadata" in clean_result and "options" in clean_result["metadata"]:
        options = clean_result["metadata"]["options"]
        if "api_key" in options:
            options.pop("api_key")
    
    # Save JSON result
    output_file = output_dir / f"{assessment_type}_result_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(clean_result, f, indent=2, ensure_ascii=False)
    
    # Generate and save markdown report
    md_content = format_assessment_report(clean_result, assessment_type)
    md_file = output_dir / f"{assessment_type}_report_{timestamp}.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    logger.info(f"Saved assessment result to {output_file} and report to {md_file}")
    
    # Return paths
    return {
        "json_path": str(output_file),
        "md_path": str(md_file)
    }