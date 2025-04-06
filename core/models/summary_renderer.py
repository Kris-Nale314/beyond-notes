"""
Summary Rendering Module

Provides clean, professional rendering of document summaries with a focus on
readability and continuous content flow without hidden elements.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Shared CSS styles - Clean, minimal approach with proper typography
SUMMARY_CSS = """
<style>
    /* Base typography */
    .bn-summary {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, sans-serif;
        line-height: 1.6;
        color: rgba(255, 255, 255, 0.95);
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Document header */
    .bn-summary-header {
        margin-bottom: 2rem;
    }
    
    .bn-summary-title {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: white;
    }
    
    .bn-summary-meta {
        font-size: 0.9rem;
        opacity: 0.7;
        margin-bottom: 1rem;
    }
    
    .bn-meta-item {
        margin-right: 1.5rem;
    }
    
    /* Overview section */
    .bn-overview {
        font-size: 1.2rem;
        line-height: 1.5;
        background: rgba(33, 150, 243, 0.15);
        border-left: 4px solid #2196F3;
        padding: 1.25rem;
        margin-bottom: 2rem;
        border-radius: 0.25rem;
    }
    
    /* Headings */
    .bn-summary h2 {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        color: white;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .bn-summary h3 {
        font-size: 1.25rem;
        font-weight: 500;
        margin: 1.5rem 0 0.75rem 0;
        color: white;
    }
    
    .bn-summary h4 {
        font-size: 1.1rem;
        font-weight: 500;
        margin: 1.25rem 0 0.5rem 0;
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Main content sections */
    .bn-section {
        margin-bottom: 2rem;
    }
    
    /* Key points section */
    .bn-key-points {
        margin-bottom: 2rem;
    }
    
    .bn-points-list {
        margin: 0;
        padding: 0 0 0 1.5rem;
    }
    
    .bn-points-list li {
        margin-bottom: 0.75rem;
        padding-left: 0.5rem;
    }
    
    /* Topic sections */
    .bn-topic {
        margin-bottom: 1.5rem;
    }
    
    /* Decisions section */
    .bn-decisions {
        margin-bottom: 2rem;
    }
    
    .bn-decision {
        background: rgba(33, 150, 243, 0.1);
        border-left: 3px solid #2196F3;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.25rem;
    }
    
    /* Action items */
    .bn-action-items {
        margin-bottom: 2rem;
    }
    
    .bn-action-item {
        background: rgba(76, 175, 80, 0.1);
        border-left: 3px solid #4CAF50;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.25rem;
    }
    
    .bn-action-meta {
        font-size: 0.9rem;
        margin-top: 0.5rem;
        opacity: 0.8;
    }
    
    /* Quotes */
    .bn-quote {
        font-style: italic;
        border-left: 3px solid rgba(255, 255, 255, 0.3);
        padding-left: 1rem;
        margin: 1rem 0;
        color: rgba(255, 255, 255, 0.8);
    }
    
    /* Footer */
    .bn-summary-footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .bn-statistics {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        font-size: 0.9rem;
        opacity: 0.7;
    }
    
    .bn-stat-item {
        margin-right: 1.5rem;
    }
    
    /* Specific format styling */
    .bn-executive .bn-overview {
        font-size: 1.3rem;
    }
    
    .bn-narrative p {
        margin-bottom: 1.2rem;
    }
    
    .bn-bullet-point {
        display: flex;
        margin-bottom: 0.75rem;
    }
    
    .bn-bullet-marker {
        flex: 0 0 20px;
    }
    
    .bn-bullet-content {
        flex: 1;
    }
</style>
"""

def render_summary(result: Dict[str, Any], format_type: str = None) -> str:
    """
    Render a summary as clean HTML with a focus on readability.
    
    Args:
        result: The summary result dictionary
        format_type: The summary format type (executive, comprehensive, bullet_points, narrative)
        
    Returns:
        Formatted HTML string for display
    """
    # Get the format type from the result if not specified
    if not format_type:
        format_type = result.get("metadata", {}).get("user_options", {}).get("format", "executive")
    
    # Extract data from the result
    formatted_data = result.get("result", {})
    document_name = result.get("metadata", {}).get("document_name", "Document")
    document_type = result.get("metadata", {}).get("document_type", "")
    
    # Start building HTML
    html = f'<div class="bn-summary bn-{format_type}">'
    
    # Add document header
    html += '<header class="bn-summary-header">'
    html += f'<h1 class="bn-summary-title">{document_name}</h1>'
    html += '<div class="bn-summary-meta">'
    
    # Add metadata
    word_count = formatted_data.get("statistics", {}).get("original_word_count")
    if word_count:
        html += f'<span class="bn-meta-item">Word Count: {word_count:,}</span>'
    
    if document_type:
        html += f'<span class="bn-meta-item">Type: {document_type}</span>'
    
    # Add format type
    format_display = {
        "executive": "Executive Summary",
        "comprehensive": "Comprehensive Summary",
        "bullet_points": "Key Points Summary",
        "narrative": "Narrative Summary"
    }.get(format_type, "Summary")
    
    html += f'<span class="bn-meta-item">{format_display}</span>'
    html += '</div></header>'
    
    # Add overview section
    overview = formatted_data.get("overview", "")
    if overview:
        html += f'<section class="bn-overview">{overview}</section>'
    
    # Add key points section if available
    key_points = formatted_data.get("key_points", [])
    if key_points:
        html += '<section class="bn-key-points bn-section">'
        html += '<h2>Key Points</h2>'
        html += '<ul class="bn-points-list">'
        
        for point in key_points:
            if isinstance(point, dict):
                point_text = point.get("point", "")
            else:
                point_text = point
                
            html += f'<li>{point_text}</li>'
            
        html += '</ul></section>'
    
    # Render main content based on format type
    if format_type == "executive":
        html += _render_executive_content(formatted_data)
    elif format_type == "comprehensive":
        html += _render_comprehensive_content(formatted_data)
    elif format_type == "bullet_points":
        html += _render_bullet_points_content(formatted_data)
    elif format_type == "narrative":
        html += _render_narrative_content(formatted_data)
    else:
        # Default to executive if unknown format
        html += _render_executive_content(formatted_data)
    
    # Add footer with statistics
    html += _render_statistics_footer(formatted_data)
    
    html += '</div>'  # Close summary div
    
    # Combine with CSS and return
    return SUMMARY_CSS + html

def _render_executive_content(data: Dict[str, Any]) -> str:
    """Render the main content for an executive summary."""
    html = ""
    
    # Add summary content
    summary_text = data.get("summary", "")
    if summary_text:
        html += '<section class="bn-section">'
        
        # Format paragraphs properly
        paragraphs = summary_text.split('\n\n')
        for paragraph in paragraphs:
            if paragraph.strip():
                html += f'<p>{paragraph}</p>'
                
        html += '</section>'
    
    # Add decisions if available
    decisions = data.get("decisions", [])
    if decisions:
        html += '<section class="bn-decisions bn-section">'
        html += '<h2>Key Decisions</h2>'
        
        for decision in decisions:
            if isinstance(decision, dict):
                decision_text = decision.get("decision", "")
                context = decision.get("context", "")
                implications = decision.get("implications", "")
                
                html += '<div class="bn-decision">'
                html += f'<h4>{decision_text}</h4>'
                
                if context:
                    html += f'<p>{context}</p>'
                    
                if implications:
                    html += f'<p><strong>Implications:</strong> {implications}</p>'
                    
                html += '</div>'
            elif isinstance(decision, str):
                html += f'<div class="bn-decision"><p>{decision}</p></div>'
        
        html += '</section>'
    
    # Add action items if available
    action_items = data.get("action_items", [])
    if action_items:
        html += '<section class="bn-action-items bn-section">'
        html += '<h2>Action Items</h2>'
        
        for item in action_items:
            if isinstance(item, dict):
                action = item.get("action", "")
                owner = item.get("owner", "")
                timeline = item.get("timeline", "")
                
                html += '<div class="bn-action-item">'
                html += f'<p>{action}</p>'
                
                if owner or timeline:
                    html += '<div class="bn-action-meta">'
                    if owner:
                        html += f'<span>Owner: {owner}</span>'
                    if timeline:
                        html += f'{" • " if owner else ""}<span>Timeline: {timeline}</span>'
                    html += '</div>'
                
                html += '</div>'
            elif isinstance(item, str):
                html += f'<div class="bn-action-item"><p>{item}</p></div>'
        
        html += '</section>'
    
    # Add conclusions if available
    conclusions = data.get("conclusions", "")
    if conclusions:
        html += '<section class="bn-section">'
        html += '<h2>Conclusions</h2>'
        html += f'<p>{conclusions}</p>'
        html += '</section>'
    
    return html

def _render_comprehensive_content(data: Dict[str, Any]) -> str:
    """Render the main content for a comprehensive summary."""
    html = ""
    
    # Add summary text if available
    summary_text = data.get("summary", "")
    if summary_text:
        html += '<section class="bn-section">'
        
        # Format paragraphs properly
        paragraphs = summary_text.split('\n\n')
        for paragraph in paragraphs:
            if paragraph.strip():
                html += f'<p>{paragraph}</p>'
                
        html += '</section>'
    
    # Add topics with detailed information
    topics = data.get("topics", [])
    if topics:
        html += '<section class="bn-section">'
        html += '<h2>Key Topics</h2>'
        
        for topic in topics:
            topic_name = topic.get("topic", "Topic")
            
            html += '<div class="bn-topic">'
            html += f'<h3>{topic_name}</h3>'
            
            # Add key points
            key_points = topic.get("key_points", [])
            if key_points:
                html += '<ul class="bn-points-list">'
                for point in key_points:
                    html += f'<li>{point}</li>'
                html += '</ul>'
            
            # Add details if available
            details = topic.get("details", "")
            if details:
                # Format paragraphs
                details_paragraphs = details.split('\n\n')
                for paragraph in details_paragraphs:
                    if paragraph.strip():
                        html += f'<p>{paragraph}</p>'
            
            html += '</div>'  # Close topic div
        
        html += '</section>'
    
    # Add decisions, action items and conclusions
    _add_decisions_and_actions(html, data)
    
    return html

def _render_bullet_points_content(data: Dict[str, Any]) -> str:
    """Render the main content for a bullet point summary."""
    html = ""
    
    # Add topics with bullet points
    topics = data.get("topics", [])
    if topics:
        html += '<section class="bn-section">'
        html += '<h2>Topics</h2>'
        
        for topic in topics:
            topic_name = topic.get("topic", "Topic")
            
            html += '<div class="bn-topic">'
            html += f'<h3>{topic_name}</h3>'
            
            # Add key points as bullets
            key_points = topic.get("key_points", [])
            if key_points:
                for point in key_points:
                    html += '<div class="bn-bullet-point">'
                    html += '<div class="bn-bullet-marker">•</div>'
                    html += f'<div class="bn-bullet-content">{point}</div>'
                    html += '</div>'
            
            html += '</div>'  # Close topic div
        
        html += '</section>'
    
    # If no topics but we have key points, show them directly
    elif data.get("key_points", []):
        key_points = data.get("key_points", [])
        html += '<section class="bn-section">'
        html += '<h2>Key Points</h2>'
        
        for point in key_points:
            if isinstance(point, dict):
                point_text = point.get("point", "")
            else:
                point_text = point
                
            html += '<div class="bn-bullet-point">'
            html += '<div class="bn-bullet-marker">•</div>'
            html += f'<div class="bn-bullet-content">{point_text}</div>'
            html += '</div>'
            
        html += '</section>'
    
    # Add decisions and action items
    decisions = data.get("decisions", [])
    if decisions:
        html += '<section class="bn-section">'
        html += '<h2>Decisions</h2>'
        
        for decision in decisions:
            if isinstance(decision, dict):
                decision_text = decision.get("decision", "")
                
                html += '<div class="bn-bullet-point">'
                html += '<div class="bn-bullet-marker">•</div>'
                html += f'<div class="bn-bullet-content">{decision_text}</div>'
                html += '</div>'
            else:
                html += '<div class="bn-bullet-point">'
                html += '<div class="bn-bullet-marker">•</div>'
                html += f'<div class="bn-bullet-content">{decision}</div>'
                html += '</div>'
                
        html += '</section>'
    
    # Add action items
    action_items = data.get("action_items", [])
    if action_items:
        html += '<section class="bn-section">'
        html += '<h2>Action Items</h2>'
        
        for item in action_items:
            if isinstance(item, dict):
                action = item.get("action", "")
                owner = item.get("owner", "")
                
                html += '<div class="bn-bullet-point">'
                html += '<div class="bn-bullet-marker">•</div>'
                html += f'<div class="bn-bullet-content">{action}'
                
                if owner:
                    html += f' <em>({owner})</em>'
                    
                html += '</div></div>'
            else:
                html += '<div class="bn-bullet-point">'
                html += '<div class="bn-bullet-marker">•</div>'
                html += f'<div class="bn-bullet-content">{item}</div>'
                html += '</div>'
                
        html += '</section>'
    
    return html

def _render_narrative_content(data: Dict[str, Any]) -> str:
    """Render the main content for a narrative summary."""
    html = ""
    
    # Add introduction if available
    intro = data.get("introduction", "")
    if intro:
        html += '<section class="bn-section">'
        html += '<h2>Introduction</h2>'
        html += '<div class="bn-narrative">'
        
        # Format paragraphs
        paragraphs = intro.split('\n\n')
        for paragraph in paragraphs:
            if paragraph.strip():
                html += f'<p>{paragraph}</p>'
                
        html += '</div></section>'
    
    # Add main content section
    main_content = data.get("main_content", data.get("summary", ""))
    if main_content:
        html += '<section class="bn-section">'
        if not intro:  # Only add a heading if we didn't already have an intro
            html += '<h2>Main Content</h2>'
        
        html += '<div class="bn-narrative">'
        
        # Format paragraphs properly
        paragraphs = main_content.split('\n\n')
        for paragraph in paragraphs:
            if paragraph.strip():
                html += f'<p>{paragraph}</p>'
        
        html += '</div></section>'
    
    # Add conclusion if available
    conclusion = data.get("conclusion", "")
    if conclusion:
        html += '<section class="bn-section">'
        html += '<h2>Conclusion</h2>'
        html += '<div class="bn-narrative">'
        
        # Format paragraphs
        paragraphs = conclusion.split('\n\n')
        for paragraph in paragraphs:
            if paragraph.strip():
                html += f'<p>{paragraph}</p>'
        
        html += '</div></section>'
    
    # Add quotes if available
    quotes = data.get("key_quotes", [])
    if quotes:
        html += '<section class="bn-section">'
        html += '<h2>Key Quotes</h2>'
        
        for quote in quotes:
            if isinstance(quote, dict):
                text = quote.get("text", "")
                source = quote.get("source", "")
                
                html += '<div class="bn-quote">'
                html += f'"{text}"'
                
                if source:
                    html += f' — {source}'
                    
                html += '</div>'
            elif isinstance(quote, str):
                html += f'<div class="bn-quote">"{quote}"</div>'
        
        html += '</section>'
    
    return html

def _add_decisions_and_actions(html: str, data: Dict[str, Any]) -> None:
    """Add decisions and action items sections to the HTML."""
    # Add decisions if available
    decisions = data.get("decisions", [])
    if decisions:
        html += '<section class="bn-decisions bn-section">'
        html += '<h2>Key Decisions</h2>'
        
        for decision in decisions:
            if isinstance(decision, dict):
                decision_text = decision.get("decision", "")
                context = decision.get("context", "")
                implications = decision.get("implications", "")
                
                html += '<div class="bn-decision">'
                html += f'<h4>{decision_text}</h4>'
                
                if context:
                    html += f'<p>{context}</p>'
                    
                if implications:
                    html += f'<p><strong>Implications:</strong> {implications}</p>'
                    
                html += '</div>'
            elif isinstance(decision, str):
                html += f'<div class="bn-decision"><p>{decision}</p></div>'
        
        html += '</section>'
    
    # Add action items if available
    action_items = data.get("action_items", [])
    if action_items:
        html += '<section class="bn-action-items bn-section">'
        html += '<h2>Action Items</h2>'
        
        for item in action_items:
            if isinstance(item, dict):
                action = item.get("action", "")
                owner = item.get("owner", "")
                timeline = item.get("timeline", "")
                
                html += '<div class="bn-action-item">'
                html += f'<p>{action}</p>'
                
                if owner or timeline:
                    html += '<div class="bn-action-meta">'
                    if owner:
                        html += f'<span>Owner: {owner}</span>'
                    if timeline:
                        html += f'{" • " if owner else ""}<span>Timeline: {timeline}</span>'
                    html += '</div>'
                
                html += '</div>'
            elif isinstance(item, str):
                html += f'<div class="bn-action-item"><p>{item}</p></div>'
        
        html += '</section>'

def _render_statistics_footer(data: Dict[str, Any]) -> str:
    """Render the statistics footer."""
    html = '<footer class="bn-summary-footer">'
    
    stats = data.get("statistics", {})
    if stats:
        html += '<div class="bn-statistics">'
        
        # Original word count
        if "original_word_count" in stats:
            html += f'<span class="bn-stat-item">Original: {stats["original_word_count"]:,} words</span>'
        
        # Summary word count
        if "summary_word_count" in stats:
            html += f'<span class="bn-stat-item">Summary: {stats["summary_word_count"]:,} words</span>'
        
        # Compression ratio
        if "original_word_count" in stats and "summary_word_count" in stats:
            original = stats["original_word_count"]
            summary = stats["summary_word_count"]
            if original > 0:
                ratio = round((summary / original) * 100, 1)
                html += f'<span class="bn-stat-item">Compression: {ratio}%</span>'
        
        # Topics count
        if "topics_covered" in stats:
            html += f'<span class="bn-stat-item">Topics: {stats["topics_covered"]}</span>'
            
        html += '</div>'
    
    html += '</footer>'
    return html