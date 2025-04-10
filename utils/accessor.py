# utils/accessor.py
"""
Data Accessor Module

Provides standardized access to data from assessment results and context objects.
This module serves as a unified data layer for renderers and other components
that need to extract information from Beyond Notes assessments.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import copy

# Configure logger
logger = logging.getLogger("beyond-notes.accessor")

class DataAccessor:
    """
    Unified data access layer for Beyond Notes assessments.
    
    Provides standardized methods to extract data from both raw result dictionaries
    and ProcessingContext objects. Handles all the complex lookup paths and fallbacks
    to ensure consistent data structure regardless of the source.
    """

    @staticmethod
    def get_summary_data(result: Dict[str, Any], context=None) -> Dict[str, Any]:
        """
        Extract standardized summary data from result or context.
        
        Args:
            result: Raw assessment result dictionary
            context: Optional ProcessingContext object
            
        Returns:
            Dictionary with standardized keys for summary data
        """
        # Initialize the standardized data structure
        summary_data = {
            "summary_content": None,             # Main summary text
            "key_points": [],                    # List of key points
            "topics": [],                        # List of identified topics
            "format_type": "executive",          # Summary format
            "executive_summary": None,           # Executive summary (if separate)
            "decisions": [],                     # Key decisions (if extracted)
            "action_items": [],                  # Action items (if extracted)
            "metadata": {},                      # Metadata about the document
            "statistics": {},                    # Processing statistics
            "evidence": {}                       # Evidence links for key points
        }
        
        try:
            # Try to get data from context first (preferred source)
            if context:
                logger.debug("Attempting to extract summary data from context")
                
                # Try to get formatted result from context
                if hasattr(context, "data") and "formatted" in context.data:
                    formatted_data = context.data["formatted"]
                    logger.debug("Found formatted data in context")
                    
                    # Extract summary content
                    if isinstance(formatted_data, dict):
                        if "summary" in formatted_data:
                            summary_data["summary_content"] = formatted_data["summary"]
                            logger.debug("Extracted summary content from formatted data")
                            
                    # Extract metadata
                    if isinstance(formatted_data, dict) and "metadata" in formatted_data:
                        summary_data["metadata"] = formatted_data["metadata"]
                        
                        # Get format type from user options
                        user_options = formatted_data["metadata"].get("user_options", {})
                        if user_options and isinstance(user_options, dict) and "format" in user_options:
                            summary_data["format_type"] = user_options["format"]
                            logger.debug(f"Format type from context: {summary_data['format_type']}")
                    
                    # Extract statistics
                    if isinstance(formatted_data, dict) and "statistics" in formatted_data:
                        summary_data["statistics"] = formatted_data["statistics"]
                
                # Try to get key points from aggregated or extracted data
                if hasattr(context, "data"):
                    # Check aggregated data first (preferred)
                    if "aggregated" in context.data and isinstance(context.data["aggregated"], dict):
                        if "key_points" in context.data["aggregated"]:
                            summary_data["key_points"] = context.data["aggregated"]["key_points"]
                            logger.debug(f"Extracted {len(summary_data['key_points'])} key points from aggregated data")
                        
                        if "topics" in context.data["aggregated"]:
                            summary_data["topics"] = context.data["aggregated"]["topics"]
                            logger.debug(f"Extracted {len(summary_data['topics'])} topics from aggregated data")
                    
                    # Check extracted data as fallback
                    elif "extracted" in context.data and isinstance(context.data["extracted"], dict):
                        if "key_points" in context.data["extracted"]:
                            summary_data["key_points"] = context.data["extracted"]["key_points"]
                            logger.debug(f"Extracted {len(summary_data['key_points'])} key points from extracted data")
                    
                    # Check evaluated data for overall assessment
                    if "evaluated" in context.data and isinstance(context.data["evaluated"], dict):
                        if "overall_assessment" in context.data["evaluated"]:
                            overall = context.data["evaluated"]["overall_assessment"]
                            if isinstance(overall, dict) and "executive_summary" in overall:
                                summary_data["executive_summary"] = overall["executive_summary"]
                                logger.debug("Extracted executive summary from evaluated data")
                
                # Try to get evidence for key points
                if hasattr(context, "evidence_store") and summary_data["key_points"]:
                    # For each key point that has an ID, get its evidence
                    for point in summary_data["key_points"]:
                        if isinstance(point, dict) and "id" in point:
                            point_id = point["id"]
                            # Try to get evidence using context methods if available
                            if hasattr(context, "get_evidence_for_item"):
                                evidence = context.get_evidence_for_item(point_id)
                                if evidence:
                                    summary_data["evidence"][point_id] = evidence
                                    logger.debug(f"Found {len(evidence)} evidence items for key point {point_id}")
            
            # If we don't have what we need from context, try the raw result
            logger.debug("Attempting to extract missing data from raw result")
            
            # Extract summary content if not already found
            if not summary_data["summary_content"]:
                summary_content = DataAccessor._get_from_paths(result, [
                    "summary",
                    "result.summary",
                    "formatted.summary"
                ])
                if summary_content:
                    summary_data["summary_content"] = summary_content
                    logger.debug("Extracted summary content from raw result")
            
            # Extract key points if not already found
            if not summary_data["key_points"]:
                key_points = DataAccessor._get_from_paths(result, [
                    "key_points",
                    "result.key_points",
                    "extracted_info.key_points"
                ])
                if key_points and isinstance(key_points, list):
                    summary_data["key_points"] = key_points
                    logger.debug(f"Extracted {len(key_points)} key points from raw result")
            
            # Extract topics if not already found
            if not summary_data["topics"]:
                topics = DataAccessor._get_from_paths(result, [
                    "topics",
                    "result.topics"
                ])
                if topics and isinstance(topics, list):
                    summary_data["topics"] = topics
                    logger.debug(f"Extracted {len(topics)} topics from raw result")
                
            # Extract executive summary if not already found
            if not summary_data["executive_summary"]:
                exec_summary = DataAccessor._get_from_paths(result, [
                    "executive_summary",
                    "result.executive_summary",
                    "overall_assessment.executive_summary",
                    "result.overall_assessment.executive_summary"
                ])
                if exec_summary:
                    summary_data["executive_summary"] = exec_summary
                    logger.debug("Extracted executive summary from raw result")
            
            # Extract metadata if not already found
            if not summary_data["metadata"]:
                metadata = DataAccessor._get_from_paths(result, [
                    "metadata",
                    "result.metadata"
                ])
                if metadata and isinstance(metadata, dict):
                    summary_data["metadata"] = metadata
                    # Extract format type from user options if available
                    user_options = metadata.get("user_options", {})
                    if user_options and isinstance(user_options, dict) and "format" in user_options:
                        summary_data["format_type"] = user_options["format"]
                        logger.debug(f"Format type from result metadata: {summary_data['format_type']}")
            
            # Extract statistics if not already found
            if not summary_data["statistics"]:
                statistics = DataAccessor._get_from_paths(result, [
                    "statistics",
                    "result.statistics"
                ])
                if statistics and isinstance(statistics, dict):
                    summary_data["statistics"] = statistics
            
            # Ensure key_points is a list (handle null or non-list values)
            if not isinstance(summary_data["key_points"], list):
                summary_data["key_points"] = []
            
            # Ensure topics is a list (handle null or non-list values)
            if not isinstance(summary_data["topics"], list):
                summary_data["topics"] = []
            
            # Infer some statistics if needed
            if "original_word_count" not in summary_data["statistics"] and "word_count" in summary_data["metadata"]:
                summary_data["statistics"]["original_word_count"] = summary_data["metadata"]["word_count"]
            
            if "summary_word_count" not in summary_data["statistics"] and summary_data["summary_content"]:
                summary_data["statistics"]["summary_word_count"] = len(summary_data["summary_content"].split())
                
            if ("compression_ratio" not in summary_data["statistics"] and 
                    "original_word_count" in summary_data["statistics"] and 
                    "summary_word_count" in summary_data["statistics"]):
                original = summary_data["statistics"]["original_word_count"]
                summary = summary_data["statistics"]["summary_word_count"]
                if original > 0:
                    summary_data["statistics"]["compression_ratio"] = (summary / original) * 100
            
            return summary_data
            
        except Exception as e:
            logger.error(f"Error extracting summary data: {str(e)}", exc_info=True)
            return summary_data  # Return the data structure even on error
    
    @staticmethod
    def get_issues_data(result: Dict[str, Any], context=None) -> Dict[str, Any]:
        """
        Extract standardized issues data from result or context.
        
        Args:
            result: Raw assessment result dictionary
            context: Optional ProcessingContext object
            
        Returns:
            Dictionary with standardized keys for issues data
        """
        # Initialize the standardized data structure
        issues_data = {
            "issues": [],                        # List of issues
            "overall_assessment": None,          # Overall assessment text
            "executive_summary": None,           # Executive summary
            "metadata": {},                      # Metadata about the document
            "statistics": {},                    # Processing statistics
            "evidence": {}                       # Evidence links for issues
        }
        
        try:
            # Try to get data from context first (preferred source)
            if context:
                logger.debug("Attempting to extract issues data from context")
                
                # Try to get from context using context manager pattern if available
                if hasattr(context, "get_data_for_agent"):
                    # Get issues from evaluated data (preferred)
                    issues = context.get_data_for_agent("evaluator", "issues")
                    if issues:
                        issues_data["issues"] = issues
                        logger.debug(f"Extracted {len(issues)} issues from evaluated data")
                    else:
                        # Try aggregated data as fallback
                        issues = context.get_data_for_agent("aggregator", "issues")
                        if issues:
                            issues_data["issues"] = issues
                            logger.debug(f"Extracted {len(issues)} issues from aggregated data")
                        else:
                            # Try extracted data as last resort
                            issues = context.get_data_for_agent("extractor", "issues")
                            if issues:
                                issues_data["issues"] = issues
                                logger.debug(f"Extracted {len(issues)} issues from extracted data")
                    
                    # Get overall assessment
                    overall = context.get_data_for_agent("evaluator", "overall_assessment")
                    if overall:
                        issues_data["overall_assessment"] = overall
                        # Extract executive summary if available
                        if isinstance(overall, dict) and "executive_summary" in overall:
                            issues_data["executive_summary"] = overall["executive_summary"]
                            logger.debug("Extracted executive summary from overall assessment")
                
                # Get formatted result for metadata
                formatted = context.get_data_for_agent("formatter")
                if formatted and isinstance(formatted, dict):
                    if "metadata" in formatted:
                        issues_data["metadata"] = formatted["metadata"]
                    if "statistics" in formatted:
                        issues_data["statistics"] = formatted["statistics"]
                
                # Try to get evidence for issues
                if hasattr(context, "evidence_store") and issues_data["issues"]:
                    # For each issue that has an ID, get its evidence
                    for issue in issues_data["issues"]:
                        if isinstance(issue, dict) and "id" in issue:
                            issue_id = issue["id"]
                            # Try to get evidence using context methods if available
                            if hasattr(context, "get_evidence_for_item"):
                                evidence = context.get_evidence_for_item(issue_id)
                                if evidence:
                                    issues_data["evidence"][issue_id] = evidence
                                    # Add evidence to the issue directly for convenience
                                    issue["_evidence"] = evidence
                                    logger.debug(f"Found {len(evidence)} evidence items for issue {issue_id}")
            
            # If we don't have what we need from context, try the raw result
            logger.debug("Attempting to extract missing data from raw result")
            
            # Extract issues if not already found
            if not issues_data["issues"]:
                issues = DataAccessor._get_from_paths(result, [
                    "issues",
                    "result.issues",
                    "extracted_info.issues"
                ])
                if issues and isinstance(issues, list):
                    issues_data["issues"] = issues
                    logger.debug(f"Extracted {len(issues)} issues from raw result")
            
            # Extract executive summary if not already found
            if not issues_data["executive_summary"]:
                exec_summary = DataAccessor._get_from_paths(result, [
                    "executive_summary",
                    "result.executive_summary",
                    "overall_assessment.executive_summary",
                    "result.overall_assessment.executive_summary"
                ])
                if exec_summary:
                    issues_data["executive_summary"] = exec_summary
                    logger.debug("Extracted executive summary from raw result")
            
            # Extract overall assessment if not already found
            if not issues_data["overall_assessment"]:
                overall = DataAccessor._get_from_paths(result, [
                    "overall_assessment",
                    "result.overall_assessment"
                ])
                if overall:
                    issues_data["overall_assessment"] = overall
                    logger.debug("Extracted overall assessment from raw result")
            
            # Extract metadata if not already found
            if not issues_data["metadata"]:
                metadata = DataAccessor._get_from_paths(result, [
                    "metadata",
                    "result.metadata"
                ])
                if metadata and isinstance(metadata, dict):
                    issues_data["metadata"] = metadata
            
            # Extract statistics if not already found
            if not issues_data["statistics"]:
                statistics = DataAccessor._get_from_paths(result, [
                    "statistics",
                    "result.statistics"
                ])
                if statistics and isinstance(statistics, dict):
                    issues_data["statistics"] = statistics
            
            # Ensure issues is a list (handle null or non-list values)
            if not isinstance(issues_data["issues"], list):
                issues_data["issues"] = []
            
            # Extract evidence from raw result if available
            if issues_data["issues"] and not issues_data["evidence"]:
                for issue in issues_data["issues"]:
                    if isinstance(issue, dict) and "evidence" in issue and isinstance(issue["evidence"], list):
                        issue_id = issue.get("id")
                        if issue_id:
                            issues_data["evidence"][issue_id] = issue["evidence"]
                        
                        # Add _evidence field for convenience
                        if "evidence" in issue and not "_evidence" in issue:
                            issue["_evidence"] = issue["evidence"]
                
            # Try to calculate some statistics if not available
            if "total_issues" not in issues_data["statistics"]:
                issues_data["statistics"]["total_issues"] = len(issues_data["issues"])
                
            # Calculate severity counts if not already done
            if "by_severity" not in issues_data["statistics"]:
                severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
                for issue in issues_data["issues"]:
                    if isinstance(issue, dict):
                        # Use evaluated_severity if available, otherwise use severity
                        severity = issue.get("evaluated_severity", issue.get("severity", "medium")).lower()
                        if severity in severity_counts:
                            severity_counts[severity] += 1
                
                issues_data["statistics"]["by_severity"] = severity_counts
            
            return issues_data
            
        except Exception as e:
            logger.error(f"Error extracting issues data: {str(e)}", exc_info=True)
            return issues_data  # Return the data structure even on error

    @staticmethod
    def get_action_items_data(result: Dict[str, Any], context=None) -> Dict[str, Any]:
        """
        Extract standardized action items data from result or context.
        
        Args:
            result: Raw assessment result dictionary
            context: Optional ProcessingContext object
            
        Returns:
            Dictionary with standardized keys for action items data
        """
        # Initialize the standardized data structure
        action_items_data = {
            "action_items": [],                  # List of action items
            "summary": None,                     # Summary text (if available)
            "metadata": {},                      # Metadata about the document
            "statistics": {},                    # Processing statistics
            "evidence": {}                       # Evidence links for action items
        }
        
        try:
            # Try to get data from context first (preferred source)
            if context:
                logger.debug("Attempting to extract action items data from context")
                
                # Try to get from context using context manager pattern if available
                if hasattr(context, "get_data_for_agent"):
                    # Get action items from evaluated data (preferred)
                    action_items = context.get_data_for_agent("evaluator", "action_items")
                    if action_items:
                        action_items_data["action_items"] = action_items
                        logger.debug(f"Extracted {len(action_items)} action items from evaluated data")
                    else:
                        # Try aggregated data as fallback
                        action_items = context.get_data_for_agent("aggregator", "action_items")
                        if action_items:
                            action_items_data["action_items"] = action_items
                            logger.debug(f"Extracted {len(action_items)} action items from aggregated data")
                        else:
                            # Try extracted data as last resort
                            action_items = context.get_data_for_agent("extractor", "action_items")
                            if action_items:
                                action_items_data["action_items"] = action_items
                                logger.debug(f"Extracted {len(action_items)} action items from extracted data")
                
                # Get formatted result for metadata
                formatted = context.get_data_for_agent("formatter")
                if formatted and isinstance(formatted, dict):
                    if "metadata" in formatted:
                        action_items_data["metadata"] = formatted["metadata"]
                    if "statistics" in formatted:
                        action_items_data["statistics"] = formatted["statistics"]
                    if "summary" in formatted:
                        action_items_data["summary"] = formatted["summary"]
                
                # Try to get evidence for action items
                if hasattr(context, "evidence_store") and action_items_data["action_items"]:
                    # For each action item that has an ID, get its evidence
                    for item in action_items_data["action_items"]:
                        if isinstance(item, dict) and "id" in item:
                            item_id = item["id"]
                            # Try to get evidence using context methods if available
                            if hasattr(context, "get_evidence_for_item"):
                                evidence = context.get_evidence_for_item(item_id)
                                if evidence:
                                    action_items_data["evidence"][item_id] = evidence
                                    # Add evidence to the item directly for convenience
                                    item["_evidence"] = evidence
                                    logger.debug(f"Found {len(evidence)} evidence items for action item {item_id}")
            
            # If we don't have what we need from context, try the raw result
            logger.debug("Attempting to extract missing data from raw result")
            
            # Extract action items if not already found
            if not action_items_data["action_items"]:
                action_items = DataAccessor._get_from_paths(result, [
                    "action_items",
                    "result.action_items",
                    "extracted_info.action_items"
                ])
                if action_items and isinstance(action_items, list):
                    action_items_data["action_items"] = action_items
                    logger.debug(f"Extracted {len(action_items)} action items from raw result")
            
            # Extract summary if not already found
            if not action_items_data["summary"]:
                summary = DataAccessor._get_from_paths(result, [
                    "summary",
                    "result.summary"
                ])
                if summary:
                    action_items_data["summary"] = summary
                    logger.debug("Extracted summary from raw result")
            
            # Extract metadata if not already found
            if not action_items_data["metadata"]:
                metadata = DataAccessor._get_from_paths(result, [
                    "metadata",
                    "result.metadata"
                ])
                if metadata and isinstance(metadata, dict):
                    action_items_data["metadata"] = metadata
            
            # Extract statistics if not already found
            if not action_items_data["statistics"]:
                statistics = DataAccessor._get_from_paths(result, [
                    "statistics",
                    "result.statistics"
                ])
                if statistics and isinstance(statistics, dict):
                    action_items_data["statistics"] = statistics
            
            # Ensure action_items is a list (handle null or non-list values)
            if not isinstance(action_items_data["action_items"], list):
                action_items_data["action_items"] = []
            
            # Extract evidence from raw result if available
            if action_items_data["action_items"] and not action_items_data["evidence"]:
                for item in action_items_data["action_items"]:
                    if isinstance(item, dict) and "evidence" in item and isinstance(item["evidence"], list):
                        item_id = item.get("id")
                        if item_id:
                            action_items_data["evidence"][item_id] = item["evidence"]
                        
                        # Add _evidence field for convenience
                        if "evidence" in item and not "_evidence" in item:
                            item["_evidence"] = item["evidence"]
            
            # Try to calculate some statistics if not available
            if "total_action_items" not in action_items_data["statistics"]:
                action_items_data["statistics"]["total_action_items"] = len(action_items_data["action_items"])
                
            # Calculate statistics by owner and priority if not already done
            if "by_owner" not in action_items_data["statistics"]:
                owner_counts = {}
                for item in action_items_data["action_items"]:
                    if isinstance(item, dict):
                        owner = item.get("owner", "Unassigned")
                        owner_counts[owner] = owner_counts.get(owner, 0) + 1
                
                action_items_data["statistics"]["by_owner"] = owner_counts
            
            if "by_priority" not in action_items_data["statistics"]:
                priority_counts = {"high": 0, "medium": 0, "low": 0}
                for item in action_items_data["action_items"]:
                    if isinstance(item, dict):
                        # Use evaluated_priority if available, otherwise use priority
                        priority = item.get("evaluated_priority", item.get("priority", "medium")).lower()
                        if priority in priority_counts:
                            priority_counts[priority] += 1
                
                action_items_data["statistics"]["by_priority"] = priority_counts
            
            return action_items_data
            
        except Exception as e:
            logger.error(f"Error extracting action items data: {str(e)}", exc_info=True)
            return action_items_data  # Return the data structure even on error

    @staticmethod
    def get_analysis_data(result: Dict[str, Any], context=None) -> Dict[str, Any]:
        """
        Extract standardized analysis data from result or context.
        
        Args:
            result: Raw assessment result dictionary
            context: Optional ProcessingContext object
            
        Returns:
            Dictionary with standardized keys for analysis data
        """
        # Initialize the standardized data structure
        analysis_data = {
            "dimension_assessments": [],         # List of dimension assessments
            "overall_rating": None,              # Overall rating
            "strategic_recommendations": [],     # List of recommendations
            "strengths": [],                     # List of strengths
            "weaknesses": [],                    # List of weaknesses
            "summary": None,                     # Summary text (if available)
            "metadata": {},                      # Metadata about the document
            "statistics": {},                    # Processing statistics
            "evidence": {}                       # Evidence links for assessments
        }
        
        try:
            # Try to get data from context first (preferred source)
            if context:
                logger.debug("Attempting to extract analysis data from context")
                
                # Try to get from context using context manager pattern if available
                if hasattr(context, "get_data_for_agent"):
                    # Get analysis data from appropriate agents
                    formatted = context.get_data_for_agent("formatter")
                    if formatted and isinstance(formatted, dict):
                        if "dimension_assessments" in formatted:
                            analysis_data["dimension_assessments"] = formatted["dimension_assessments"]
                        if "overall_rating" in formatted:
                            analysis_data["overall_rating"] = formatted["overall_rating"]
                        if "strategic_recommendations" in formatted:
                            analysis_data["strategic_recommendations"] = formatted["strategic_recommendations"]
                        if "strengths" in formatted:
                            analysis_data["strengths"] = formatted["strengths"]
                        if "weaknesses" in formatted:
                            analysis_data["weaknesses"] = formatted["weaknesses"]
                        if "summary" in formatted:
                            analysis_data["summary"] = formatted["summary"]
                        if "metadata" in formatted:
                            analysis_data["metadata"] = formatted["metadata"]
                        if "statistics" in formatted:
                            analysis_data["statistics"] = formatted["statistics"]
                
                # Try to get evidence for dimensions
                if hasattr(context, "evidence_store") and analysis_data["dimension_assessments"]:
                    # For each dimension that has an ID, get its evidence
                    for dimension in analysis_data["dimension_assessments"]:
                        if isinstance(dimension, dict) and "id" in dimension:
                            dimension_id = dimension["id"]
                            # Try to get evidence using context methods if available
                            if hasattr(context, "get_evidence_for_item"):
                                evidence = context.get_evidence_for_item(dimension_id)
                                if evidence:
                                    analysis_data["evidence"][dimension_id] = evidence
                                    # Add evidence to the dimension directly for convenience
                                    dimension["_evidence"] = evidence
            
            # If we don't have what we need from context, try the raw result
            logger.debug("Attempting to extract missing data from raw result")
            
            # Extract dimension assessments if not already found
            if not analysis_data["dimension_assessments"]:
                dimensions = DataAccessor._get_from_paths(result, [
                    "dimension_assessments",
                    "result.dimension_assessments",
                    "dimensions"
                ])
                if dimensions and isinstance(dimensions, list):
                    analysis_data["dimension_assessments"] = dimensions
                    logger.debug(f"Extracted {len(dimensions)} dimension assessments from raw result")
            
            # Extract overall rating if not already found
            if not analysis_data["overall_rating"]:
                overall_rating = DataAccessor._get_from_paths(result, [
                    "overall_rating",
                    "result.overall_rating"
                ])
                if overall_rating:
                    analysis_data["overall_rating"] = overall_rating
                    logger.debug("Extracted overall rating from raw result")
            
            # Extract recommendations if not already found
            if not analysis_data["strategic_recommendations"]:
                recommendations = DataAccessor._get_from_paths(result, [
                    "strategic_recommendations",
                    "result.strategic_recommendations",
                    "recommendations"
                ])
                if recommendations and isinstance(recommendations, list):
                    analysis_data["strategic_recommendations"] = recommendations
                    logger.debug(f"Extracted {len(recommendations)} recommendations from raw result")
            
            # Extract strengths if not already found
            if not analysis_data["strengths"]:
                strengths = DataAccessor._get_from_paths(result, [
                    "strengths",
                    "result.strengths",
                    "overall_assessment.strengths"
                ])
                if strengths and isinstance(strengths, list):
                    analysis_data["strengths"] = strengths
                    logger.debug(f"Extracted {len(strengths)} strengths from raw result")
            
            # Extract weaknesses if not already found
            if not analysis_data["weaknesses"]:
                weaknesses = DataAccessor._get_from_paths(result, [
                    "weaknesses",
                    "result.weaknesses",
                    "overall_assessment.weaknesses"
                ])
                if weaknesses and isinstance(weaknesses, list):
                    analysis_data["weaknesses"] = weaknesses
                    logger.debug(f"Extracted {len(weaknesses)} weaknesses from raw result")
            
            # Extract summary if not already found
            if not analysis_data["summary"]:
                summary = DataAccessor._get_from_paths(result, [
                    "summary",
                    "result.summary",
                    "executive_summary"
                ])
                if summary:
                    analysis_data["summary"] = summary
                    logger.debug("Extracted summary from raw result")
            
            # Extract metadata if not already found
            if not analysis_data["metadata"]:
                metadata = DataAccessor._get_from_paths(result, [
                    "metadata",
                    "result.metadata"
                ])
                if metadata and isinstance(metadata, dict):
                    analysis_data["metadata"] = metadata
            
            # Extract statistics if not already found
            if not analysis_data["statistics"]:
                statistics = DataAccessor._get_from_paths(result, [
                    "statistics",
                    "result.statistics"
                ])
                if statistics and isinstance(statistics, dict):
                    analysis_data["statistics"] = statistics
            
            # Ensure lists are actually lists (handle null or non-list values)
            if not isinstance(analysis_data["dimension_assessments"], list):
                analysis_data["dimension_assessments"] = []
            if not isinstance(analysis_data["strategic_recommendations"], list):
                analysis_data["strategic_recommendations"] = []
            if not isinstance(analysis_data["strengths"], list):
                analysis_data["strengths"] = []
            if not isinstance(analysis_data["weaknesses"], list):
                analysis_data["weaknesses"] = []
            
            return analysis_data
            
        except Exception as e:
            logger.error(f"Error extracting analysis data: {str(e)}", exc_info=True)
            return analysis_data  # Return the data structure even on error

    @staticmethod
    def get_data_by_assessment_type(result: Dict[str, Any], assessment_type: str, context=None) -> Dict[str, Any]:
        """
        Get standardized data based on assessment type.
        
        Args:
            result: Raw assessment result dictionary
            assessment_type: Type of assessment ("distill", "extract", "assess", "analyze")
            context: Optional ProcessingContext object
            
        Returns:
            Dictionary with standardized data for the specified assessment type
        """
        if assessment_type == "distill":
            return DataAccessor.get_summary_data(result, context)
        elif assessment_type == "extract":
            return DataAccessor.get_action_items_data(result, context)
        elif assessment_type == "assess":
            return DataAccessor.get_issues_data(result, context)
        elif assessment_type == "analyze":
            return DataAccessor.get_analysis_data(result, context)
        else:
            logger.warning(f"Unknown assessment type: {assessment_type}")
            return {}

    @staticmethod
    def validate_data(data: Dict[str, Any], assessment_type: str) -> Tuple[bool, Optional[str]]:
        """
        Validate the data for a specific assessment type.
        
        Args:
            data: Data dictionary to validate
            assessment_type: Type of assessment
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if assessment_type == "distill":
                # Check for either a summary content or key points
                has_summary = data.get("summary_content") is not None and len(data.get("summary_content", "")) > 10
                has_key_points = isinstance(data.get("key_points"), list) and len(data.get("key_points", [])) > 0
                
                if not (has_summary or has_key_points):
                    return False, "Missing both summary content and key points"
                
                return True, None
                
            elif assessment_type == "assess":
                # Check for issues
                has_issues = isinstance(data.get("issues"), list) and len(data.get("issues", [])) > 0
                
                if not has_issues:
                    return False, "No issues found in assessment data"
                
                return True, None
                
            elif assessment_type == "extract":
                # Check for action items
                has_action_items = isinstance(data.get("action_items"), list) and len(data.get("action_items", [])) > 0
                
                if not has_action_items:
                    return False, "No action items found in assessment data"
                
                return True, None
                
            elif assessment_type == "analyze":
                # Check for dimension assessments
                has_dimensions = isinstance(data.get("dimension_assessments"), list) and len(data.get("dimension_assessments", [])) > 0
                
                if not has_dimensions:
                    return False, "No dimension assessments found in analysis data"
                
                return True, None
                
            else:
                return False, f"Unknown assessment type: {assessment_type}"
                
        except Exception as e:
            return False, f"Error validating data: {str(e)}"

    @staticmethod
    def _get_from_paths(data: Dict[str, Any], paths: List[str], default=None) -> Any:
        """
        Try to extract data from a dictionary using multiple possible paths.
        
        Args:
            data: Dictionary to extract data from
            paths: List of dot-separated paths to try (e.g., ["result.issues", "issues"])
            default: Default value to return if no path works
            
        Returns:
            Extracted data or default value
        """
        if not data or not isinstance(data, dict):
            return default
            
        for path in paths:
            try:
                current = data
                parts = path.split('.')
                
                # Navigate through each part of the path
                for part in parts:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        # Path doesn't exist, break and try next path
                        current = None
                        break
                
                # If we got a value, return it
                if current is not None:
                    return current
                    
            except Exception:
                # If any error occurs, try the next path
                continue
                
        # If no path worked, return the default
        return default

    @staticmethod
    def add_evidence_to_items(data: Dict[str, Any], context=None) -> Dict[str, Any]:
        """
        Add evidence to items in the data dictionary from context.
        
        Args:
            data: Data dictionary containing items (issues, key points, etc.)
            context: ProcessingContext object to get evidence from
            
        Returns:
            Updated data dictionary
        """
        # Create a deep copy to avoid modifying the original
        result = copy.deepcopy(data)
        
        try:
            if not context or not hasattr(context, "get_evidence_for_item"):
                return result
                
            # Define item lists based on assessment type
            item_lists = []
            
            if "issues" in result and isinstance(result["issues"], list):
                item_lists.append(("issues", result["issues"]))
            if "key_points" in result and isinstance(result["key_points"], list):
                item_lists.append(("key_points", result["key_points"]))
            if "action_items" in result and isinstance(result["action_items"], list):
                item_lists.append(("action_items", result["action_items"]))
            if "dimension_assessments" in result and isinstance(result["dimension_assessments"], list):
                item_lists.append(("dimension_assessments", result["dimension_assessments"]))
            
            # Process each item list
            for list_name, items in item_lists:
                for item in items:
                    if isinstance(item, dict) and "id" in item:
                        item_id = item["id"]
                        evidence = context.get_evidence_for_item(item_id)
                        if evidence:
                            item["_evidence"] = evidence
                            logger.debug(f"Added {len(evidence)} evidence items to {list_name} item {item_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error adding evidence to items: {str(e)}", exc_info=True)
            return result  # Return the unchanged data on error