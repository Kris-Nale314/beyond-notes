"""
Result Accessor Utility

Provides a data access layer to normalize and simplify access to data in the complex
nested result structures produced by the multi-agent assessment pipeline.

Usage:
    from utils.result_accessor import get_assessment_data, get_item_count
    
    # Get normalized data for display or reporting
    data = get_assessment_data(result, assessment_type)
    
    # Access specific data elements regardless of their location
    action_items = data.get("action_items", [])
    summary = data.get("summary", "")
"""

import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

def get_from_result(result: Dict[str, Any], path: str, default=None) -> Any:
    """
    Access a value from the result dictionary using a dot-notation path.
    
    Args:
        result: Result dictionary
        path: Dot-notation path to the value (e.g., "result.summary" or "extracted_info.action_items")
        default: Default value if path not found
        
    Returns:
        The value at the path or default if not found
    """
    if not result or not isinstance(result, dict):
        return default
        
    parts = path.split('.')
    current = result
    
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
        
    return current

def get_assessment_data(result: Dict[str, Any], assessment_type: str) -> Dict[str, Any]:
    """
    Create a normalized view of assessment data regardless of structure.
    
    Args:
        result: The raw result dictionary
        assessment_type: Type of assessment (distill, extract, assess, analyze)
        
    Returns:
        Dictionary with normalized access to assessment data
    """
    # Define search paths for different data types based on the known agent output patterns
    paths = {
        "distill": {
            "overview": ["overview", "result.overview", "summary", "result.summary"],
            "summary": ["summary", "result.summary", "formatting.summary", "results.formatting.summary"],
            "topics": ["topics", "result.topics", "formatting.topics", "results.formatting.topics", 
                      "extracted_info.topics", "results.extracted_key_points", "results.aggregated_key_points"],
            "key_findings": ["key_findings", "result.key_findings", "formatting.key_findings",
                           "extracted_info.key_points", "results.evaluated_key_points"]
        },
        "extract": {
            "summary": ["summary", "result.summary", "formatting.summary", "results.formatting.summary"],
            "action_items": ["action_items", "result.action_items", "formatting.action_items", 
                          "extracted_info.action_items", "results.evaluated_action_items", 
                          "results.aggregated_action_items", "results.extracted_action_items"]
        },
        "assess": {
            "executive_summary": ["executive_summary", "result.executive_summary", "formatting.executive_summary",
                                "results.overall_assessment.executive_summary"],
            "issues": ["issues", "result.issues", "formatting.issues", "extracted_info.issues", 
                     "results.evaluated_issues", "results.aggregated_issues", "evaluations.issues.evaluated_issues"]
        },
        "analyze": {
            "executive_summary": ["executive_summary", "result.executive_summary", "formatting.executive_summary",
                                "results.overall_assessment.executive_summary"],
            "dimension_assessments": ["dimension_assessments", "result.dimension_assessments", 
                                   "formatting.dimension_assessments", "results.evaluated_evidence",
                                   "evaluations.dimension_assessments"],
            "strategic_recommendations": ["strategic_recommendations", "result.strategic_recommendations",
                                      "formatting.strategic_recommendations", 
                                      "results.overall_assessment.strategic_recommendations"]
        }
    }
    
    # Get default paths for this assessment type
    type_paths = paths.get(assessment_type, {})
    
    # Build normalized data dictionary
    data = {}
    
    # First, try to get data from the formatter's output (most reliable location in final output)
    formatter_output = get_from_result(result, "formatting") or get_from_result(result, "result")
    if formatter_output and isinstance(formatter_output, dict):
        # Copy primary formatter output to normalized data
        for key, value in formatter_output.items():
            if value is not None:  # Skip None values
                data[key] = value
                logger.debug(f"Found data for '{key}' in formatter output")
    
    # Then check additional paths for data that might be missing or not yet formatted
    for key, possible_paths in type_paths.items():
        # If key already exists in data with a non-None value, skip lookup
        if key in data and data[key] is not None:
            continue
            
        # Try each possible path
        for path in possible_paths:
            value = get_from_result(result, path)
            if value is not None:
                data[key] = value
                logger.debug(f"Found data for '{key}' via path: {path}")
                break
    
    # Always include metadata and statistics
    if "metadata" not in data or not data["metadata"]:
        data["metadata"] = get_from_result(result, "metadata", {})
    
    if "statistics" not in data or not data["statistics"]:
        data["statistics"] = get_from_result(result, "statistics", {})
    
    # Ensure assessment type is available
    if "assessment_type" not in data.get("metadata", {}):
        data["metadata"]["assessment_type"] = assessment_type
    
    return data

def get_item_count(result: Dict[str, Any], assessment_type: str) -> int:
    """
    Get count of primary items for this assessment type.
    
    Args:
        result: Raw result dictionary
        assessment_type: Type of assessment
        
    Returns:
        Count of items (topics, action items, issues, or dimensions)
    """
    data = get_assessment_data(result, assessment_type)
    
    if assessment_type == "distill":
        topics = data.get("topics", [])
        return len(topics) if isinstance(topics, list) else 0
    elif assessment_type == "extract":
        items = data.get("action_items", [])
        return len(items) if isinstance(items, list) else 0
    elif assessment_type == "assess":
        issues = data.get("issues", [])
        return len(issues) if isinstance(issues, list) else 0
    elif assessment_type == "analyze":
        dimensions = data.get("dimension_assessments", {})
        if isinstance(dimensions, dict):
            return len(dimensions)
        elif isinstance(dimensions, list):
            return len(dimensions)
        return 0
    else:
        return 0

def debug_result_structure(result: Dict[str, Any], max_depth: int = 3) -> Dict[str, Any]:
    """
    Create a debug representation of the result structure.
    
    Args:
        result: The raw result dictionary
        max_depth: Maximum depth to analyze
        
    Returns:
        Dictionary with debug information
    """
    def get_paths(data, prefix="", current_depth=0):
        """Recursively get all paths in the data structure."""
        paths = []
        if current_depth >= max_depth:
            return [f"{prefix} (max depth reached)"]
            
        if isinstance(data, dict):
            if not data:
                return [f"{prefix} (empty dict)"]
                
            for key, value in data.items():
                current = f"{prefix}.{key}" if prefix else key
                paths.append(current)
                if isinstance(value, (dict, list)) and value:
                    paths.extend(get_paths(value, current, current_depth+1))
        elif isinstance(data, list):
            if not data:
                return [f"{prefix} (empty list)"]
                
            # Show first item structure for non-empty lists
            paths.append(f"{prefix} (list with {len(data)} items)")
            if data and current_depth < max_depth - 1:
                sample_paths = get_paths(data[0], f"{prefix}[0]", current_depth+1)
                paths.extend(sample_paths)
                
        return paths
    
    all_paths = get_paths(result)
    
    # Process paths to show structured content
    debug_info = {
        "top_level_keys": list(result.keys() if isinstance(result, dict) else []),
        "paths": all_paths,
        "primary_data_locations": {}
    }
    
    # Check for primary data in common locations
    locations = [
        "result", "formatting", "extracted_info", "evaluations", "statistics",
        "results.formatting", "results.evaluated_issues", "results.evaluated_action_items",
        "results.extracted_key_points"
    ]
    
    for loc in locations:
        content = get_from_result(result, loc)
        if content:
            if isinstance(content, dict):
                debug_info["primary_data_locations"][loc] = list(content.keys())
            elif isinstance(content, list):
                debug_info["primary_data_locations"][loc] = f"list with {len(content)} items"
            else:
                debug_info["primary_data_locations"][loc] = type(content).__name__
    
    return debug_info

def find_data_in_result(result: Dict[str, Any], key: str, assessment_type: str = None) -> Any:
    """
    Look for data in multiple possible locations in the result.
    
    This is a legacy function maintained for backward compatibility.
    Prefer get_assessment_data() for new code.
    
    Args:
        result: Result dictionary
        key: Key to find (e.g., "action_items", "issues")
        assessment_type: Optional assessment type for context
        
    Returns:
        Data found or None
    """
    # Check direct key
    if key in result and result[key]:
        return result[key]
        
    # Check in extracted_info
    if "extracted_info" in result and key in result["extracted_info"] and result["extracted_info"][key]:
        return result["extracted_info"][key]
    
    # Check in results structure
    if "results" in result:
        # Try formatted results first
        if "formatting" in result["results"] and key in result["results"]["formatting"]:
            return result["results"]["formatting"][key]
            
        # Then try evaluated/aggregated items
        for prefix in ["evaluated_", "aggregated_"]:
            results_key = f"{prefix}{key}"
            if results_key in result["results"]:
                return result["results"][results_key]
    
    # Check in result key (Formatter output)
    if "result" in result and key in result["result"]:
        return result["result"][key]
        
    # If assessment type provided, check in type-specific locations
    if assessment_type:
        if assessment_type == "assess" and key == "issues":
            if "evaluations" in result and "issues" in result["evaluations"]:
                return result["evaluations"]["issues"].get("evaluated_issues", [])
    
        elif assessment_type == "analyze" and key == "dimension_assessments":
            if "result" in result and "dimension_assessments" in result["result"]:
                return result["result"]["dimension_assessments"]
    
        elif assessment_type == "distill" and key == "summary":
            if "result" in result and "summary" in result["result"]:
                return result["result"]["summary"]
    
    # Not found
    return None