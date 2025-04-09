import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

# Import base class and context/LLM types
from .base import BaseAgent
from core.llm.customllm import CustomLLM
from core.models.context import ProcessingContext

logger = logging.getLogger(__name__)

class ReviewerAgent(BaseAgent):
    """
    Performs a quality control review of the formatted output produced by
    the FormatterAgent, checking for consistency, completeness, and adherence
    to the assessment configuration.
    
    This implementation addresses truncation issues by using a "smart preview"
    approach that preserves key information while staying within context windows.
    """

    def __init__(self, llm: CustomLLM, options: Optional[Dict[str, Any]] = None):
        """Initialize the ReviewerAgent."""
        super().__init__(llm, options)
        self.role = "reviewer"
        self.logger = logging.getLogger(f"core.agents.{self.__class__.__name__}")
        self.logger.info(f"ReviewerAgent initialized")

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Reviews the Formatter's output and stores feedback in the context.

        Args:
            context: The shared ProcessingContext object containing the formatted report.

        Returns:
            The dictionary containing the review results, or None if review cannot be performed.
        """
        self._log_info(f"Starting review phase for assessment: '{context.display_name}'", context)

        # --- Get Data using BaseAgent methods ---
        formatted_output = self._get_data(context, "formatter", None)
        if not formatted_output or not isinstance(formatted_output, dict):
            self._log_warning("No valid formatted output found in context. Cannot perform review.", context)
            # Store minimal review result in context
            error_review = {
                "error": "Could not perform review: No formatted output available.",
                "status": "failed"
            }
            self._store_data(context, None, error_review)
            return error_review

        try:
            # --- Get Configuration ---
            assessment_type = context.assessment_type
            output_schema = context.get_output_schema()
            reviewer_instructions = context.get_workflow_instructions(self.role) or "Review the formatted output for quality, consistency, and adherence to the schema."

            # --- Define Schema for Review Output ---
            review_output_schema = {
                "type": "object",
                "properties": {
                    "overall_quality_score": {"type": "number", "minimum": 0, "maximum": 10, "description": "Overall quality rating (0-10)."},
                    "schema_adherence_score": {"type": "number", "minimum": 0, "maximum": 10, "description": "How well the output matches the required schema (0-10)."},
                    "clarity_score": {"type": "number", "minimum": 0, "maximum": 10, "description": "Clarity and readability rating (0-10)."},
                    "completeness_score": {"type": "number", "minimum": 0, "maximum": 10, "description": "Assessment of whether key information seems present based on inputs (0-10)."},
                    "review_summary": {"type": "string", "description": "A concise summary paragraph of the review findings."},
                    "strengths": {"type": "array", "items": {"type": "string"}, "description": "List 2-3 specific strengths of the formatted output."},
                    "areas_for_improvement": {"type": "array", "items": {"type": "string"}, "description": "List 2-3 specific areas needing improvement."},
                    "specific_suggestions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string", "description": "Specific key/path within the formatted output where issue occurs (e.g., 'result.summary', 'result.action_items[0].owner')."},
                                "issue_description": {"type": "string", "description": "Description of the identified issue."},
                                "suggestion": {"type": "string", "description": "Concrete suggestion for improvement."}
                            },
                            "required": ["location", "issue_description", "suggestion"]
                        },
                        "description": "List of specific, actionable suggestions for improvement."
                    }
                },
                "required": ["overall_quality_score", "schema_adherence_score", "clarity_score", "completeness_score", "review_summary", "strengths", "areas_for_improvement", "specific_suggestions"]
            }

            # --- Prepare Smart Preview of Output ---
            preview_data = self._create_smart_preview(formatted_output, assessment_type)
            
            # Get evaluated items summary for context
            evaluated_items_count = 0
            data_type = self._get_data_type_for_assessment(assessment_type)
            if data_type:
                evaluated_items = self._get_data(context, "evaluator", data_type, [])
                evaluated_items_count = len(evaluated_items)

            # --- Construct Prompt for LLM Review ---
            prompt = f"""
You are an expert AI 'Reviewer' agent. Your task is to perform a quality control check on a JSON output generated by a previous 'Formatter' agent.

**Assessment Context:**
* **Assessment Type:** {assessment_type} ({context.display_name})
* **Document:** {context.document_info.get('filename', 'N/A')}
* **Total Items:** {evaluated_items_count} items were evaluated

**Expected Output Schema (What the Formatter should follow):**
```json
{json.dumps(output_schema, indent=2)}
```

**Formatted Output to Review (Smart Preview):**
```json
{json.dumps(preview_data, indent=2, default=str)}
```

**Additional Context:**
The output contains a total of {preview_data.get('_metadata', {}).get('total_items', 0)} items across all arrays.
{preview_data.get('_metadata', {}).get('truncation_note', '')}

**Review Instructions:**
{reviewer_instructions}

Critically evaluate the 'Formatted Output' for:
1. **Quality** - Is the content well-organized, valuable, and actionable?
2. **Schema Adherence** - Does it follow the expected schema structure?
3. **Clarity** - Is the information presented clearly and logically?
4. **Completeness** - Are all required elements present and substantive?

**Output Format:** Respond only with a valid JSON object matching this schema:
```json
{json.dumps(review_output_schema, indent=2)}
```

Focus on providing actionable suggestions if improvements are needed.
Be objective in your scoring.
If the output is excellent, say so, but still provide at least one suggestion for improvement.
"""

            # --- LLM Call to Perform Review ---
            temperature = self.options.get("reviewer_temperature", 0.3)
            max_tokens = self.options.get("reviewer_max_tokens", 3000)
            
            review_result = await self._generate_structured(
                prompt=prompt,
                output_schema=review_output_schema,
                context=context,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # --- Add Review Outcome & Metadata ---
            # Add quality status based on scores
            self._add_review_outcome(review_result)
            
            # Add metadata
            review_result["metadata"] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "assessment_id": context.assessment_id,
                "assessment_type": context.assessment_type
            }

            # --- Store Review Results ---
            self._store_data(context, None, review_result)
            self._log_info(f"Review complete. Overall Score: {review_result.get('overall_quality_score', 'N/A')}/10", context)
            
            return review_result

        except Exception as e:
            self._log_error(f"Review phase failed: {str(e)}", context, exc_info=True)
            
            # Store error info in context
            error_review = {
                "error": f"Review Agent Failed: {str(e)}",
                "message": "Could not generate review feedback for the formatted output.",
                "status": "failed"
            }
            self._store_data(context, None, error_review)
            return error_review

    def _create_smart_preview(self, output: Dict[str, Any], assessment_type: str) -> Dict[str, Any]:
        """
        Create a smart preview of the output that preserves key information
        while staying within context limits.
        """
        # Make a deep copy to avoid modifying the original
        import copy
        preview = copy.deepcopy(output)
        
        # Add metadata placeholder
        preview["_metadata"] = {
            "truncation_note": "",
            "total_items": 0
        }
        
        # Identify and count items in arrays
        total_items = 0
        arrays_to_sample = []
        
        # Helper to find array properties
        def find_arrays(obj, path=None):
            nonlocal total_items
            
            if path is None:
                path = []
                
            if not isinstance(obj, dict):
                return
                
            for key, value in obj.items():
                current_path = path + [key]
                
                # If this is an array of objects
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                    total_items += len(value)
                    arrays_to_sample.append({"path": current_path, "count": len(value)})
                
                # Recurse into nested objects
                elif isinstance(value, dict):
                    find_arrays(value, current_path)
        
        # Find arrays
        find_arrays(preview)
        preview["_metadata"]["total_items"] = total_items
        
        # If we found arrays that need sampling
        if arrays_to_sample:
            # Add note about sampling
            preview["_metadata"]["truncation_note"] = f"The complete output contains {total_items} items across {len(arrays_to_sample)} arrays. This preview shows representative samples."
            
            # Sample arrays
            for array_info in arrays_to_sample:
                path = array_info["path"]
                count = array_info["count"]
                
                # Navigate to array
                current = preview
                for i, key in enumerate(path[:-1]):
                    current = current.get(key, {})
                
                last_key = path[-1]
                if last_key in current:
                    # Sample based on array size
                    items = current[last_key]
                    if len(items) > 5:
                        # Sample items from beginning, middle, and end
                        sampled = items[:2]  # Beginning
                        
                        # Add a middle item if array is large enough
                        if len(items) > 4:
                            middle_idx = len(items) // 2
                            sampled.append(items[middle_idx])
                        
                        # Add end items
                        sampled.extend(items[-2:])
                        
                        # Replace with sampled items
                        current[last_key] = sampled
                        
                        # Add note about sampling
                        sampled.insert(0, {"_note": f"[Showing 5 samples from {count} total items]"})
        
        # Remove metadata if no sampling was needed
        if preview["_metadata"]["total_items"] == 0:
            del preview["_metadata"]
            
        return preview

    def _get_data_type_for_assessment(self, assessment_type: str) -> str:
        """Determine the key/type of data being processed based on assessment type."""
        data_type_map = {
            "extract": "action_items",
            "assess": "issues",
            "distill": "key_points",
            "analyze": "evidence"
        }
        return data_type_map.get(assessment_type, "items")

    def _add_review_outcome(self, review_result: Dict[str, Any]) -> None:
        """
        Add review outcome (status and rating) based on scores.
        """
        # Calculate aggregate score
        scores = [
            review_result.get("overall_quality_score", 0),
            review_result.get("schema_adherence_score", 0),
            review_result.get("clarity_score", 0),
            review_result.get("completeness_score", 0)
        ]
        avg_score = sum(scores) / len(scores) if scores else 0
        avg_score = round(avg_score, 1)
        review_result["aggregate_score"] = avg_score

        # Determine review status
        if avg_score >= 8.0:
            review_result["status"] = "excellent"
        elif avg_score >= 6.0:
            review_result["status"] = "good"
        elif avg_score >= 4.0:
            review_result["status"] = "needs_improvement"
        else:
            review_result["status"] = "poor"
            
        # Add rating for UI display
        rating_descriptions = {
            "excellent": "The output is exceptional, with only minor suggestions for improvement.",
            "good": "The output is solid but has a few areas for improvement.",
            "needs_improvement": "The output has several issues that should be addressed.",
            "poor": "The output has significant problems and should be regenerated."
        }
        
        review_result["rating"] = {
            "value": review_result["status"],
            "score": avg_score,
            "description": rating_descriptions.get(review_result["status"], "")
        }