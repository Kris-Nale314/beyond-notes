# core/agents/reviewer.py
import logging
import json
from typing import Dict, Any, List, Optional

# Import base class and context/LLM types
from .base import BaseAgent
from core.llm.customllm import CustomLLM
from core.models.context import ProcessingContext

class ReviewerAgent(BaseAgent):
    """
    Performs a quality control review of the formatted output produced by
    the FormatterAgent, checking for consistency, completeness, and alignment
    with the assessment configuration. Provides feedback but does not modify
    the formatted output directly.
    
    Uses the enhanced ProcessingContext for standardized data storage and retrieval.
    """

    def __init__(self, llm: CustomLLM, options: Optional[Dict[str, Any]] = None):
        """Initialize the ReviewerAgent."""
        super().__init__(llm, options)
        self.role = "reviewer"
        self.name = "ReviewerAgent"
        self.logger = logging.getLogger(f"core.agents.{self.name}")
        self.logger.info(f"ReviewerAgent initialized.")

    async def process(self, context: ProcessingContext) -> Optional[Dict[str, Any]]:
        """
        Reviews the Formatter's output and stores feedback in the context.

        Args:
            context: The shared ProcessingContext object containing the formatted report.

        Returns:
            The dictionary containing the review results, or None if review cannot be performed.
        """
        self._log_info(f"Starting review phase for assessment: '{context.display_name}'", context)

        # --- Get Data using the enhanced context method ---
        formatted_output = self._get_data(context, "formatter", "formatted", None)
        if not formatted_output or not isinstance(formatted_output, dict):
            self._log_warning("No valid formatted output found in context. Cannot perform review.", context)
            # Store minimal review result in context
            error_review = {"error": "Could not perform review: No formatted output available."}
            self._store_data(context, "review", error_review)
            return error_review

        try:
            # --- Get Configuration ---
            assessment_config = context.get_assessment_config()
            output_schema = context.get_output_schema()
            base_reviewer_instructions = context.get_workflow_instructions(self.role) or "Review the formatted output for quality, consistency, and adherence to the schema."

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

            # --- Construct Prompt for LLM Review ---
            # Summarize the report slightly for the prompt if it's huge
            report_preview_json = json.dumps(formatted_output, indent=2, default=str, ensure_ascii=False)
            max_preview_len = self.options.get("review_preview_length", 6000)
            if len(report_preview_json) > max_preview_len:
                report_preview_json = report_preview_json[:max_preview_len] + "\n... [Report Truncated for Review Prompt] ..."
                self._log_warning(f"Report preview truncated to {max_preview_len} chars for reviewer prompt.", context)

            prompt = f"""
You are an expert AI 'Reviewer' agent. Your task is to perform a quality control check on a JSON output generated by a previous 'Formatter' agent.

**Assessment Context:**
* **Assessment ID:** {context.assessment_id}
* **Assessment Type:** {context.assessment_type} ({context.display_name})
* **Document:** {context.document_info.get('filename', 'N/A')}

**Expected Output Schema (What the Formatter *should* have produced):**
```json
{json.dumps(output_schema, indent=2)}
```

**Formatted Output to Review:**
```json
{report_preview_json}
```

**Base Instructions for Reviewer:** {base_reviewer_instructions}

**Your Task:**
Critically evaluate the 'Formatted Output to Review'. Assess its quality, clarity, completeness (based on common sense for the assessment type), and how well it adheres to the 'Expected Output Schema'. Provide constructive feedback.

**Output Format:** Respond only with a valid JSON object matching this schema:
```json
{json.dumps(review_output_schema, indent=2)}
```

Focus on providing actionable suggestions if improvements are needed.
Be objective in your scoring.
"""

            # --- LLM Call to Perform Review ---
            review_result = await self._generate_structured(
                prompt=prompt,
                output_schema=review_output_schema,
                context=context,
                temperature=self.options.get("reviewer_temperature", 0.3),
                max_tokens=self.options.get("reviewer_max_tokens", 2000)
            )

            # --- Store review results using the enhanced context method ---
            self._store_data(context, "review", review_result)
            self._log_info(f"Review complete. Overall Score: {review_result.get('overall_quality_score', 'N/A')}/10", context)

            return review_result

        except Exception as e:
            self._log_error(f"Review phase failed: {str(e)}", context, exc_info=True)
            # Store error info in context
            error_review = {
                "error": f"Review Agent Failed: {str(e)}",
                "message": "Could not generate review feedback for the formatted output."
            }
            self._store_data(context, "review", error_review)
            # Re-raise the exception to fail the stage
            raise