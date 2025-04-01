# core/agents/planner.py
import logging
from typing import Dict, Any, List, Optional
import json

# Import base class and context/LLM types
from .base import BaseAgent
from core.llm.customllm import CustomLLM
from core.models.context import ProcessingContext

class PlannerAgent(BaseAgent):
    """
    Analyzes the document and assessment configuration to create a tailored
    processing plan and potentially refine instructions for downstream agents.
    
    Uses the enhanced ProcessingContext for standardized data storage and retrieval.
    """

    def __init__(self, llm: CustomLLM, options: Optional[Dict[str, Any]] = None):
        """Initialize the PlannerAgent."""
        super().__init__(llm, options)
        self.role = "planner"  # Define the specific role
        self.name = "PlannerAgent"
        self.logger = logging.getLogger(f"core.agents.{self.name}")
        self.logger.info(f"PlannerAgent initialized.")

    async def process(self, context: ProcessingContext) -> Optional[Dict[str, Any]]:
        """
        Analyzes the document context and assessment config to generate a plan.

        Args:
            context: The shared ProcessingContext object.

        Returns:
            A dictionary containing the processing plan, or None if planning fails.
        """
        self._log_info(f"Starting planning phase for assessment: '{context.display_name}'", context)

        try:
            # --- Get necessary info using context helpers ---
            document_text_preview = context.document_text[:self.options.get("preview_length", 5000)]
            document_info = context.document_info
            assessment_type = context.assessment_type
            assessment_config = context.get_assessment_config()
            assessment_description = assessment_config.get('description', '')

            # Get base planner instructions from the workflow config via context
            base_planner_instructions = context.get_workflow_instructions(self.role)
            if not base_planner_instructions:
                self._log_warning("No specific instructions found for planner in workflow config.", context)
                base_planner_instructions = "Analyze the document start and create a processing plan."  # Basic fallback

            # Get the core target definition for this assessment type
            target_definition = context.get_target_definition()
            if not target_definition:
                self._log_error(f"Could not retrieve target definition for assessment type '{assessment_type}'. Cannot proceed with planning.", context)
                raise ValueError(f"Missing target definition for type {assessment_type} in config.")

            target_name = target_definition.get("name", assessment_type)
            target_desc = target_definition.get("description", "No description provided.")

            # --- Construct the Planning Prompt ---
            prompt = f"""
You are an expert AI assistant acting as the 'Planner' in a multi-agent document analysis system.
Your goal is to analyze the start of a document and the assessment configuration to create a strategic plan for subsequent agents (Extractor, Evaluator, Formatter).

**Assessment Context:**
* **Assessment ID:** {context.assessment_id}
* **Assessment Type:** {assessment_type} ({self.options.get(assessment_type,{}).get('display_name', assessment_type)})
* **Description:** {assessment_description}
* **Primary Target:** Analyze/Extract '{target_name}' ({target_desc})

**Base Instructions for Planner:**
{base_planner_instructions}

**Document Information:**
* Word Count: {document_info.get('word_count', 'N/A')}
* Source Filename: {document_info.get('filename', 'N/A')}

**Document Preview (First {len(document_text_preview)} characters):**
{document_text_preview}


**Your Task:**
Based on the preview, assessment context, and your instructions, generate a JSON object outlining the processing plan. Include the following keys:

1.  `document_type`: (string) Your assessment of the document type (e.g., "Meeting Transcript", "Project Report", "Customer Feedback", "Technical Article", "General Text").
2.  `key_topics_or_sections`: (list of strings) Identify the main topics, sections, or themes apparent from the preview.
3.  `extraction_focus`: (string) Specific guidance for the Extractor agent. What specific types of information, keywords, patterns, or entities related to '{target_name}' should it prioritize finding in the full document? Reference properties from the target definition if applicable (e.g., for action items: description, owner, due_date).
4.  `evaluation_focus`: (string) Guidance for the Evaluator agent. What criteria should be used to assess the extracted information? (e.g., for issues: severity, impact; for action items: clarity, completeness; for readiness: evidence supporting maturity levels). Reference rating scales or classification rules if relevant.
5.  `special_considerations`: (string) Any potential challenges, biases, required sensitivities, or specific formatting notes based on the preview and assessment type (e.g., "Document seems informal", "Contains PII - needs anonymization", "Focus on quantitative data"). Use "None" if no special considerations are apparent.

**Output Format:** Respond *only* with a valid JSON object containing these keys.
"""

            # Define the expected schema for the structured output
            output_schema = {
                "type": "object",
                "properties": {
                    "document_type": {
                        "type": "string",
                        "description": "The assessed type of the document (e.g., Meeting Transcript, Report)."
                    },
                    "key_topics_or_sections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of main topics or sections identified in the preview."
                    },
                    "extraction_focus": {
                        "type": "string",
                        "description": f"Specific guidance for the Extractor agent regarding '{target_name}'."
                    },
                    "evaluation_focus": {
                        "type": "string",
                        "description": "Guidance for the Evaluator agent on assessing extracted information."
                    },
                    "special_considerations": {
                        "type": "string",
                        "description": "Any special handling notes or potential challenges. 'None' if none."
                    }
                },
                "required": [
                    "document_type",
                    "key_topics_or_sections",
                    "extraction_focus",
                    "evaluation_focus",
                    "special_considerations"
                ]
            }

            # --- Use BaseAgent helper for LLM call ---
            planning_result = await self._generate_structured(
                prompt=prompt,
                output_schema=output_schema,
                context=context,
                temperature=self.options.get("planner_temperature", 0.3)
            )

            # --- Post-processing (Example: Add framework info if 'analyze') ---
            if assessment_type == "analyze":
                dimensions = context.get_framework_dimensions()
                if dimensions:
                    planning_result["framework_info"] = {
                        "name": target_definition.get("name"),
                        "dimensions": [{"name": d.get("name"), "criteria": [c.get("name") for c in d.get("criteria", [])]} for d in dimensions]
                    }
                    self._log_debug("Added framework dimension info to planning result.", context)

            self._log_info(f"Planning complete. Assessed document type: {planning_result.get('document_type', 'Unknown')}", context)

            # Store the plan using the enhanced context method
            self._store_data(context, "planning", planning_result)

            return planning_result

        except Exception as e:
            self._log_error(f"PlannerAgent failed: {str(e)}", context, exc_info=True)
            # Re-raise the exception so the orchestrator can mark the stage as failed
            raise