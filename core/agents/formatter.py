# core/agents/formatter.py
import logging
import json
import copy
from typing import Dict, Any, List, Optional

# Import base class and context/LLM types
from .base import BaseAgent
from core.llm.customllm import CustomLLM
from core.models.context import ProcessingContext

class FormatterAgent(BaseAgent):
    """
    Formats the evaluated and aggregated information into the final,
    structured output according to the assessment configuration's output_schema.
    
    Uses the enhanced ProcessingContext for standardized data storage and retrieval.
    """

    def __init__(self, llm: CustomLLM, options: Optional[Dict[str, Any]] = None):
        """Initialize the FormatterAgent."""
        super().__init__(llm, options)
        self.role = "formatter"
        self.name = "FormatterAgent"
        self.logger = logging.getLogger(f"core.agents.{self.name}")
        self.logger.info(f"FormatterAgent initialized.")

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Generates the final structured report/output based on evaluated data
        and the assessment configuration's output_schema.

        Args:
            context: The shared ProcessingContext object containing evaluated data.

        Returns:
            The final formatted output dictionary, conforming to the output_schema.
        """
        self._log_info(f"Starting formatting phase for assessment: '{context.display_name}'", context)

        assessment_type = context.assessment_type
        final_output_dict = {}

        try:
            # --- Get Configuration ---
            output_schema = context.get_output_schema()
            if not output_schema:
                raise ValueError("FormatterAgent cannot proceed: 'output_schema' is missing from the assessment configuration.")
                
            output_format_hints = context.get_output_format_config() or {}
            base_formatter_instructions = context.get_workflow_instructions(self.role) or "Format the evaluated data into the final report structure defined by the output schema."
            
            # --- Get Data using the enhanced context methods ---
            data_type_map = {
                "extract": "action_items",
                "assess": "issues",
                "distill": "key_points",
                "analyze": "evidence"
            }
            
            data_type = data_type_map.get(assessment_type)
            if not data_type:
                self._log_warning(f"Unsupported assessment type: '{assessment_type}'. Will use minimal formatting.", context)
            
            # First try to get evaluated items
            evaluated_items = self._get_data(context, "evaluator", data_type, [])
            
            # If no evaluated items, try aggregated items
            if not evaluated_items:
                self._log_debug(f"No evaluated {data_type} found. Trying aggregated items.", context)
                evaluated_items = self._get_data(context, "aggregator", data_type, [])
                
                # If still nothing, try extracted items
                if not evaluated_items:
                    self._log_debug(f"No aggregated {data_type} found. Trying extracted items.", context)
                    evaluated_items = self._get_data(context, "extractor", data_type, [])
            
            # Get overall assessment if available
            overall_assessment = self._get_data(context, "evaluator", "overall_assessment", {})
            
            # Get planning results for context
            planning_results = self._get_data(context, "planner", "planning", {})
            
            # --- Build Data Summary for Prompt ---
            item_count = len(evaluated_items)
            self._log_info(f"Found {item_count} {data_type} to format.", context)
            
            # Set the maximum number of items to include in prompt
            max_items_in_prompt = self.options.get("formatter_max_items", 50)
            items_for_prompt = evaluated_items[:max_items_in_prompt] 
            
            if len(evaluated_items) > max_items_in_prompt:
                self._log_warning(f"Prompt will include only first {max_items_in_prompt} of {len(evaluated_items)} items due to length constraints.", context)
            
            # Get different fields to include based on assessment type
            self._log_debug("Preparing items for prompt", context)
            
            # Create summarized versions of items for the prompt
            summarized_items = self._prepare_items_for_prompt(items_for_prompt, assessment_type)
            
            # --- Construct Prompt for LLM ---
            prompt = f"""
You are an expert AI 'Formatter' agent. Your task is to synthesize the final evaluated analysis results into a well-structured JSON output that strictly adheres to the provided `FINAL_OUTPUT_SCHEMA`.

**Assessment Context:**
* **Assessment ID:** {context.assessment_id}
* **Assessment Type:** {assessment_type} ({context.display_name})
* **Document:** {context.document_info.get('filename', 'N/A')}

**Input Data:**

1. **Data Summary:**
   * Assessment Type: {assessment_type}
   * Total Items: {item_count} {data_type}
   * Overall Assessment Available: {"Yes" if overall_assessment else "No"}

2. **{data_type.capitalize()} (Sample of up to {max_items_in_prompt}):**
```json
{json.dumps(summarized_items, indent=2, default=str)}
```

3. **Overall Assessment:**
```json
{json.dumps(overall_assessment, indent=2, default=str) if overall_assessment else "N/A"}
```

4. **Planning Analysis:**
```json
{json.dumps(planning_results, indent=2, default=str) if planning_results else "N/A"}
```

**Formatting Guidance:**
* **Base Instructions:** {base_formatter_instructions}
* **Presentation Hints:** {json.dumps(output_format_hints.get('presentation', {}))}
* **Required Sections:** {json.dumps(output_format_hints.get('sections', []))}

**Your Primary Task:**
Generate a JSON object that **exactly matches** the following `FINAL_OUTPUT_SCHEMA`. Use the provided data to populate the fields. Synthesize information where necessary (e.g., for summaries, conclusions). Ensure all required fields in the schema are present in your output.

**FINAL_OUTPUT_SCHEMA:**
```json
{json.dumps(output_schema, indent=2)}
```

Output Format: Respond only with the single, valid JSON object conforming to the FINAL_OUTPUT_SCHEMA. Do not include any text before or after the JSON object.
"""

            # --- LLM Call to generate the final formatted output ---
            final_output_dict = await self._generate_structured(
                prompt=prompt,
                output_schema=output_schema,
                context=context,
                temperature=self.options.get("formatter_temperature", 0.3),
                max_tokens=self.options.get("formatter_max_tokens", 3500)
            )

            # --- Post-processing & Validation ---
            # Basic validation: Check if required keys from schema are present
            missing_keys = []
            if isinstance(output_schema.get("required"), list):
                for key in output_schema["required"]:
                    if key not in final_output_dict:
                        missing_keys.append(key)
            
            if missing_keys:
                self._log_warning(f"LLM output is missing required keys defined in output_schema: {missing_keys}. Output may be incomplete.", context)
                # Add warning to the output itself
                final_output_dict["formatter_warnings"] = final_output_dict.get("formatter_warnings", []) + [f"Output missing required keys: {missing_keys}"]

            # --- Store formatted output using the enhanced context method ---
            self._store_data(context, "formatted", final_output_dict)
            self._log_info("Formatting phase complete. Final structure generated and stored in context.", context)
            
            return final_output_dict

        except Exception as e:
            self._log_error(f"Formatting failed: {str(e)}", context, exc_info=True)
            # Return a minimal error structure
            error_output = {
                "error": f"Formatting Agent Failed: {str(e)}",
                "details": f"Could not generate report based on schema for assessment {context.assessment_id}."
            }
            # Store the error output anyway
            self._store_data(context, "formatted", error_output)
            return error_output
            
    def _prepare_items_for_prompt(self, items: List[Dict[str, Any]], assessment_type: str) -> List[Dict[str, Any]]:
        """Prepare a summarized version of items for the LLM prompt."""
        summarized_items = []
        
        # Define which fields to include based on assessment type
        key_fields = {
            "distill": ["topic", "text", "importance", "evaluated_importance"],
            "extract": ["description", "owner", "due_date", "priority", "evaluated_priority"],
            "assess": ["title", "description", "severity", "evaluated_severity", "potential_impact"],
            "analyze": ["dimension", "criteria", "evidence_text", "maturity_rating", "rating_rationale"]
        }
        
        fields_to_include = key_fields.get(assessment_type, [])
        
        for item in items:
            # Include only the key fields and ensure each has an ID
            summarized_item = {"id": item.get("id", "unknown_id")}
            
            for field in fields_to_include:
                if field in item and item[field] is not None:
                    summarized_item[field] = item[field]
            
            # Add a truncated version of any long text fields
            for field in ["description", "text", "evidence_text"]:
                if field in summarized_item and isinstance(summarized_item[field], str) and len(summarized_item[field]) > 200:
                    summarized_item[field] = summarized_item[field][:200] + "..."
            
            summarized_items.append(summarized_item)
            
        return summarized_items