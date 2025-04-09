import logging
import time
import json
import copy
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

# Import base class and context/LLM types
from .base import BaseAgent
from core.llm.customllm import CustomLLM
from core.models.context import ProcessingContext

logger = logging.getLogger(__name__)

class FormatterAgent(BaseAgent):
    """
    Formats the evaluated and aggregated information into the final,
    structured output according to the assessment configuration's output_schema.
    
    This improved implementation uses a sectional approach to handle large
    datasets without truncation and ensures comprehensive, well-structured 
    outputs even with complex assessment results.
    """

    DEFAULT_FORMATTER_TEMP = 0.5
    DEFAULT_FORMATTER_TOKENS = 4000
    DEFAULT_SECTION_TOKENS = 3000

    def __init__(self, llm: CustomLLM, options: Optional[Dict[str, Any]] = None):
        """Initialize the FormatterAgent."""
        super().__init__(llm, options)
        self.role = "formatter"
        self.logger = logging.getLogger(f"core.agents.{self.__class__.__name__}")
        
        # Get token-based parameters from options
        self.max_items_per_section = self.options.get("max_items_per_section", 30)
        self.token_budget_per_section = self.options.get("token_budget_per_section", self.DEFAULT_SECTION_TOKENS)
        
        self.logger.info(f"FormatterAgent initialized with {self.max_items_per_section} max items per section")

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
        final_output = {}

        try:
            # --- Get Configuration ---
            output_schema = context.get_output_schema()
            if not output_schema:
                raise ValueError("FormatterAgent cannot proceed: 'output_schema' is missing from the assessment configuration.")
                
            output_format = context.get_output_format_config() or {}
            formatter_instructions = context.get_workflow_instructions(self.role) or "Format the evaluated data into the final report structure."

            # --- Get Input Data ---
            # Determine data type based on assessment type
            data_type = self._get_data_type_for_assessment(assessment_type)
            
            # Get evaluated items in priority order: evaluator → aggregator → extractor
            evaluated_items = self._get_data(context, "evaluator", data_type, None)
            if not evaluated_items:
                self._log_debug(f"No evaluated {data_type} found. Trying aggregated items.", context)
                evaluated_items = self._get_data(context, "aggregator", data_type, None)
                
                if not evaluated_items:
                    self._log_debug(f"No aggregated {data_type} found. Trying extracted items.", context)
                    evaluated_items = self._get_data(context, "extractor", data_type, [])
            
            # Ensure items is a list
            if not isinstance(evaluated_items, list):
                self._log_warning(f"Items data for '{data_type}' is not a list. Converting to empty list.", context)
                evaluated_items = []
                
            item_count = len(evaluated_items)
            self._log_info(f"Found {item_count} {data_type} to format.", context)
            
            # Get overall assessment
            overall_assessment = self._get_data(context, "evaluator", "overall_assessment", {})
            
            # Get planning results for context
            planning_results = self._get_data(context, "planner", None, {})
            
            # --- Determine Formatting Approach ---
            # Use sectional approach if we have many items or complex output schema
            use_sectional_approach = (
                item_count > self.max_items_per_section or
                self._is_complex_output_schema(output_schema) or
                self.options.get("force_sectional_approach", False)
            )
            
            if use_sectional_approach:
                self._log_info(f"Using sectional approach for formatting {item_count} items", context)
                final_output = await self._format_using_sectional_approach(
                    context, 
                    evaluated_items, 
                    overall_assessment, 
                    planning_results,
                    assessment_type, 
                    output_schema, 
                    output_format,
                    formatter_instructions
                )
            else:
                self._log_info(f"Using single-pass approach for formatting {item_count} items", context)
                final_output = await self._format_using_single_pass(
                    context, 
                    evaluated_items, 
                    overall_assessment, 
                    planning_results,
                    assessment_type, 
                    output_schema, 
                    output_format,
                    formatter_instructions
                )
            
            # --- Post-Process Output ---
            # Add metadata if needed
            self._enrich_output_metadata(final_output, context)
            
            # Validate against output schema
            self._validate_output_against_schema(final_output, output_schema, context)
            
            # --- Store Results ---
            self._store_data(context, None, final_output)
            self._log_info("Formatting complete. Final output structure generated.", context)
            
            return final_output

        except Exception as e:
            self._log_error(f"Formatting failed: {str(e)}", context, exc_info=True)
            
            # Return a minimal error structure
            error_output = {
                "error": f"Formatting Agent Failed: {str(e)}",
                "details": f"Could not generate report based on schema for assessment {context.assessment_id}."
            }
            # Store the error output anyway
            self._store_data(context, None, error_output)
            return error_output

    def _get_data_type_for_assessment(self, assessment_type: str) -> str:
        """Determine the key/type of data being processed based on assessment type."""
        data_type_map = {
            "extract": "action_items",
            "assess": "issues",
            "distill": "key_points",
            "analyze": "evidence"
        }
        return data_type_map.get(assessment_type, "items")

    def _is_complex_output_schema(self, schema: Dict[str, Any]) -> bool:
        """
        Determine if an output schema is complex enough to warrant sectional formatting.
        A schema is considered complex if it has nested objects, multiple arrays,
        or many required fields.
        """
        # Count number of nested objects and arrays
        nested_count = 0
        array_count = 0
        property_count = 0
        
        def count_complex_elements(obj):
            nonlocal nested_count, array_count, property_count
            
            if not isinstance(obj, dict):
                return
            
            # Count properties at this level
            properties = obj.get("properties", {})
            property_count += len(properties)
            
            # Check each property
            for prop_name, prop_schema in properties.items():
                # Count arrays
                if prop_schema.get("type") == "array":
                    array_count += 1
                    # Check array items
                    items_schema = prop_schema.get("items", {})
                    if isinstance(items_schema, dict) and items_schema.get("type") == "object":
                        nested_count += 1
                        count_complex_elements(items_schema)
                
                # Count objects
                elif prop_schema.get("type") == "object":
                    nested_count += 1
                    count_complex_elements(prop_schema)
        
        # Analyze the schema
        count_complex_elements(schema)
        
        # Decision logic
        required_fields = len(schema.get("required", []))
        
        return (
            nested_count > 2 or  # More than 2 levels of nesting
            array_count > 2 or   # More than 2 arrays
            property_count > 10 or  # More than 10 total properties
            required_fields > 5     # More than 5 required fields
        )

    async def _format_using_single_pass(self,
                                      context: ProcessingContext,
                                      evaluated_items: List[Dict[str, Any]],
                                      overall_assessment: Dict[str, Any],
                                      planning_results: Dict[str, Any],
                                      assessment_type: str,
                                      output_schema: Dict[str, Any],
                                      output_format: Dict[str, Any],
                                      formatter_instructions: str
                                      ) -> Dict[str, Any]:
        """
        Format the complete output in a single LLM pass.
        Used for simpler outputs with fewer items.
        """
        # Prepare a sampled/summarized version of items if there are many
        max_items_in_prompt = self.options.get("max_items_in_prompt", 50)
        if len(evaluated_items) > max_items_in_prompt:
            items_for_prompt = self._get_representative_sample(evaluated_items, assessment_type, max_items_in_prompt)
            self._log_info(f"Using representative sample of {len(items_for_prompt)} items for formatter prompt", context)
        else:
            items_for_prompt = evaluated_items
        
        # Create summarized versions of items for the prompt
        summarized_items = self._prepare_items_for_prompt(items_for_prompt, assessment_type)
        
        # Build the prompt
        prompt = self._build_formatter_prompt(
            summarized_items=summarized_items,
            overall_assessment=overall_assessment,
            planning_results=planning_results,
            assessment_type=assessment_type,
            output_schema=output_schema,
            output_format=output_format,
            formatter_instructions=formatter_instructions,
            total_item_count=len(evaluated_items)
        )
        
        # Call LLM to generate the complete formatted output
        temperature = self.options.get("formatter_temperature", self.DEFAULT_FORMATTER_TEMP)
        max_tokens = self.options.get("formatter_max_tokens", self.DEFAULT_FORMATTER_TOKENS)
        
        formatted_output = await self._generate_structured(
            prompt=prompt,
            output_schema=output_schema,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return formatted_output

    async def _format_using_sectional_approach(self,
                                             context: ProcessingContext,
                                             evaluated_items: List[Dict[str, Any]],
                                             overall_assessment: Dict[str, Any],
                                             planning_results: Dict[str, Any],
                                             assessment_type: str,
                                             output_schema: Dict[str, Any],
                                             output_format: Dict[str, Any],
                                             formatter_instructions: str
                                             ) -> Dict[str, Any]:
        """
        Format the output using a sectional approach, generating different 
        parts of the output with separate LLM calls and then combining them.
        Used for complex outputs with many items.
        """
        # Step 1: Create the output skeleton with overall structure
        self._log_info("Generating output skeleton...", context)
        output_skeleton = await self._generate_output_skeleton(
            context, 
            overall_assessment, 
            planning_results, 
            assessment_type, 
            output_schema, 
            output_format
        )
        
        if not output_skeleton:
            self._log_warning("Failed to generate output skeleton, falling back to empty structure", context)
            output_skeleton = self._create_fallback_skeleton(output_schema)
        
        # Step 2: Identify which sections need to be populated with items
        sections_to_populate = self._identify_item_sections(
            output_skeleton, output_schema, assessment_type
        )
        
        # Step 3: Populate each section
        final_output = copy.deepcopy(output_skeleton)
        section_progress = 0
        total_sections = len(sections_to_populate)
        
        for section_info in sections_to_populate:
            section_path = section_info.get("path", [])
            section_schema = section_info.get("schema", {})
            section_name = ".".join(section_path) or "root"
            
            section_progress += 1
            progress_message = f"Populating section {section_progress}/{total_sections}: {section_name}"
            context.update_stage_progress(section_progress / total_sections, progress_message)
            
            self._log_info(f"Populating section: {section_name}", context)
            
            # Filter items relevant to this section
            relevant_items = self._filter_items_for_section(
                evaluated_items, section_name, assessment_type
            )
            
            if not relevant_items:
                self._log_debug(f"No relevant items found for section: {section_name}", context)
                continue
            
            # Populate the section
            try:
                section_content = await self._populate_section(
                    context,
                    relevant_items,
                    overall_assessment,
                    section_name,
                    section_schema,
                    assessment_type
                )
                
                # Update the section in the final output
                current = final_output
                for i, key in enumerate(section_path):
                    if i == len(section_path) - 1:
                        current[key] = section_content
                    else:
                        if key not in current:
                            current[key] = {}
                        current = current[key]
                        
            except Exception as e:
                self._log_error(f"Error populating section {section_name}: {str(e)}", context, exc_info=True)
                # Continue with other sections
        
        # Step 4: Final integration pass if needed
        if self.options.get("final_integration_pass", False):
            self._log_info("Performing final integration pass...", context)
            try:
                final_output = await self._perform_integration_pass(
                    context, final_output, output_schema, assessment_type
                )
            except Exception as e:
                self._log_error(f"Error in final integration pass: {str(e)}", context, exc_info=True)
                # Continue with current output
        
        return final_output

    async def _generate_output_skeleton(self,
                                      context: ProcessingContext,
                                      overall_assessment: Dict[str, Any],
                                      planning_results: Dict[str, Any],
                                      assessment_type: str,
                                      output_schema: Dict[str, Any],
                                      output_format: Dict[str, Any]
                                      ) -> Dict[str, Any]:
        """
        Generate the overall structure of the output without detailed item lists.
        This creates a skeleton that will be filled in with items later.
        """
        # Create a simplified schema for the skeleton
        skeleton_schema = self._create_skeleton_schema(output_schema)
        
        # Build a prompt specifically for generating the skeleton
        prompt = f"""
You are an expert AI 'Formatter' agent. Your task is to create the overall structure of a report based on assessment results.

**Assessment Context:**
* **Assessment Type:** {assessment_type} ({context.display_name})
* **Document:** {context.document_info.get('filename', 'N/A')}

**Overall Assessment:**
```json
{json.dumps(overall_assessment, indent=2, default=str) if overall_assessment else "N/A"}
```

**Document Analysis:**
```json
{json.dumps(planning_results, indent=2, default=str) if planning_results else "N/A"}
```

**Output Format Configuration:**
```json
{json.dumps(output_format, indent=2, default=str) if output_format else "N/A"}
```

**Your Task:**
Create the overall skeleton/structure for the report WITHOUT including the detailed item lists.
For sections that would contain lists of items (like 'issues', 'action_items', 'key_points'), just use empty arrays as placeholders - these will be filled in later.
Focus on creating a complete structure with:
- Executive summaries
- Overview sections
- Statistics
- Metadata
- Any other non-item-list sections required by the schema

**SKELETON_SCHEMA:**
```json
{json.dumps(skeleton_schema, indent=2)}
```

Output Format: Respond only with the JSON object conforming to the SKELETON_SCHEMA above.
"""

        # Call LLM to generate the skeleton
        temperature = self.options.get("skeleton_temperature", 0.3)  # Lower temperature for more consistent structure
        max_tokens = self.options.get("skeleton_max_tokens", 2000)   # Should need fewer tokens for skeleton
        
        try:
            skeleton = await self._generate_structured(
                prompt=prompt,
                output_schema=skeleton_schema,
                context=context,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return skeleton
        except Exception as e:
            self._log_error(f"Error generating output skeleton: {str(e)}", context, exc_info=True)
            return {}

    def _create_skeleton_schema(self, full_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a simplified version of the output schema for the skeleton generation.
        Removes detailed validation requirements for array items.
        """
        # Base structure - deep copy to avoid modifying the original
        skeleton = copy.deepcopy(full_schema)
        
        # Helper function to simplify array schemas
        def simplify_arrays(schema_obj):
            if not isinstance(schema_obj, dict):
                return schema_obj
            
            # If this is an array property, simplify its items schema
            if schema_obj.get("type") == "array":
                # Replace complex item definitions with empty array placeholders
                schema_obj["items"] = {"type": "object"}
            
            # Process all sub-properties
            for key, value in schema_obj.items():
                if key == "properties" and isinstance(value, dict):
                    for prop_name, prop_schema in value.items():
                        value[prop_name] = simplify_arrays(prop_schema)
                elif isinstance(value, dict):
                    schema_obj[key] = simplify_arrays(value)
                    
            return schema_obj
        
        # Apply simplification
        return simplify_arrays(skeleton)

    def _identify_item_sections(self, 
                              skeleton: Dict[str, Any], 
                              output_schema: Dict[str, Any],
                              assessment_type: str
                              ) -> List[Dict[str, Any]]:
        """
        Identify sections in the output schema that need to be populated with items.
        These are typically array properties that will contain the assessment items.
        
        Returns:
            List of section info dictionaries with path and schema
        """
        sections = []
        
        # Common array field names by assessment type
        array_fields_by_type = {
            "extract": ["action_items"],
            "assess": ["issues"],
            "distill": ["key_points", "topics"],
            "analyze": ["evidence", "dimension_ratings"]
        }
        
        # Default array field names for any type
        default_array_fields = ["items", "findings", "results"]
        
        # Fields to check for this assessment type
        array_fields = array_fields_by_type.get(assessment_type, []) + default_array_fields
        
        # Helper function to find array properties with a path
        def find_array_sections(obj, schema_obj, current_path=None):
            if current_path is None:
                current_path = []
                
            if not isinstance(obj, dict) or not isinstance(schema_obj, dict):
                return
            
            # Look for array properties at this level
            for key, value in obj.items():
                path = current_path + [key]
                
                # Check if this is a known item array
                is_likely_item_array = (
                    key in array_fields or
                    any(key.endswith(suffix) for suffix in ["_items", "_issues", "_points"])
                )
                
                # Check schema to confirm this is an array
                property_schema = None
                if "properties" in schema_obj:
                    property_schema = schema_obj["properties"].get(key, {})
                
                is_array_in_schema = (
                    property_schema and 
                    property_schema.get("type") == "array" and
                    isinstance(property_schema.get("items"), dict)
                )
                
                if isinstance(value, list) and is_array_in_schema and is_likely_item_array:
                    # This is likely an item array section
                    sections.append({
                        "path": path,
                        "schema": property_schema,
                        "items_schema": property_schema.get("items", {})
                    })
                
                # Recurse into nested objects
                elif isinstance(value, dict):
                    # Find corresponding schema for this property
                    next_schema = {}
                    if "properties" in schema_obj and key in schema_obj["properties"]:
                        next_schema = schema_obj["properties"][key]
                    find_array_sections(value, next_schema, path)
        
        # Start the search from the root
        find_array_sections(skeleton, output_schema)
        
        return sections

    def _filter_items_for_section(self, 
                                items: List[Dict[str, Any]], 
                                section_name: str,
                                assessment_type: str
                                ) -> List[Dict[str, Any]]:
        """
        Filter the evaluation items to find those relevant to a specific section.
        Uses rules based on item properties and section name.
        """
        # If no specific filtering rules, return all items
        if not section_name or section_name == "root":
            return items
        
        # Extract section key from path
        section_key = section_name.split('.')[-1]
        
        filtered_items = []
        
        # Filter based on assessment type and section name
        if assessment_type == "assess":
            # For issues assessment
            if "critical" in section_key.lower():
                filtered_items = [i for i in items if i.get("severity", "").lower() == "critical" or 
                                 i.get("evaluated_severity", "").lower() == "critical"]
            elif "high" in section_key.lower():
                filtered_items = [i for i in items if i.get("severity", "").lower() == "high" or 
                                 i.get("evaluated_severity", "").lower() == "high"]
            elif "medium" in section_key.lower():
                filtered_items = [i for i in items if i.get("severity", "").lower() == "medium" or 
                                 i.get("evaluated_severity", "").lower() == "medium"]
            elif "low" in section_key.lower():
                filtered_items = [i for i in items if i.get("severity", "").lower() == "low" or 
                                 i.get("evaluated_severity", "").lower() == "low"]
            else:
                # Default: all issues
                filtered_items = items
                
        elif assessment_type == "extract":
            # For action items assessment
            if "high" in section_key.lower() and "priority" in section_key.lower():
                filtered_items = [i for i in items if i.get("priority", "").lower() == "high" or 
                                 i.get("evaluated_priority", "").lower() == "high"]
            elif "medium" in section_key.lower() and "priority" in section_key.lower():
                filtered_items = [i for i in items if i.get("priority", "").lower() == "medium" or 
                                 i.get("evaluated_priority", "").lower() == "medium"]
            elif "low" in section_key.lower() and "priority" in section_key.lower():
                filtered_items = [i for i in items if i.get("priority", "").lower() == "low" or 
                                 i.get("evaluated_priority", "").lower() == "low"]
            elif "actionable" in section_key.lower():
                filtered_items = [i for i in items if i.get("is_actionable") == True]
            else:
                # Default: all action items
                filtered_items = items
                
        elif assessment_type == "distill":
            # For key points/summaries
            if "high" in section_key.lower() and ("significance" in section_key.lower() or "importance" in section_key.lower()):
                filtered_items = [i for i in items if i.get("importance", "").lower() == "high" or 
                                 i.get("evaluated_significance", "").lower() == "high"]
            elif "topic" in section_key.lower():
                # Group items by topic - this is more complex and would ideally use a topic grouping function
                # For now, return all items as topics will be grouped in the population stage
                filtered_items = items
            else:
                # Default: all key points
                filtered_items = items
        else:
            # For other assessment types, return all items
            filtered_items = items
            
        self._log_debug(f"Filtered {len(items)} items to {len(filtered_items)} for section: {section_name}", None)
        return filtered_items

    async def _populate_section(self,
                              context: ProcessingContext,
                              section_items: List[Dict[str, Any]],
                              overall_assessment: Dict[str, Any],
                              section_name: str,
                              section_schema: Dict[str, Any],
                              assessment_type: str
                              ) -> Any:
        """
        Generate content for a specific section using the relevant items.
        
        Returns:
            Formatted content for the section (array or object depending on schema)
        """
        # Skip if no items and section expects an array
        if not section_items and section_schema.get("type") == "array":
            return []
            
        # Special handling for large item arrays: batched processing
        items_schema = section_schema.get("items", {})
        if section_schema.get("type") == "array" and len(section_items) > self.max_items_per_section:
            return await self._populate_large_item_array(
                context, section_items, section_name, items_schema, assessment_type
            )
            
        # For smaller sections, process in one call
        
        # Prepare a representative sample if needed
        if len(section_items) > self.max_items_per_section:
            items_for_prompt = self._get_representative_sample(
                section_items, assessment_type, self.max_items_per_section
            )
        else:
            items_for_prompt = section_items
            
        # Create summarized items for the prompt
        summarized_items = self._prepare_items_for_prompt(items_for_prompt, assessment_type)
        
        # Build a schema specifically for this section
        if section_schema.get("type") == "array":
            section_output_schema = {
                "type": "array",
                "items": items_schema
            }
        else:
            section_output_schema = section_schema
            
        # Build the prompt for this section
        prompt = f"""
You are an expert AI 'Formatter' agent. Your task is to format content for a specific section of an assessment report.

**Section Context:**
* **Section Name:** {section_name}
* **Assessment Type:** {assessment_type}
* **Total Items for Section:** {len(section_items)}

**Available Items for Formatting ({len(summarized_items)} items):**
```json
{json.dumps(summarized_items, indent=2, default=str)}
```

**Your Task:**
Format these items into the structure required for the '{section_name}' section.
Follow the given schema exactly and ensure all required fields are present.

**SECTION_SCHEMA:**
```json
{json.dumps(section_output_schema, indent=2)}
```

Output Format: Respond only with the JSON content for this section, conforming exactly to the SECTION_SCHEMA above.
"""

        # Call LLM to generate the section content
        temperature = self.options.get("section_temperature", 0.4)
        max_tokens = self.options.get("section_max_tokens", self.DEFAULT_SECTION_TOKENS)
        
        try:
            section_content = await self._generate_structured(
                prompt=prompt,
                output_schema=section_output_schema,
                context=context,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return section_content
        except Exception as e:
            self._log_error(f"Error generating section {section_name}: {str(e)}", context, exc_info=True)
            # Return fallback content based on schema type
            if section_schema.get("type") == "array":
                return []
            return {}

    async def _populate_large_item_array(self,
                                       context: ProcessingContext,
                                       section_items: List[Dict[str, Any]],
                                       section_name: str,
                                       items_schema: Dict[str, Any],
                                       assessment_type: str
                                       ) -> List[Dict[str, Any]]:
        """
        Handle large arrays of items by processing them in batches and then combining.
        Used for sections with many items that won't fit in a single call.
        """
        self._log_info(f"Processing large item array for section {section_name} with {len(section_items)} items", context)
        
        # Calculate number of batches needed
        items_per_batch = self.max_items_per_section
        num_batches = (len(section_items) + items_per_batch - 1) // items_per_batch
        
        # Process in batches
        all_formatted_items = []
        for batch_num in range(num_batches):
            start_idx = batch_num * items_per_batch
            end_idx = min(start_idx + items_per_batch, len(section_items))
            batch_items = section_items[start_idx:end_idx]
            
            self._log_debug(f"Processing batch {batch_num + 1}/{num_batches} with {len(batch_items)} items for section {section_name}", context)
            
            # Create batch-specific schema
            batch_schema = {
                "type": "array",
                "items": items_schema
            }
            
            # Build prompt for this batch
            prompt = f"""
You are an expert AI 'Formatter' agent. Your task is to format a batch of items for a section of an assessment report.

**Batch Context:**
* **Section Name:** {section_name}
* **Assessment Type:** {assessment_type}
* **Batch:** {batch_num + 1} of {num_batches}
* **Items in Batch:** {len(batch_items)}

**Items to Format:**
```json
{json.dumps(self._prepare_items_for_prompt(batch_items, assessment_type), indent=2, default=str)}
```

**Your Task:**
Format these items according to the required schema. Ensure each item is correctly structured.
Do NOT summarize or combine items - format each individual item per the schema.

**ITEM_SCHEMA:**
```json
{json.dumps(items_schema, indent=2)}
```

Output Format: Respond only with a JSON array of formatted items conforming to the ITEM_SCHEMA.
"""

            # Call LLM to format this batch
            temperature = self.options.get("batch_temperature", 0.4)
            max_tokens = self.options.get("batch_max_tokens", self.DEFAULT_SECTION_TOKENS)
            
            try:
                batch_formatted_items = await self._generate_structured(
                    prompt=prompt,
                    output_schema=batch_schema,
                    context=context,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Add to our collection
                if isinstance(batch_formatted_items, list):
                    all_formatted_items.extend(batch_formatted_items)
                else:
                    self._log_warning(f"Expected list response for batch {batch_num + 1}, got {type(batch_formatted_items)}", context)
                    
            except Exception as e:
                self._log_error(f"Error processing batch {batch_num + 1} for section {section_name}: {str(e)}", context, exc_info=True)
                # Continue with other batches
        
        return all_formatted_items

    async def _perform_integration_pass(self,
                                      context: ProcessingContext,
                                      formatted_output: Dict[str, Any],
                                      output_schema: Dict[str, Any],
                                      assessment_type: str
                                      ) -> Dict[str, Any]:
        """
        Perform a final integration pass to ensure consistency across the entire output.
        Used to fix cross-references and ensure overall coherence.
        """
        # Create a simplified version of the formatted output to fit in context
        simplified_output = self._simplify_output_for_prompt(formatted_output, output_schema)
        
        # Build prompt for integration
        prompt = f"""
You are an expert AI 'Formatter' agent. Your task is to perform a final integration of a report.

**Report Context:**
* **Assessment Type:** {assessment_type}
* **Status:** The report structure and sections have been formatted but may need integration.

**Current Report Structure:**
```json
{json.dumps(simplified_output, indent=2, default=str)}
```

**Your Task:**
Review the current report structure for consistency and completeness. Focus on:
1. Ensuring summaries and analysis sections reflect the content of item arrays
2. Fixing any inconsistencies or missing required fields
3. Ensuring cross-references between different sections are accurate

DO NOT modify any item arrays (like issues, action_items, etc.) - only adjust summaries, metadata, and overall structure.

**OUTPUT_SCHEMA:**
```json
{json.dumps(output_schema, indent=2)}
```

Output Format: Respond with the complete integrated report conforming to the OUTPUT_SCHEMA.
"""

        # Call LLM for integration
        temperature = self.options.get("integration_temperature", 0.4)
        max_tokens = self.options.get("integration_max_tokens", self.DEFAULT_FORMATTER_TOKENS)
        
        try:
            integrated_output = await self._generate_structured(
                prompt=prompt,
                output_schema=output_schema,
                context=context,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Preserve the original item arrays in the integrated output
            integrated_output = self._preserve_item_arrays(formatted_output, integrated_output, output_schema)
            
            return integrated_output
        except Exception as e:
            self._log_error(f"Error in integration pass: {str(e)}", context, exc_info=True)
            # Return the original formatted output
            return formatted_output

    def _simplify_output_for_prompt(self, 
                                  output: Dict[str, Any],
                                  schema: Dict[str, Any]
                                  ) -> Dict[str, Any]:
        """
        Create a simplified version of the output for use in prompts.
        Truncates arrays and long text to save tokens.
        """
        if not isinstance(output, dict):
            return output
            
        simplified = {}
        
        for key, value in output.items():
            if isinstance(value, list):
                # For arrays, keep the first few items and indicate count
                if len(value) > 5:
                    if len(value) > 0 and isinstance(value[0], dict):
                        simplified[key] = value[:3]  # Keep first 3 items
                        # Add a count indicator
                        simplified[key].append({
                            "_note": f"[...{len(value) - 3} more items truncated for prompt...]"
                        })
                    else:
                        # For simple value arrays
                        simplified[key] = value[:5]
                        if len(value) > 5:
                            simplified[key].append(f"[...{len(value) - 5} more items...]")
                else:
                    simplified[key] = value
            elif isinstance(value, dict):
                # Recursively simplify nested objects
                simplified[key] = self._simplify_output_for_prompt(value, schema.get("properties", {}).get(key, {}))
            elif isinstance(value, str) and len(value) > 300:
                # Truncate long strings
                simplified[key] = value[:300] + "... [truncated]"
            else:
                simplified[key] = value
                
        return simplified

    def _preserve_item_arrays(self,
                            original_output: Dict[str, Any],
                            integrated_output: Dict[str, Any],
                            schema: Dict[str, Any]
                            ) -> Dict[str, Any]:
        """
        Preserve the original item arrays in the integrated output.
        This prevents the LLM from modifying or summarizing the carefully formatted items.
        """
        if not isinstance(original_output, dict) or not isinstance(integrated_output, dict):
            return integrated_output
            
        result = copy.deepcopy(integrated_output)
        
        def find_and_preserve_arrays(orig, integ, result_obj, current_schema=None):
            if not isinstance(orig, dict) or not isinstance(integ, dict) or not isinstance(result_obj, dict):
                return
                
            if current_schema is None:
                current_schema = {}
                
            # Check all keys in the original output
            for key, value in orig.items():
                # If this is an array in the original output
                if isinstance(value, list) and len(value) > 0:
                    # Check if this property is an array of objects in the schema
                    property_schema = current_schema.get("properties", {}).get(key, {})
                    items_schema = property_schema.get("items", {})
                    
                    if (property_schema.get("type") == "array" and 
                        isinstance(items_schema, dict) and
                        items_schema.get("type") == "object"):
                        # This looks like an item array - preserve it
                        result_obj[key] = value
                        
                # Recursively process nested objects
                elif isinstance(value, dict) and key in integ and isinstance(integ[key], dict):
                    if key not in result_obj:
                        result_obj[key] = {}
                    next_schema = current_schema.get("properties", {}).get(key, {})
                    find_and_preserve_arrays(value, integ[key], result_obj[key], next_schema)
        
        # Start the process from the root
        find_and_preserve_arrays(original_output, integrated_output, result, schema)
        
        return result

    def _build_formatter_prompt(self,
                              summarized_items: List[Dict[str, Any]],
                              overall_assessment: Dict[str, Any],
                              planning_results: Dict[str, Any],
                              assessment_type: str,
                              output_schema: Dict[str, Any],
                              output_format: Dict[str, Any],
                              formatter_instructions: str,
                              total_item_count: int
                              ) -> str:
        """Build the prompt for the formatter LLM call."""
        prompt = f"""
You are an expert AI 'Formatter' agent. Your task is to synthesize the final evaluated analysis results into a well-structured JSON output that strictly adheres to the provided `OUTPUT_SCHEMA`.

**Assessment Context:**
* **Assessment Type:** {assessment_type}
* **Document:** {planning_results.get('document_type', 'Document')}
* **Key Topics:** {', '.join(planning_results.get('key_topics_or_sections', ['N/A']))}

**Input Data:**

1. **Data Summary:**
   * Total Items: {total_item_count} {self._get_data_type_for_assessment(assessment_type)}
   * Sample Size: {len(summarized_items)} items shown below
   * Overall Assessment Available: {"Yes" if overall_assessment else "No"}

2. **Items (Sample of {len(summarized_items)} out of {total_item_count}):**
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
* **Instructions:** {formatter_instructions}
* **Presentation Hints:** {json.dumps(output_format.get('presentation', {}))}
* **Required Sections:** {json.dumps(output_format.get('sections', []))}

**Your Primary Task:**
Generate a JSON object that **exactly matches** the following `OUTPUT_SCHEMA`. Use the provided data to populate the fields. Synthesize information where necessary (e.g., for summaries, conclusions). Ensure all required fields in the schema are present in your output.

**OUTPUT_SCHEMA:**
```json
{json.dumps(output_schema, indent=2)}
```

Output Format: Respond only with the single, valid JSON object conforming to the OUTPUT_SCHEMA. Do not include any text before or after the JSON object.
"""
        return prompt

    def _prepare_items_for_prompt(self, items: List[Dict[str, Any]], assessment_type: str) -> List[Dict[str, Any]]:
        """
        Create a prompt-friendly version of items by removing internal fields
        and truncating long text values to save tokens.
        """
        summarized_items = []
        
        # Define key fields to include based on assessment type
        key_fields = {
            "assess": ["id", "title", "description", "evaluated_severity", "severity", "category", "potential_impact"],
            "extract": ["id", "description", "owner", "due_date", "evaluated_priority", "priority", "is_actionable"],
            "distill": ["id", "text", "topic", "evaluated_significance", "importance", "evaluated_relevance_score"],
            "analyze": ["id", "dimension", "criteria", "evidence_text", "maturity_rating", "evaluator_confidence"]
        }.get(assessment_type, ["id", "text", "type", "evaluated_relevance", "evaluated_importance"])
        
        # Add "rationale" to all assessment types if available
        key_fields.append("rationale")
        
        # Text fields that might need truncation
        text_fields = ["description", "text", "evidence_text", "impact", "rationale", "potential_impact"]
        
        for item in items:
            summary = {}
            
            # Include only the key fields
            for field in key_fields:
                if field in item and item[field] is not None:
                    summary[field] = item[field]
            
            # Truncate any long text fields
            for field in text_fields:
                if field in summary and isinstance(summary[field], str) and len(summary[field]) > 200:
                    summary[field] = summary[field][:200] + "... [truncated]"
            
            # Include evaluation status if available (but not all internal fields)
            if "_evaluation_status" in item:
                summary["_evaluation_status"] = item["_evaluation_status"]
                
            summarized_items.append(summary)
            
        return summarized_items

    def _get_representative_sample(self, 
                                 items: List[Dict[str, Any]], 
                                 assessment_type: str,
                                 max_items: int
                                 ) -> List[Dict[str, Any]]:
        """
        Select a representative sample of items across different levels of importance,
        severity, or other key dimensions based on assessment type.
        """
        if len(items) <= max_items:
            return items
            
        selected_items = []
        
        # 1. Start with any failed items (they're important to show)
        failed_items = [item for item in items if item.get("_evaluation_status") == "failed"]
        # Limit to at most 20% of the sample
        max_failed = max(1, int(max_items * 0.2))
        if len(failed_items) > max_failed:
            failed_items = failed_items[:max_failed]
        selected_items.extend(failed_items)
        
        # 2. Select items based on assessment type
        remaining_slots = max_items - len(selected_items)
        
        if assessment_type == "assess":
            # Group by severity
            severity_levels = ["critical", "high", "medium", "low"]
            
            # Allocate slots proportionally with emphasis on higher severities
            allocation = {
                "critical": int(remaining_slots * 0.4),
                "high": int(remaining_slots * 0.3),
                "medium": int(remaining_slots * 0.2),
                "low": int(remaining_slots * 0.1)
            }
            
            # Ensure at least 1 slot per severity if possible
            for level in allocation:
                if allocation[level] == 0 and remaining_slots > 0:
                    allocation[level] = 1
                    remaining_slots -= 1
                    
            # Select items for each severity
            for severity in severity_levels:
                matching_items = [item for item in items if 
                                 (item.get("severity", "").lower() == severity or 
                                  item.get("evaluated_severity", "").lower() == severity) and
                                 item not in selected_items]
                
                # If more items than slots, spread the selection
                slots = allocation[severity]
                if not slots:
                    continue
                    
                if len(matching_items) > slots:
                    step = max(1, len(matching_items) // slots)
                    indices = [i for i in range(0, len(matching_items), step)][:slots]
                    selected_items.extend([matching_items[i] for i in indices])
                else:
                    selected_items.extend(matching_items[:slots])
                    
        elif assessment_type == "extract":
            # Similar approach but group by priority
            priority_levels = ["high", "medium", "low"]
            
            # Allocate slots
            slots_per_priority = remaining_slots // 3
            extras = remaining_slots % 3
            
            allocation = {
                "high": slots_per_priority + (1 if extras > 0 else 0),
                "medium": slots_per_priority + (1 if extras > 1 else 0),
                "low": slots_per_priority
            }
            
            # Select items for each priority
            for priority in priority_levels:
                matching_items = [item for item in items if 
                                 (item.get("priority", "").lower() == priority or 
                                  item.get("evaluated_priority", "").lower() == priority) and
                                 item not in selected_items]
                
                slots = allocation[priority]
                if len(matching_items) > slots:
                    step = max(1, len(matching_items) // slots)
                    indices = [i for i in range(0, len(matching_items), step)][:slots]
                    selected_items.extend([matching_items[i] for i in indices])
                else:
                    selected_items.extend(matching_items[:slots])
        
        else:
            # For other assessment types, select items spread throughout the list
            remaining_items = [item for item in items if item not in selected_items]
            if remaining_items:
                step = max(1, len(remaining_items) // remaining_slots)
                indices = [i for i in range(0, len(remaining_items), step)][:remaining_slots]
                selected_items.extend([remaining_items[i] for i in indices])
        
        # If we still have space, add random items that weren't selected
        if len(selected_items) < max_items:
            remaining_items = [item for item in items if item not in selected_items]
            additional_needed = min(len(remaining_items), max_items - len(selected_items))
            selected_items.extend(remaining_items[:additional_needed])
        
        return selected_items

    def _create_fallback_skeleton(self, output_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a minimal fallback structure based on the output schema
        in case the skeleton generation fails.
        """
        # Start with an empty dictionary
        skeleton = {}
        
        if not isinstance(output_schema, dict) or "properties" not in output_schema:
            return skeleton
            
        # Add all required fields with placeholder values
        required_fields = output_schema.get("required", [])
        properties = output_schema.get("properties", {})
        
        for field in required_fields:
            if field not in properties:
                continue
                
            prop_schema = properties[field]
            prop_type = prop_schema.get("type")
            
            if prop_type == "object":
                skeleton[field] = self._create_fallback_skeleton(prop_schema)
            elif prop_type == "array":
                skeleton[field] = []
            elif prop_type == "string":
                skeleton[field] = ""
            elif prop_type == "number" or prop_type == "integer":
                skeleton[field] = 0
            elif prop_type == "boolean":
                skeleton[field] = False
            else:
                skeleton[field] = None
                
        return skeleton

    def _enrich_output_metadata(self, output: Dict[str, Any], context: ProcessingContext) -> None:
        """Add or update metadata in the output with current processing info."""
        if not isinstance(output, dict):
            return
            
        # Find metadata field - could be at root or in a nested location
        metadata = None
        
        # Check if metadata exists at root
        if "metadata" in output and isinstance(output["metadata"], dict):
            metadata = output["metadata"]
        
        # If not at root, check for result.metadata pattern
        elif "result" in output and isinstance(output["result"], dict) and "metadata" in output["result"]:
            metadata = output["result"]["metadata"]
            
        # If still not found, try to find it in any top-level object
        if metadata is None:
            for key, value in output.items():
                if isinstance(value, dict) and "metadata" in value and isinstance(value["metadata"], dict):
                    metadata = value["metadata"]
                    break
        
        # If still not found and we need to add metadata, create at root
        if metadata is None and self.options.get("ensure_metadata", True):
            output["metadata"] = {}
            metadata = output["metadata"]
            
        # Exit if no metadata found and we're not supposed to create it
        if metadata is None:
            return
            
        # Update metadata with current info
        metadata.update({
            "document_name": context.document_info.get("filename", "Unknown"),
            "word_count": context.document_info.get("word_count", 0),
            "processing_time": time.time() - context.start_time if hasattr(context, "start_time") else 0,
            "date_analyzed": datetime.now(timezone.utc).isoformat(),
            "assessment_id": context.assessment_id,
            "assessment_type": context.assessment_type,
            "user_options": context.options
        })

    def _validate_output_against_schema(self, 
                                      output: Dict[str, Any], 
                                      schema: Dict[str, Any],
                                      context: ProcessingContext
                                      ) -> bool:
        """
        Perform basic validation of output against the schema.
        Logs warnings for missing required fields.
        
        Returns:
            True if validation passes, False otherwise
        """
        if not isinstance(output, dict) or not isinstance(schema, dict):
            return False
            
        # Check required fields at root level
        required_fields = schema.get("required", [])
        missing_fields = [field for field in required_fields if field not in output]
        
        if missing_fields:
            self._log_warning(f"Output is missing required fields: {missing_fields}", context)
            return False
            
        # Basic type checking for root fields
        properties = schema.get("properties", {})
        for field, value in output.items():
            if field not in properties:
                continue
                
            prop_schema = properties[field]
            prop_type = prop_schema.get("type")
            
            if prop_type == "object" and not isinstance(value, dict):
                self._log_warning(f"Field '{field}' should be an object but is {type(value).__name__}", context)
                return False
            elif prop_type == "array" and not isinstance(value, list):
                self._log_warning(f"Field '{field}' should be an array but is {type(value).__name__}", context)
                return False
            elif prop_type == "string" and not isinstance(value, str):
                self._log_warning(f"Field '{field}' should be a string but is {type(value).__name__}", context)
                return False
                
        return True