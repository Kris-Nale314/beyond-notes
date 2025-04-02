# core/agents/aggregator.py
import logging
import json
import copy
from typing import Dict, Any, List, Optional

# Import base class and context/LLM types
from .base import BaseAgent
from core.llm.customllm import CustomLLM
from core.models.context import ProcessingContext

class AggregatorAgent(BaseAgent):
    """
    Aggregates and consolidates information extracted by the ExtractorAgent.
    Uses an LLM to deduplicate and merge similar items (e.g., issues, action items)
    found across different document chunks, based on the assessment configuration.
    
    Uses the enhanced ProcessingContext for standardized data storage and retrieval.
    """

    def __init__(self, llm: CustomLLM, options: Optional[Dict[str, Any]] = None):
        """Initialize the AggregatorAgent."""
        super().__init__(llm, options)
        self.role = "aggregator"
        self.name = "AggregatorAgent"
        self.logger = logging.getLogger(f"core.agents.{self.name}")
        self.logger.info(f"AggregatorAgent initialized.")

    async def process(self, context: ProcessingContext) -> Optional[Dict[str, Any]]:
        """
        Orchestrates the aggregation process based on assessment type.
        Fetches extracted items, calls LLM for aggregation, stores results.

        Args:
            context: The shared ProcessingContext object.

        Returns:
            A dictionary summarizing aggregation results (items before/after), or None.
        """
        self._log_info(f"Starting aggregation phase for assessment: '{context.display_name}'", context)

        assessment_type = context.assessment_type
        summary_stats = {"items_before": 0, "items_after": 0}
        processed = False  # Flag to track if any aggregation was attempted

        # --- Define data types based on assessment type ---
        data_type_map = {
            "extract": "action_items",
            "assess": "issues",
            "distill": "key_points",
            "analyze": "evidence"
        }
        
        data_type = data_type_map.get(assessment_type)
        if not data_type:
            self._log_warning(f"Unsupported assessment type '{assessment_type}'. Skipping aggregation.", context)
            return None

        try:
            # --- Get Data To Aggregate using the enhanced context method ---
            items_to_aggregate = self._get_data(context, "extractor", data_type, [])
            items_before = len(items_to_aggregate)
            summary_stats["items_before"] = items_before

            if not isinstance(items_to_aggregate, list):
                self._log_error(f"Data from extractor is not a list (Type: {type(items_to_aggregate)}). Cannot aggregate.", context)
                raise TypeError(f"Input data for aggregation is not a list.")

            self._log_info(f"Found {items_before} raw {data_type} from extractor.", context)

            # --- Perform Aggregation (if necessary) ---
            if items_before <= 1:
                self._log_info(f"Skipping aggregation as only {items_before} {data_type} found.", context)
                aggregated_list = copy.deepcopy(items_to_aggregate)  # Pass through original if <= 1
            else:
                processed = True  # Aggregation will be attempted
                # Call the enhanced LLM-based aggregation method
                aggregated_list = await self._aggregate_items_llm(
                    context=context,
                    items=items_to_aggregate,
                    data_type=data_type,
                    assessment_type=assessment_type
                )

            items_after = len(aggregated_list)
            summary_stats["items_after"] = items_after

            # --- Preserve Evidence Links During Aggregation ---
            self._update_evidence_links_for_merged_items(context, aggregated_list)

            # --- Store Aggregated Results using the enhanced context method ---
            self._store_data(context, data_type, aggregated_list)
            self._log_info(f"Stored {items_after} aggregated {data_type} in context", context)

            if processed:
                self._log_info(f"Aggregation phase complete. Before: {items_before}, After: {items_after}", context)
            return summary_stats

        except Exception as e:
            self._log_error(f"Aggregation phase failed: {str(e)}", context, exc_info=True)
            # Re-raise the exception so the orchestrator can mark the stage as failed
            raise RuntimeError(f"Aggregation failed: {e}") from e

    def _get_item_properties_schema(self, context: ProcessingContext, assessment_type: str) -> Dict[str, Any]:
        """Helper to determine the expected properties of items being aggregated."""
        target_definition = context.get_target_definition()
        if not target_definition: 
            return {}

        properties = {}
        if assessment_type in ["extract", "assess"]:
            properties = target_definition.get("properties", {})
        elif assessment_type == "distill":
            # Define expected structure for key points from extractor
            properties = {
                "id": {"type": "string", "description": "Unique ID."},
                "point_type": {"type": "string"}, 
                "text": {"type": "string"},
                "topic": {"type": "string"}, 
                "importance": {"type": "string"}
            }
        elif assessment_type == "analyze":
            # Define expected structure for evidence snippets from extractor
            properties = {
                "id": {"type": "string", "description": "Unique ID."},
                "dimension": {"type": "string"}, 
                "criteria": {"type": "string"},
                "evidence_text": {"type": "string"}, 
                "commentary": {"type": "string"}
            }
        
        # Add simple descriptions if missing for prompt clarity
        for key, value in properties.items():
            if isinstance(value, dict) and "description" not in value:
                value["description"] = f"The {key.replace('_', ' ')} of the item."

        return properties

    def _update_evidence_links_for_merged_items(self, context: ProcessingContext, aggregated_items: List[Dict[str, Any]]) -> None:
        """
        Update evidence links for merged items to ensure evidence is preserved.
        
        Args:
            context: The processing context
            aggregated_items: List of aggregated items including merged_item_ids
        """
        # Check if context has evidence store
        if not hasattr(context, 'evidence_store') or not context.evidence_store:
            self._log_debug("No evidence store found in context. Skipping evidence link updates.", context)
            return
            
        for item in aggregated_items:
            merged_ids = item.get("merged_item_ids", [])
            item_id = item.get("id")
            
            if not item_id or not merged_ids:
                continue
                
            # Ensure this item has an entry in the evidence references
            if item_id not in context.evidence_store["references"]:
                context.evidence_store["references"][item_id] = []
                
            # Gather evidence from all merged items
            for merged_id in merged_ids:
                if merged_id == item_id:
                    continue  # Skip self
                    
                # Get evidence for this merged item
                merged_evidence_refs = context.evidence_store["references"].get(merged_id, [])
                
                # Add to the main item's evidence
                for evidence_ref in merged_evidence_refs:
                    # Check if already exists
                    if evidence_ref not in context.evidence_store["references"][item_id]:
                        context.evidence_store["references"][item_id].append(evidence_ref)
                        
        self._log_debug("Updated evidence links for merged items.", context)

    async def _aggregate_items_llm(self,
                               context: ProcessingContext,
                               items: List[Dict[str, Any]],
                               data_type: str,
                               assessment_type: str) -> List[Dict[str, Any]]:
        """Uses an LLM to deduplicate and merge a list of extracted items."""
        items_count = len(items)
        self._log_info(f"Performing LLM-based aggregation for {items_count} {data_type}.", context)

        # --- Get Config Details ---
        target_definition = context.get_target_definition()
        base_aggregator_instructions = context.get_workflow_instructions(self.role) or f"Aggregate the provided {data_type} by merging duplicates."
        item_properties_schema = self._get_item_properties_schema(context, assessment_type)
        
        # --- Define LLM Output Schema ---
        # Get friendly name for better prompting
        item_name = data_type.rstrip('s')  # Remove trailing 's' if present
        
        # Describes the structure of the *aggregated* list items
        aggregated_list_key = f"aggregated_{data_type}"
        aggregated_item_schema = {
            "type": "object",
            "properties": {
                **(item_properties_schema or {}),  # Include original properties if known
                # Add fields to track merging
                "merged_item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Original IDs of all raw items combined into this aggregated item."
                },
                "merge_confidence": {
                    "type": "number",
                    "description": "Confidence level for the merge decision (0.0-1.0).",
                    "minimum": 0,
                    "maximum": 1
                }
            },
            "required": ["merged_item_ids"]  # At minimum require tracking which items were merged
        }
        
        # Add key required fields based on assessment type
        if assessment_type == "extract": 
            aggregated_item_schema["required"].extend(["description", "owner"])
        elif assessment_type == "assess": 
            aggregated_item_schema["required"].extend(["title", "description", "severity"])
        elif assessment_type == "distill": 
            aggregated_item_schema["required"].extend(["text", "topic"])
        elif assessment_type == "analyze": 
            aggregated_item_schema["required"].extend(["dimension", "criteria", "evidence_text"])

        llm_output_schema = {
            "type": "object",
            "properties": {
                aggregated_list_key: {
                    "type": "array",
                    "description": f"Final list of deduplicated and intelligently merged {item_name} items.",
                    "items": aggregated_item_schema
                }
            },
            "required": [aggregated_list_key]
        }

        # --- Construct Prompt ---
        merge_rules = "Merge Rules:\n"
        merge_rules += "- Intelligently combine descriptions/text from merged items to create a comprehensive summary.\n"
        merge_rules += "- Assign a merge_confidence score (0.0-1.0) indicating your confidence in the merge decision.\n"
        
        if assessment_type == "assess": 
            merge_rules += "- Use the HIGHEST severity rating.\n- Preserve the most specific category.\n"
        if assessment_type == "extract": 
            merge_rules += "- Preserve the earliest due_date.\n- Consolidate owners; list multiple if conflicting and necessary.\n"
        
        merge_rules += "- The 'merged_item_ids' list MUST contain the original 'id' field from ALL input items that were combined.\n"
        merge_rules += "- Ensure each output item contains all properties defined in its schema."
        merge_rules += "- If items are semantically identical or very similar (covering the same core idea), they should be merged."
        merge_rules += "- If you're unsure about merging, keep items separate (don't merge) and assign a low merge_confidence."

        # Handle large input lists with batching
        if items_count > 50:
            return await self._aggregate_in_batches(context, items, data_type, assessment_type, llm_output_schema, merge_rules)

        # Prepare items for prompt (limiting size)
        input_items_json = json.dumps(items, indent=2, default=str)
        max_prompt_items_len = self.options.get("aggregator_max_input_len", 10000)
        if len(input_items_json) > max_prompt_items_len:
            self._log_warning(f"Input items JSON length ({len(input_items_json)}) exceeds limit ({max_prompt_items_len}). Truncating for prompt.", context)
            input_items_json = input_items_json[:max_prompt_items_len] + "\n...[Input Truncated]...\n]}"

        prompt = f"""
You are an expert AI 'Aggregator' agent specializing in consolidating extracted information. Your goal is to process the provided list of '{item_name}' items, identify duplicates or items describing the same core concept, and merge them into a single, comprehensive entry.

**Base Instructions:** {base_aggregator_instructions}

**Input Item Schema (Properties the extractor found):**
```json
{json.dumps(item_properties_schema or {'info':'Schema not determined, rely on item content.'}, indent=2)}
```

{merge_rules}

**Input Items (List of {items_count} raw {item_name} items):**
```json
{input_items_json}
```

**Your Task:**
Analyze the input items. Identify duplicates/overlaps based on semantic meaning. Merge them according to the rules. Produce a final, consolidated list of unique, merged {item_name} items.

**Output Format:**
Respond only with a valid JSON object matching this schema:
```json
{json.dumps(llm_output_schema, indent=2)}
```
Ensure the output is a single JSON object with the key '{aggregated_list_key}' containing the array of aggregated items.
Pay close attention to correctly populating 'merged_item_ids' to maintain traceability.
"""

        # --- LLM Call ---
        try:
            # Estimate max tokens needed - can be large if merging complex descriptions
            output_token_estimate = items_count * 150  # Rough estimate per output item
            # Ensure reasonable minimum and apply cap from options
            max_tokens = min(max(2000, output_token_estimate), self.options.get("aggregator_token_limit", 4000))

            self._log_debug(f"Calling LLM for aggregation with max_tokens={max_tokens}", context)

            structured_response = await self._generate_structured(
                prompt=prompt,
                output_schema=llm_output_schema,
                context=context,
                temperature=self.options.get("aggregator_temperature", 0.15),
                max_tokens=max_tokens
            )

            # --- Process LLM Response ---
            aggregated_list = structured_response.get(aggregated_list_key, [])

            # Validate the primary list structure
            if not isinstance(aggregated_list, list):
                self._log_error(f"LLM aggregation response under '{aggregated_list_key}' was not a list. Type: {type(aggregated_list)}", context)
                # Attempt to recover
                if isinstance(aggregated_list, dict) and isinstance(aggregated_list.get(aggregated_list_key), list):
                    aggregated_list = aggregated_list[aggregated_list_key]
                    self._log_warning(f"Recovered list found nested within response dictionary.", context)
                else:
                    # If still not a list, raise error
                    raise ValueError(f"LLM did not return a valid list structure under '{aggregated_list_key}'.")

            # --- Basic Validation of Aggregated Items ---
            valid_items = []
            required_fields = aggregated_item_schema.get("required", [])
            self._log_debug(f"Validating {len(aggregated_list)} aggregated items against required fields: {required_fields}", context)

            for idx, agg_item in enumerate(aggregated_list):
                if not isinstance(agg_item, dict):
                    self._log_warning(f"Aggregated item at index {idx} is not a dictionary, skipping.", context)
                    continue  # Skip non-dict items

                # Check for required fields defined in the schema for each item
                missing_req = [req for req in required_fields if req not in agg_item]
                if missing_req:
                    # Log clearly which item is missing fields
                    item_id_for_log = f"index_{idx}"
                    if "merged_item_ids" in agg_item and agg_item["merged_item_ids"]:
                        item_id_for_log = f"merged:{len(agg_item['merged_item_ids'])} items"
                    
                    self._log_warning(f"Aggregated item ({item_id_for_log}) is missing required fields: {missing_req}. Flagging and adding defaults.", context)
                    agg_item["_validation_warnings"] = f"Missing required fields: {missing_req}"  # Flag item
                    
                    # Try to add defaults for missing fields
                    if "merged_item_ids" in missing_req and "id" in agg_item:
                        # If merged_item_ids missing but id exists, use id as the only merged id
                        agg_item["merged_item_ids"] = [agg_item["id"]]
                
                # Add to valid items list
                valid_items.append(agg_item)

            self._log_info(f"LLM aggregation successful. Produced {len(valid_items)} aggregated {item_name} items.", context)
            return valid_items

        except Exception as e:
            # Catch errors from LLM call or subsequent processing/validation
            self._log_error(f"LLM-based aggregation failed: {e}", context, exc_info=True)
            # Re-raise a specific error type to indicate aggregation failure
            raise RuntimeError(f"Aggregation for {data_type} failed: {str(e)}") from e

    async def _aggregate_in_batches(self, 
                                  context: ProcessingContext,
                                  items: List[Dict[str, Any]],
                                  data_type: str,
                                  assessment_type: str,
                                  llm_output_schema: Dict[str, Any],
                                  merge_rules: str) -> List[Dict[str, Any]]:
        """
        Aggregate a large number of items by processing them in batches.
        
        Args:
            context: The processing context
            items: Items to aggregate
            data_type: Type of data (action_items, issues, etc)
            assessment_type: Type of assessment
            llm_output_schema: Schema for LLM output
            merge_rules: Merge rules string for prompts
            
        Returns:
            List of aggregated items
        """
        self._log_info(f"Using batch processing for {len(items)} items", context)
        
        # Get item key name
        item_name = data_type.rstrip('s')
        aggregated_list_key = f"aggregated_{data_type}"
        
        # Determine batch size based on total items
        items_count = len(items)
        batch_size = 25  # Default batch size
        
        # Calculate number of batches
        num_batches = (items_count + batch_size - 1) // batch_size  # Ceiling division
        
        # Process each batch
        all_aggregated_items = []
        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, items_count)
            batch_items = items[start_idx:end_idx]
            
            self._log_info(f"Processing batch {batch_num+1}/{num_batches} with {len(batch_items)} items", context)
            
            # Prepare batch items for prompt
            batch_items_json = json.dumps(batch_items, indent=2, default=str)
            
            # Construct prompt for this batch
            batch_prompt = f"""
You are an expert AI 'Aggregator' agent specializing in consolidating extracted information. Your goal is to process the provided list of '{item_name}' items, identify duplicates or items describing the same core concept, and merge them into a single, comprehensive entry.

**Base Instructions:** Batch {batch_num+1} of {num_batches}: Aggregate the provided {item_name} items by merging duplicates.

{merge_rules}

**Input Items (Batch {batch_num+1}/{num_batches}, with {len(batch_items)} items):**
```json
{batch_items_json}
```

**Your Task:**
Analyze the input items. Identify duplicates/overlaps based on semantic meaning. Merge them according to the rules. Produce a consolidated list.

**Output Format:**
Respond only with a valid JSON object matching this schema:
```json
{json.dumps(llm_output_schema, indent=2)}
```
Ensure the output is a single JSON object with the key '{aggregated_list_key}' containing the array of aggregated items.
"""

            # Call LLM for this batch
            try:
                # Update progress for this batch
                batch_progress = (batch_num + 0.5) / num_batches
                context.update_stage_progress(batch_progress, f"Aggregating batch {batch_num+1}/{num_batches}")
                
                # Process batch
                structured_response = await self._generate_structured(
                    prompt=batch_prompt,
                    output_schema=llm_output_schema,
                    context=context,
                    temperature=self.options.get("aggregator_temperature", 0.15),
                    max_tokens=4000
                )
                
                # Extract batch results
                batch_results = structured_response.get(aggregated_list_key, [])
                if isinstance(batch_results, list):
                    all_aggregated_items.extend(batch_results)
                    self._log_info(f"Batch {batch_num+1} aggregation successful with {len(batch_results)} items", context)
                else:
                    self._log_warning(f"Batch {batch_num+1} returned invalid result type: {type(batch_results)}", context)
                
            except Exception as e:
                self._log_error(f"Error processing batch {batch_num+1}: {e}", context, exc_info=True)
                # Continue with next batch despite error
                
        # If we have multiple batches, we need a final aggregation pass to merge across batches
        if num_batches > 1 and len(all_aggregated_items) > batch_size:
            self._log_info(f"Performing final aggregation pass across all batches", context)
            
            # Update progress
            context.update_stage_progress(0.9, "Performing final cross-batch aggregation")
            
            # Recursively call this method but with a smaller set
            # The recursive call will use the normal single-batch path
            return await self._aggregate_items_llm(
                context=context,
                items=all_aggregated_items,
                data_type=data_type,
                assessment_type=assessment_type
            )
            
        return all_aggregated_items