# core/agents/aggregator.py
import logging
import json
import copy
from typing import Dict, Any, List, Optional, Tuple

# Import base class and context/LLM types
from .base import BaseAgent
from core.llm.customllm import CustomLLM
from core.models.context import ProcessingContext

logger = logging.getLogger(__name__) # Use module-level logger if preferred

class AggregatorAgent(BaseAgent):
    """
    Aggregates and consolidates information extracted by the ExtractorAgent.
    Uses an LLM to deduplicate and merge similar items based on semantic meaning,
    assessment configuration, and specific merge rules. Preserves evidence links.
    """

    DEFAULT_BATCH_SIZE = 25
    DEFAULT_MAX_INPUT_LEN = 15000
    DEFAULT_AGGREGATOR_TEMP = 0.15
    DEFAULT_AGGREGATOR_TOKENS = 4000

    def __init__(self, llm: CustomLLM, options: Optional[Dict[str, Any]] = None):
        """Initialize the AggregatorAgent."""
        super().__init__(llm, options)
        self.role = "aggregator"
        self.logger = logging.getLogger(f"core.agents.{self.name}") # Agent-specific logger from BaseAgent
        self.logger.info(f"AggregatorAgent initialized with options: {self.options}")

    async def process(self, context: ProcessingContext) -> Optional[Dict[str, Any]]:
        """
        Orchestrates the aggregation process: fetches extracted items, performs
        LLM-based aggregation (potentially in batches), updates evidence links,
        and stores the consolidated results in the context.

        Args:
            context: The shared ProcessingContext object.

        Returns:
            A dictionary summarizing aggregation results (items before/after), or None if skipped/failed.
        """
        self._log_info(f"Starting aggregation phase for assessment: '{context.display_name}'", context)

        assessment_type = context.assessment_type
        summary_stats = {"items_before": 0, "items_after": 0, "batches_processed": 0}

        # --- Determine Data Type ---
        data_type = self._get_data_type_for_assessment(assessment_type)
        if not data_type:
            self._log_warning(f"Unsupported assessment type '{assessment_type}'. Skipping aggregation.", context)
            return None # Cannot proceed without a valid data type

        try:
            # --- Get Data To Aggregate ---
            # Use BaseAgent helper which maps "extractor" role to "extracted" category in context
            items_to_aggregate = self._get_data(context, "extractor", data_type, default=[])

            if not isinstance(items_to_aggregate, list):
                # This check is important as context.get_data could return non-list default
                self._log_error(f"Data received from extractor for '{data_type}' is not a list (Type: {type(items_to_aggregate)}). Cannot aggregate.", context)
                raise TypeError(f"Input data for aggregation ('{data_type}') is not a list.")

            items_before = len(items_to_aggregate)
            summary_stats["items_before"] = items_before
            self._log_info(f"Found {items_before} raw '{data_type}' items from extractor.", context)

            # --- Perform Aggregation ---
            if items_before <= 1:
                self._log_info(f"Skipping LLM aggregation as {items_before} item(s) found.", context)
                aggregated_list = copy.deepcopy(items_to_aggregate) # Pass through original if 0 or 1 item
            else:
                self._log_info(f"Starting LLM-based aggregation for {items_before} items.", context)
                aggregated_list, batches_processed = await self._perform_aggregation(
                    context=context,
                    items=items_to_aggregate,
                    data_type=data_type,
                    assessment_type=assessment_type
                )
                summary_stats["batches_processed"] = batches_processed

            items_after = len(aggregated_list)
            summary_stats["items_after"] = items_after

            # --- Preserve Evidence Links ---
            # This step is crucial for maintaining traceability after merging
            self._update_evidence_links_for_merged_items(context, aggregated_list)

            # --- Store Aggregated Results ---
            # Use BaseAgent helper which maps "aggregator" role to "aggregated" category
            self._store_data(context, data_type, aggregated_list)
            self._log_info(f"Stored {items_after} aggregated '{data_type}' items in context.", context)

            self._log_info(f"Aggregation phase complete. Before: {items_before}, After: {items_after}, Batches: {summary_stats['batches_processed']}", context)
            return summary_stats

        except Exception as e:
            self._log_error(f"Aggregation phase failed: {str(e)}", context, exc_info=True)
            # Re-raise the exception so the orchestrator can mark the stage as failed
            # Consider adding more context to the exception if helpful downstream
            raise RuntimeError(f"Aggregation failed for {data_type}: {e}") from e

    def _get_data_type_for_assessment(self, assessment_type: str) -> Optional[str]:
        """Determine the key/type of data being processed based on assessment type."""
        # Consistent with Extractor/Evaluator logic
        data_type_map = {
            "extract": "action_items",
            "assess": "issues",
            "distill": "key_points",
            "analyze": "evidence"
        }
        return data_type_map.get(assessment_type)

    async def _perform_aggregation(self,
                                   context: ProcessingContext,
                                   items: List[Dict[str, Any]],
                                   data_type: str,
                                   assessment_type: str
                                   ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Manages the aggregation process, deciding whether to use batching.

        Returns:
            Tuple[List of aggregated items, number of batches processed]
        """
        items_count = len(items)
        batch_size = self.options.get("aggregator_batch_size", self.DEFAULT_BATCH_SIZE)
        num_batches = (items_count + batch_size - 1) // batch_size # Ceiling division

        if num_batches <= 1:
            # Process all items in a single batch/call
            self._log_info(f"Processing {items_count} items in a single batch.", context)
            final_results = await self._aggregate_batch(context, items, data_type, assessment_type, batch_num=0, total_batches=1)
            return final_results, 1
        else:
            # Process in multiple batches and then perform a final merge pass
            self._log_info(f"Processing {items_count} items in {num_batches} batches (size: {batch_size}).", context)
            return await self._aggregate_in_batches(context, items, data_type, assessment_type, batch_size, num_batches)

    async def _aggregate_in_batches(self,
                                    context: ProcessingContext,
                                    items: List[Dict[str, Any]],
                                    data_type: str,
                                    assessment_type: str,
                                    batch_size: int,
                                    num_batches: int
                                    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Handles aggregation for large item lists by processing in batches
        and performing a final aggregation pass.

        Returns:
            Tuple[List of final aggregated items, total number of batches processed including final pass]
        """
        all_batch_results = []
        batches_processed_count = 0

        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(items))
            batch_items = items[start_idx:end_idx]

            self._log_info(f"Processing batch {batch_num + 1}/{num_batches} with {len(batch_items)} items.", context)
            batch_result = await self._aggregate_batch(
                context, batch_items, data_type, assessment_type, batch_num, num_batches
            )
            all_batch_results.extend(batch_result)
            batches_processed_count += 1

            # Update context progress more granularly within batch processing
            batch_progress = (batch_num + 1) / (num_batches + 1) # +1 for final pass
            context.update_stage_progress(batch_progress, f"Aggregated batch {batch_num + 1}/{num_batches}")

        # Perform a final aggregation pass on the results from all batches
        if num_batches > 1 and len(all_batch_results) > 1:
            self._log_info(f"Performing final aggregation pass on {len(all_batch_results)} items from {num_batches} batches.", context)
            context.update_stage_progress(num_batches / (num_batches + 1), "Performing final cross-batch aggregation")

            final_results = await self._aggregate_batch(
                context, all_batch_results, data_type, assessment_type, batch_num=-1, total_batches=-1 # Indicate final pass
            )
            batches_processed_count += 1
            return final_results, batches_processed_count
        else:
            # If only one batch or final result is small, no final pass needed
             return all_batch_results, batches_processed_count


    async def _aggregate_batch(self,
                               context: ProcessingContext,
                               batch_items: List[Dict[str, Any]],
                               data_type: str,
                               assessment_type: str,
                               batch_num: int,
                               total_batches: int # Used for logging/progress messages
                               ) -> List[Dict[str, Any]]:
        """
        Performs the core LLM call to aggregate a single batch of items.
        """
        item_count = len(batch_items)
        item_name_singular = data_type.rstrip('s')
        aggregated_list_key = f"aggregated_{data_type}"

        # --- Get Config/Instructions ---
        merge_rules = self._get_merge_rules(assessment_type)
        item_properties_schema = self._get_item_properties_schema(context, assessment_type)
        base_instructions = context.get_workflow_instructions(self.role) or f"Aggregate the provided {data_type} by merging duplicates based on semantic meaning."

        # --- Define LLM Output Schema ---
        aggregated_item_schema = self._define_aggregated_item_schema(item_properties_schema, assessment_type)
        llm_output_schema = {
            "type": "object",
            "properties": {
                aggregated_list_key: {
                    "type": "array",
                    "description": f"Final list of deduplicated and intelligently merged {item_name_singular} items for this batch.",
                    "items": aggregated_item_schema
                }
            },
            "required": [aggregated_list_key]
        }

        # --- Construct Prompt ---
        input_items_json = json.dumps(batch_items, indent=2, default=str) # Use default=str for non-serializable types
        max_len = self.options.get("aggregator_max_input_len", self.DEFAULT_MAX_INPUT_LEN)
        if len(input_items_json) > max_len:
            self._log_warning(f"Input items JSON length ({len(input_items_json)}) exceeds limit ({max_len}). Truncating for prompt.", context)
            input_items_json = input_items_json[:max_len] + "\n...[Input Truncated]...\n]}" # Basic truncation

        batch_info = f"Batch {batch_num + 1}/{total_batches}" if total_batches > 0 else "Final Aggregation Pass"

        prompt = f"""
You are an expert AI 'Aggregator' agent specializing in consolidating extracted information. Your goal is to process the provided list of '{item_name_singular}' items, identify duplicates or items describing the same core concept, and merge them into a single, comprehensive entry.

**Context:** {batch_info}
**Base Instructions:** {base_instructions}
**Merge Rules:**
{merge_rules}

**Input Item Properties Schema (What the extractor likely found):**
```json
{json.dumps(item_properties_schema or {'info':'Schema not determined, rely on item content.'}, indent=2)}
```

**Input Items ({item_count} raw '{item_name_singular}' items for this batch):**
```json
{input_items_json}
```

**Your Task:**
Analyze the input items. Identify duplicates/overlaps based on semantic meaning. Merge them according to the rules. Produce a final, consolidated list of unique, merged {item_name_singular} items **for this batch**.

**Output Format:**
Respond ONLY with a valid JSON object matching this schema. The object MUST contain the key '{aggregated_list_key}' holding the array of aggregated items. Ensure 'merged_item_ids' is accurately populated.
```json
{json.dumps(llm_output_schema, indent=2)}
```
"""

        # --- LLM Call ---
        try:
            max_tokens = self.options.get("aggregator_token_limit", self.DEFAULT_AGGREGATOR_TOKENS)
            temperature = self.options.get("aggregator_temperature", self.DEFAULT_AGGREGATOR_TEMP)

            self._log_debug(f"Calling LLM for aggregation ({batch_info}) with max_tokens={max_tokens}, temp={temperature}", context)

            # Use BaseAgent's structured generation method
            structured_response = await self._generate_structured(
                prompt=prompt,
                output_schema=llm_output_schema,
                context=context, # Pass context for logging and token tracking
                temperature=temperature,
                max_tokens=max_tokens
            )

            # --- Process & Validate LLM Response ---
            aggregated_list = structured_response.get(aggregated_list_key, [])

            if not isinstance(aggregated_list, list):
                self._log_error(f"LLM aggregation response key '{aggregated_list_key}' did not contain a list. Type: {type(aggregated_list)}. Raw: {str(structured_response)[:200]}...", context)
                # Attempt recovery if nested incorrectly
                if isinstance(aggregated_list, dict) and isinstance(aggregated_list.get(aggregated_list_key), list):
                    aggregated_list = aggregated_list[aggregated_list_key]
                    self._log_warning("Recovered list found nested within response.", context)
                else:
                    raise ValueError(f"LLM response for '{aggregated_list_key}' is not a list.")

            validated_items = self._validate_aggregated_items(context, aggregated_list, aggregated_item_schema)

            self._log_info(f"Aggregation successful for batch ({batch_info}). Produced {len(validated_items)} items.", context)
            return validated_items

        except Exception as e:
            self._log_error(f"LLM-based aggregation failed for batch ({batch_info}): {e}", context, exc_info=True)
            # Decide on fallback: return empty list, return original batch items, or raise error?
            # Raising error seems appropriate to signal failure upstream.
            raise RuntimeError(f"Aggregation LLM call failed for batch {batch_info}: {str(e)}") from e


    def _get_merge_rules(self, assessment_type: str) -> str:
        """Defines specific rules for the LLM on how to merge items."""
        rules = [
            "- Merge items that are semantically identical or describe the same core concept/issue/action.",
            "- Intelligently combine descriptions/text from merged items to create a comprehensive summary in the final item.",
            "- Populate 'merged_item_ids' with the original 'id' field from ALL input items combined into the new item.",
            "- Assign 'merge_confidence' (0.0-1.0) indicating certainty in the merge decision (1.0 = highly confident).",
            "- Ensure each output item contains all required properties defined in its schema.",
            "- If unsure about merging two items, keep them separate (assign merge_confidence < 0.5 if forced to merge).",
            "- Preserve key details accurately during merging.",
        ]
        # Add type-specific rules
        if assessment_type == "assess": # Issues
            rules.extend([
                "- When merging severities, use the HIGHEST severity rating among merged items.",
                "- Preserve the most specific category if categories differ.",
                "- Combine potential impact descriptions."
            ])
        elif assessment_type == "extract": # Action Items
            rules.extend([
                "- If due dates differ, preserve the EARLIEST due_date.",
                "- Consolidate owners; list multiple distinct owners if necessary.",
                "- Use the HIGHEST priority among merged items."
            ])
        elif assessment_type == "analyze": # Evidence
             rules.extend([
                 "- Merge evidence snippets ONLY if they refer to the exact same point for the same dimension/criteria.",
                 "- Combine commentary logically."
             ])
        elif assessment_type == "distill": # Key Points
             rules.extend([
                 "- Merge points ONLY if they convey the identical core message.",
                 "- Retain the most representative 'point_type' and 'topic'."
             ])

        return "\n".join(rules)

    def _get_item_properties_schema(self, context: ProcessingContext, assessment_type: str) -> Dict[str, Any]:
        """Determines the expected structure of items *before* aggregation (from Extractor)."""
        # This relies on Extractor producing consistent output based on its own schema generation
        # We can try to get the target definition to infer, or define common structures
        properties = {}
        # Example based on previous ExtractorAgent structure:
        if assessment_type == "extract":
            properties = {"id":{}, "description": {}, "owner": {}, "due_date": {}, "priority": {}}
        elif assessment_type == "assess":
            properties = {"id":{}, "title": {}, "description": {}, "severity": {}, "category": {}, "impact": {}}
        elif assessment_type == "distill":
            properties = {"id":{}, "text": {}, "point_type": {}, "topic": {}, "importance": {}}
        elif assessment_type == "analyze":
            properties = {"id":{}, "dimension": {}, "criteria": {}, "evidence_text": {}, "commentary": {}}
        else:
            properties = {"id":{}, "text": {}, "type": {}} # Generic fallback

        # Basic type assignment (can be refined)
        for key in properties:
            properties[key] = {"type": "string", "description": f"Original {key} from extractor."}
        properties["id"] = {"type": "string", "description": "Unique ID assigned by extractor."}

        return properties


    def _define_aggregated_item_schema(self, input_properties_schema: Dict[str, Any], assessment_type: str) -> Dict[str, Any]:
        """Defines the schema for the items *after* aggregation."""
        schema = {
            "type": "object",
            "properties": {
                **input_properties_schema, # Include original properties
                # Add specific aggregation fields
                "merged_item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Original IDs of all raw items combined into this aggregated item."
                },
                "merge_confidence": {
                    "type": "number",
                    "description": "Confidence score (0.0-1.0) for the merge decision.",
                    "minimum": 0.0, "maximum": 1.0
                }
                # Consider adding 'aggregation_rationale': {'type': 'string'} if needed
            },
            "required": ["merged_item_ids", "merge_confidence"] # Always require these for aggregated items
        }

        # Ensure core fields are required based on type
        required_core_fields = {
            "extract": ["description"],
            "assess": ["title", "description"],
            "distill": ["text"],
            "analyze": ["dimension", "criteria", "evidence_text"]
        }.get(assessment_type, ["id"]) # Fallback requires ID if type unknown

        schema["required"].extend(required_core_fields)
        # Remove duplicates from required list
        schema["required"] = sorted(list(set(schema["required"])))

        return schema


    def _validate_aggregated_items(self, context: ProcessingContext, items: List[Any], item_schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validates items from LLM response against the expected schema."""
        validated_items = []
        required_fields = item_schema.get("required", [])

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                self._log_warning(f"Aggregated item at index {idx} is not a dictionary ({type(item)}), skipping.", context)
                continue

            missing_req = [req for req in required_fields if req not in item or item[req] is None or item[req] == ""]
            if missing_req:
                item_id_log = item.get("id", f"index_{idx}") # Prefer ID if available after merge
                self._log_warning(f"Aggregated item '{item_id_log}' is missing required fields: {missing_req}. Flagging.", context)
                # Add a warning flag to the item itself
                item["_aggregation_warnings"] = item.get("_aggregation_warnings", []) + [f"Missing required fields: {missing_req}"]

            # Ensure merged_item_ids is a list and contains strings
            merged_ids = item.get("merged_item_ids")
            if not isinstance(merged_ids, list) or not all(isinstance(i, str) for i in merged_ids):
                 self._log_warning(f"Aggregated item '{item.get('id', f'index_{idx}')}' has invalid 'merged_item_ids' ({merged_ids}). Flagging.", context)
                 item["_aggregation_warnings"] = item.get("_aggregation_warnings", []) + ["Invalid 'merged_item_ids' format"]
                 # Attempt to fix if possible, e.g., wrap single ID in list
                 if isinstance(merged_ids, str):
                      item["merged_item_ids"] = [merged_ids]


            # TODO: Add more validation? (e.g., type checking, enum checks) - jsonschema library could be used here

            validated_items.append(item)

        return validated_items


    def _update_evidence_links_for_merged_items(self, context: ProcessingContext, aggregated_items: List[Dict[str, Any]]) -> None:
        """
        Ensures evidence associated with original items is linked to the new merged item ID
        by directly manipulating the context's evidence store.
        """
        if not hasattr(context, 'evidence_store') or not isinstance(context.evidence_store.get("references"), dict):
            self._log_debug("Evidence store not found or invalid in context. Skipping evidence link updates.", context)
            return

        items_processed = 0
        links_added = 0
        references = context.evidence_store["references"] # Direct access for modification

        for agg_item in aggregated_items:
            merged_ids = agg_item.get("merged_item_ids")
            # The 'id' of the aggregated item might be one of the merged IDs or newly generated.
            # We need a stable ID for the *aggregated* item itself. Let's assume the LLM might
            # sometimes pick one of the merged IDs as the primary ID, or we assign one.
            # Safest is to use the first merged ID as the representative ID if 'id' field isn't explicitly set by LLM
            # or generate a new one if needed. For simplicity, let's assume the LLM provides a usable ID or we use the first merged ID.

            # Determine the primary ID for the aggregated item
            # Option 1: Use 'id' field if present and valid
            agg_item_id = agg_item.get("id")
            # Option 2: Use first merged ID if 'id' is missing or same as a merged ID
            if not agg_item_id and merged_ids and isinstance(merged_ids, list) and merged_ids:
                agg_item_id = merged_ids[0]
                agg_item["id"] = agg_item_id # Ensure the item has an ID field
            elif not agg_item_id:
                 # Option 3: Generate new ID if needed (less ideal for stability)
                 agg_item_id = f"agg-{uuid.uuid4().hex[:8]}"
                 agg_item["id"] = agg_item_id
                 self._log_debug(f"Generated new ID {agg_item_id} for aggregated item.", context)


            if not agg_item_id or not isinstance(merged_ids, list) or len(merged_ids) <= 1:
                 # Skip if no valid ID, not a list, or only contains itself (not a merge)
                continue

            items_processed += 1
            # Ensure the aggregated item has an entry in the references dict
            if agg_item_id not in references:
                references[agg_item_id] = []

            current_refs_set = {entry.get("ref_id") for entry in references[agg_item_id]}

            # Collect evidence from all original merged items
            for original_id in merged_ids:
                # Skip if trying to merge evidence from the target ID itself into the target ID list
                if original_id == agg_item_id and agg_item_id in references:
                    # Make sure its own initial refs are in the set
                    current_refs_set.update({entry.get("ref_id") for entry in references[original_id]})
                    continue

                original_evidence_entries = references.get(original_id, [])
                for entry in original_evidence_entries:
                    ref_id = entry.get("ref_id")
                    # Add evidence reference if not already linked to the aggregated item
                    if ref_id and ref_id not in current_refs_set:
                        references[agg_item_id].append(entry) # Add the original entry {ref_id, confidence?}
                        current_refs_set.add(ref_id)
                        links_added += 1

                # Optional: Clean up old references if desired (might break other links if items are merged multiple times)
                # if original_id != agg_item_id and original_id in references:
                #     del references[original_id]

        self._log_info(f"Updated evidence links for {items_processed} merged items. Added {links_added} new links.", context)
