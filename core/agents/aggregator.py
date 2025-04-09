import logging
import json
import copy
import uuid
from typing import Dict, Any, List, Optional, Tuple

# Import base class and context/LLM types
from .base import BaseAgent
from core.llm.customllm import CustomLLM
from core.models.context import ProcessingContext

logger = logging.getLogger(__name__)

class AggregatorAgent(BaseAgent):
    """
    Aggregates and consolidates information extracted by the ExtractorAgent.
    Uses token estimation for smarter batching and ensures better preservation
    of context and evidence links across batches.
    
    This improved implementation addresses truncation issues while maintaining
    a simple architecture that doesn't require embeddings or complex clustering.
    """

    DEFAULT_BATCH_SIZE = 20
    DEFAULT_AGGREGATOR_TEMP = 0.15
    DEFAULT_AGGREGATOR_TOKENS = 4000
    # Token estimates based on typical field lengths
    DEFAULT_TOKENS_PER_ITEM = 300

    def __init__(self, llm: CustomLLM, options: Optional[Dict[str, Any]] = None):
        """Initialize the AggregatorAgent."""
        super().__init__(llm, options)
        self.role = "aggregator"
        self.logger = logging.getLogger(f"core.agents.{self.name}")
        
        # Get token-based parameters from options
        self.max_tokens_per_batch = self.options.get("max_tokens_per_batch", 4000)
        self.estimated_tokens_per_item = self.options.get("estimated_tokens_per_item", self.DEFAULT_TOKENS_PER_ITEM)
        
        self.logger.info(f"AggregatorAgent initialized with token_limit: {self.max_tokens_per_batch}")

    async def process(self, context: ProcessingContext) -> Optional[Dict[str, Any]]:
        """
        Orchestrates the aggregation process: fetches extracted items, performs
        token-aware batching, and stores the consolidated results in the context.

        Args:
            context: The shared ProcessingContext object.

        Returns:
            A dictionary summarizing aggregation results, or None if skipped/failed.
        """
        self._log_info(f"Starting aggregation phase for assessment: '{context.display_name}'", context)

        assessment_type = context.assessment_type
        summary_stats = {"items_before": 0, "items_after": 0, "batches_processed": 0}

        # --- Determine Data Type ---
        data_type = self._get_data_type_for_assessment(assessment_type)
        if not data_type:
            self._log_warning(f"Unsupported assessment type '{assessment_type}'. Skipping aggregation.", context)
            return None

        try:
            # --- Get Data To Aggregate ---
            items_to_aggregate = self._get_data(context, "extractor", data_type, default=[])

            if not isinstance(items_to_aggregate, list):
                self._log_error(f"Data received from extractor for '{data_type}' is not a list (Type: {type(items_to_aggregate)}). Cannot aggregate.", context)
                raise TypeError(f"Input data for aggregation ('{data_type}') is not a list.")

            items_before = len(items_to_aggregate)
            summary_stats["items_before"] = items_before
            self._log_info(f"Found {items_before} raw '{data_type}' items from extractor.", context)

            # --- Perform Token-Aware Aggregation ---
            if items_before <= 1:
                self._log_info(f"Skipping aggregation as only {items_before} item(s) found.", context)
                aggregated_list = copy.deepcopy(items_to_aggregate)
            else:
                self._log_info(f"Starting token-aware aggregation for {items_before} items.", context)
                aggregated_list, batches_processed = await self._perform_token_aware_aggregation(
                    context=context,
                    items=items_to_aggregate,
                    data_type=data_type,
                    assessment_type=assessment_type
                )
                summary_stats["batches_processed"] = batches_processed

            items_after = len(aggregated_list)
            summary_stats["items_after"] = items_after

            # --- Preserve Evidence Links ---
            self._update_evidence_links_for_merged_items(context, aggregated_list)

            # --- Store Aggregated Results ---
            self._store_data(context, data_type, aggregated_list)
            self._log_info(f"Stored {items_after} aggregated '{data_type}' items in context.", context)

            return summary_stats

        except Exception as e:
            self._log_error(f"Aggregation phase failed: {str(e)}", context, exc_info=True)
            raise RuntimeError(f"Aggregation failed for {data_type}: {e}") from e

    def _get_data_type_for_assessment(self, assessment_type: str) -> Optional[str]:
        """Determine the key/type of data being processed based on assessment type."""
        data_type_map = {
            "extract": "action_items",
            "assess": "issues",
            "distill": "key_points",
            "analyze": "evidence"
        }
        return data_type_map.get(assessment_type)

    async def _perform_token_aware_aggregation(self,
                                           context: ProcessingContext,
                                           items: List[Dict[str, Any]],
                                           data_type: str,
                                           assessment_type: str
                                           ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Manages the aggregation process using token estimation for batch sizing.
        
        Returns:
            Tuple[List of aggregated items, number of batches processed]
        """
        # Estimate tokens based on item sizes and content length
        total_items = len(items)
        self._log_debug(f"Estimating tokens for {total_items} items", context)
        
        # Use default token estimation or dynamically calculate based on content length
        if self.options.get("dynamic_token_estimation", False):
            estimated_tokens = self._estimate_tokens_for_items(items)
            self._log_debug(f"Dynamic token estimation: ~{estimated_tokens} tokens for {total_items} items", context)
        else:
            estimated_tokens = total_items * self.estimated_tokens_per_item
            self._log_debug(f"Fixed token estimation: ~{estimated_tokens} tokens for {total_items} items", context)
        
        # Calculate batch size based on token estimation
        items_per_batch = max(2, self.max_tokens_per_batch // self.estimated_tokens_per_item)
        
        # If small enough, process in a single batch
        if total_items <= items_per_batch:
            self._log_info(f"Processing all {total_items} items in a single batch", context)
            result = await self._aggregate_batch(context, items, data_type, assessment_type, 0, 1)
            return result, 1
        
        # If we need multiple batches, execute with token-aware batching
        self._log_info(f"Processing {total_items} items using token-aware batching", context)
        return await self._execute_token_aware_batching(
            context, items, data_type, assessment_type, items_per_batch
        )

    async def _execute_token_aware_batching(self,
                                          context: ProcessingContext,
                                          items: List[Dict[str, Any]],
                                          data_type: str,
                                          assessment_type: str,
                                          items_per_batch: int
                                          ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Executes the token-aware batching strategy, with initial batches followed by a final pass.
        
        Returns:
            Tuple[List of final aggregated items, total batches processed]
        """
        total_items = len(items)
        num_batches = (total_items + items_per_batch - 1) // items_per_batch
        
        self._log_info(f"Will process {total_items} items in {num_batches} batches (~{items_per_batch} items per batch)", context)
        
        # First pass: Process items in batches
        all_batch_results = []
        batches_processed = 0
        
        for batch_num in range(num_batches):
            start_idx = batch_num * items_per_batch
            end_idx = min(start_idx + items_per_batch, total_items)
            batch_items = items[start_idx:end_idx]
            
            self._log_info(f"Processing batch {batch_num + 1}/{num_batches} with {len(batch_items)} items", context)
            batch_result = await self._aggregate_batch(
                context, batch_items, data_type, assessment_type, batch_num, num_batches
            )
            all_batch_results.extend(batch_result)
            batches_processed += 1
            
            # Update progress
            progress = (batch_num + 1) / (num_batches + 1)  # +1 for potential final pass
            message = f"Processed batch {batch_num + 1}/{num_batches}"
            context.update_stage_progress(progress, message)
            
        # Check if we need a final pass (if we still have many items)
        # But make the threshold dynamic based on the model's context window
        final_pass_threshold = self.options.get("final_pass_threshold", items_per_batch)
        
        if len(all_batch_results) > final_pass_threshold:
            self._log_info(f"First pass produced {len(all_batch_results)} items - performing final cross-batch pass", context)
            context.update_stage_progress(num_batches/(num_batches + 1), "Performing final cross-batch aggregation")
            
            # Improved final pass: Process strategic groups of items
            final_results = await self._strategic_final_pass(
                context, all_batch_results, data_type, assessment_type
            )
            batches_processed += 1
            return final_results, batches_processed
        else:
            self._log_info(f"First pass produced {len(all_batch_results)} items - no final pass needed", context)
            context.update_stage_progress(1.0, "Aggregation complete")
            return all_batch_results, batches_processed

    async def _strategic_final_pass(self,
                                  context: ProcessingContext,
                                  items: List[Dict[str, Any]],
                                  data_type: str,
                                  assessment_type: str
                                  ) -> List[Dict[str, Any]]:
        """
        Performs a more strategic final pass that groups potentially related items together.
        This provides better context for the LLM when making final merge decisions.
        """
        # For issues/assessments, group by severity and category
        if assessment_type == "assess":
            return await self._final_pass_grouped_by_property(
                context, items, data_type, assessment_type, 
                primary_grouping="severity", secondary_grouping="category"
            )
        # For action items, group by priority and owner
        elif assessment_type == "extract":
            return await self._final_pass_grouped_by_property(
                context, items, data_type, assessment_type,
                primary_grouping="priority", secondary_grouping="owner" 
            )
        # For other types, use a simple batch-based approach
        else:
            # Simple batching for final pass
            items_per_batch = self.max_tokens_per_batch // self.estimated_tokens_per_item
            final_result = []
            
            for i in range(0, len(items), items_per_batch):
                batch = items[i:i+items_per_batch]
                batch_result = await self._aggregate_batch(
                    context, batch, data_type, assessment_type, 
                    batch_num=-1, total_batches=-1  # Indicate final pass
                )
                final_result.extend(batch_result)
                
            return final_result

    async def _final_pass_grouped_by_property(self,
                                            context: ProcessingContext,
                                            items: List[Dict[str, Any]],
                                            data_type: str,
                                            assessment_type: str,
                                            primary_grouping: str,
                                            secondary_grouping: str
                                            ) -> List[Dict[str, Any]]:
        """
        Groups items by primary and secondary properties, then processes each group.
        This ensures related items are processed together for better context.
        """
        # Group items by primary property
        grouped_items = {}
        for item in items:
            primary_value = item.get(primary_grouping, "unknown")
            if primary_value not in grouped_items:
                grouped_items[primary_value] = []
            grouped_items[primary_value].append(item)
        
        # Process each primary group, potentially subdividing by secondary property
        final_results = []
        group_count = len(grouped_items)
        
        for idx, (primary_value, group) in enumerate(grouped_items.items()):
            self._log_info(f"Processing group {idx+1}/{group_count}: {primary_value} ({len(group)} items)", context)
            
            # If group is small enough, process directly
            if len(group) <= self.max_tokens_per_batch // self.estimated_tokens_per_item:
                batch_result = await self._aggregate_batch(
                    context, group, data_type, assessment_type,
                    batch_num=idx, total_batches=group_count
                )
                final_results.extend(batch_result)
                continue
            
            # Otherwise, subdivide by secondary property
            secondary_groups = {}
            for item in group:
                secondary_value = item.get(secondary_grouping, "unknown")
                if secondary_value not in secondary_groups:
                    secondary_groups[secondary_value] = []
                secondary_groups[secondary_value].append(item)
            
            # Process each secondary group
            for secondary_value, subgroup in secondary_groups.items():
                subgroup_result = await self._aggregate_batch(
                    context, subgroup, data_type, assessment_type,
                    batch_num=idx, total_batches=group_count
                )
                final_results.extend(subgroup_result)
        
        return final_results

    def _estimate_tokens_for_items(self, items: List[Dict[str, Any]]) -> int:
        """
        Provide a rough token estimation based on content length.
        This is a simple heuristic - real token count would require tokenization.
        """
        # Sample up to 10 items for estimation
        sample_size = min(10, len(items))
        sample_items = items[:sample_size]
        
        total_chars = 0
        for item in sample_items:
            # Count characters in string fields
            for value in item.values():
                if isinstance(value, str):
                    total_chars += len(value)
        
        # Rough estimate: ~4 characters per token
        avg_chars_per_item = total_chars / sample_size if sample_size > 0 else 0
        avg_tokens_per_item = max(self.estimated_tokens_per_item, avg_chars_per_item / 4)
        
        return int(len(items) * avg_tokens_per_item)

    async def _aggregate_batch(self,
                              context: ProcessingContext,
                              batch_items: List[Dict[str, Any]],
                              data_type: str,
                              assessment_type: str,
                              batch_num: int,
                              total_batches: int
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
        batch_info = f"Batch {batch_num + 1}/{total_batches}" if total_batches > 0 else "Final Aggregation Pass"
        
        # Prepare items JSON with size control
        input_items_json = json.dumps(batch_items, indent=2, default=str)
        max_len = self.options.get("max_input_length", 15000)
        
        if len(input_items_json) > max_len:
            self._log_warning(f"Input JSON length ({len(input_items_json)}) exceeds limit ({max_len}). Using summarized items.", context)
            summarized_items = self._create_summarized_items(batch_items, assessment_type)
            input_items_json = json.dumps(summarized_items, indent=2, default=str)
            
            # If still too long, truncate
            if len(input_items_json) > max_len:
                input_items_json = input_items_json[:max_len] + "\n...[Input Truncated]...\n]}"

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

            self._log_debug(f"Calling LLM for aggregation ({batch_info})", context)

            structured_response = await self._generate_structured(
                prompt=prompt,
                output_schema=llm_output_schema,
                context=context,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # --- Process & Validate LLM Response ---
            aggregated_list = structured_response.get(aggregated_list_key, [])

            if not isinstance(aggregated_list, list):
                self._log_error(f"LLM response key '{aggregated_list_key}' did not contain a list", context)
                # Attempt recovery if nested incorrectly
                if isinstance(aggregated_list, dict) and isinstance(aggregated_list.get(aggregated_list_key), list):
                    aggregated_list = aggregated_list[aggregated_list_key]
                    self._log_warning("Recovered list found nested within response", context)
                else:
                    raise ValueError(f"LLM response for '{aggregated_list_key}' is not a list")

            validated_items = self._validate_aggregated_items(context, aggregated_list, aggregated_item_schema)

            self._log_info(f"Aggregation successful for batch ({batch_info}). Produced {len(validated_items)} items.", context)
            return validated_items

        except Exception as e:
            self._log_error(f"LLM-based aggregation failed for batch ({batch_info}): {e}", context, exc_info=True)
            raise RuntimeError(f"Aggregation LLM call failed for batch {batch_info}: {str(e)}") from e

    def _create_summarized_items(self, items: List[Dict[str, Any]], assessment_type: str) -> List[Dict[str, Any]]:
        """
        Create summarized versions of items for the prompt to reduce token usage.
        Keeps essential fields but truncates long text.
        """
        summarized_items = []
        
        # Define key fields to keep based on assessment type
        key_fields = {
            "assess": ["id", "title", "description", "severity", "category"],
            "extract": ["id", "description", "owner", "due_date", "priority"],
            "distill": ["id", "text", "topic", "importance"],
            "analyze": ["id", "dimension", "criteria", "evidence_text"]
        }
        
        fields_to_keep = key_fields.get(assessment_type, ["id"])
        if "id" not in fields_to_keep:
            fields_to_keep.append("id")  # Always include ID
        
        # Text fields that might need truncation
        text_fields = ["description", "text", "evidence_text", "impact"]
        
        for item in items:
            summary = {field: item.get(field) for field in fields_to_keep if field in item}
            
            # Truncate any long text fields
            for field in text_fields:
                if field in summary and isinstance(summary[field], str) and len(summary[field]) > 200:
                    summary[field] = summary[field][:200] + "... [truncated]"
            
            summarized_items.append(summary)
            
        return summarized_items

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
        """Determines the expected structure of items before aggregation."""
        properties = {}
        
        if assessment_type == "extract":
            properties = {"id":{}, "description": {}, "owner": {}, "due_date": {}, "priority": {}}
        elif assessment_type == "assess":
            properties = {"id":{}, "title": {}, "description": {}, "severity": {}, "category": {}, "impact": {}}
        elif assessment_type == "distill":
            properties = {"id":{}, "text": {}, "point_type": {}, "topic": {}, "importance": {}}
        elif assessment_type == "analyze":
            properties = {"id":{}, "dimension": {}, "criteria": {}, "evidence_text": {}, "commentary": {}}
        else:
            properties = {"id":{}, "text": {}, "type": {}}

        # Basic type assignment
        for key in properties:
            properties[key] = {"type": "string", "description": f"Original {key} from extractor."}
        properties["id"] = {"type": "string", "description": "Unique ID assigned by extractor."}

        return properties

    def _define_aggregated_item_schema(self, input_properties_schema: Dict[str, Any], assessment_type: str) -> Dict[str, Any]:
        """Defines the schema for the items after aggregation."""
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
            },
            "required": ["merged_item_ids", "merge_confidence"]
        }

        # Ensure core fields are required based on type
        required_core_fields = {
            "extract": ["description"],
            "assess": ["title", "description"],
            "distill": ["text"],
            "analyze": ["dimension", "criteria", "evidence_text"]
        }.get(assessment_type, ["id"])

        schema["required"].extend(required_core_fields)
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
                item_id_log = item.get("id", f"index_{idx}")
                self._log_warning(f"Aggregated item '{item_id_log}' is missing required fields: {missing_req}", context)
                item["_aggregation_warnings"] = item.get("_aggregation_warnings", []) + [f"Missing required fields: {missing_req}"]

            # Ensure merged_item_ids is a list and contains strings
            merged_ids = item.get("merged_item_ids")
            if not isinstance(merged_ids, list) or not all(isinstance(i, str) for i in merged_ids):
                 item_id_log = item.get("id", f"index_{idx}")
                 self._log_warning(f"Aggregated item '{item_id_log}' has invalid 'merged_item_ids'", context)
                 item["_aggregation_warnings"] = item.get("_aggregation_warnings", []) + ["Invalid 'merged_item_ids' format"]
                 
                 # Attempt to fix if possible
                 if isinstance(merged_ids, str):
                      item["merged_item_ids"] = [merged_ids]
                 elif merged_ids is None:
                      # Create a default using the item's ID if available
                      item["merged_item_ids"] = [item.get("id", f"unknown_{uuid.uuid4().hex[:8]}")] 

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
        references = context.evidence_store["references"]

        for agg_item in aggregated_items:
            merged_ids = agg_item.get("merged_item_ids")
            
            # Determine the primary ID for the aggregated item
            agg_item_id = agg_item.get("id")
            
            # Use first merged ID if 'id' is missing or same as a merged ID
            if not agg_item_id and merged_ids and isinstance(merged_ids, list) and merged_ids:
                agg_item_id = merged_ids[0]
                agg_item["id"] = agg_item_id
            elif not agg_item_id:
                 # Generate new ID if needed
                 agg_item_id = f"agg-{uuid.uuid4().hex[:8]}"
                 agg_item["id"] = agg_item_id

            if not agg_item_id or not isinstance(merged_ids, list) or len(merged_ids) <= 1:
                continue

            items_processed += 1
            
            # Ensure the aggregated item has an entry in the references dict
            if agg_item_id not in references:
                references[agg_item_id] = []

            current_refs_set = {entry.get("ref_id") for entry in references[agg_item_id]}

            # Collect evidence from all original merged items
            for original_id in merged_ids:
                # Skip if trying to merge from itself
                if original_id == agg_item_id:
                    continue
                    
                original_evidence_entries = references.get(original_id, [])
                for entry in original_evidence_entries:
                    ref_id = entry.get("ref_id")
                    # Add evidence reference if not already linked to the aggregated item
                    if ref_id and ref_id not in current_refs_set:
                        references[agg_item_id].append(entry)
                        current_refs_set.add(ref_id)
                        links_added += 1
                        
        self._log_info(f"Updated evidence links for {items_processed} merged items. Added {links_added} new links.", context)