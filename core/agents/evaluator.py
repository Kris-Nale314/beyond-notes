import logging
import json
import copy
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

# Import base class and context/LLM types
from .base import BaseAgent
from core.llm.customllm import CustomLLM
from core.models.context import ProcessingContext

logger = logging.getLogger(__name__)

class EvaluatorAgent(BaseAgent):
    """
    Evaluates aggregated items based on assessment goals, criteria, evidence,
    and planner guidance. Uses token-aware batching and enhanced context
    retrieval to improve evaluation quality, especially with truncated content.
    
    This improved implementation addresses evidence context issues while maintaining
    compatibility with the existing agent architecture.
    """

    DEFAULT_BATCH_SIZE = 15  # Smaller default batch size to allow more context per item
    DEFAULT_EVALUATOR_TEMP = 0.2
    DEFAULT_EVALUATOR_TOKENS = 4000
    DEFAULT_OVERALL_ASSESSMENT_TEMP = 0.4
    DEFAULT_OVERALL_ASSESSMENT_TOKENS = 4000
    # Token estimates based on typical field lengths plus evidence
    DEFAULT_TOKENS_PER_ITEM = 500

    def __init__(self, llm: CustomLLM, options: Optional[Dict[str, Any]] = None):
        """Initialize the EvaluatorAgent."""
        super().__init__(llm, options)
        self.role = "evaluator"
        self.logger = logging.getLogger(f"core.agents.{self.name}")
        
        # Get token-based parameters from options
        self.max_tokens_per_batch = self.options.get("max_tokens_per_batch", 4000)
        self.estimated_tokens_per_item = self.options.get("estimated_tokens_per_item", self.DEFAULT_TOKENS_PER_ITEM)
        self.max_evidence_length = self.options.get("max_evidence_length", 1000)  # Increased from 500
        
        self.logger.info(f"EvaluatorAgent initialized with token_limit: {self.max_tokens_per_batch}")

    async def process(self, context: ProcessingContext) -> Optional[Dict[str, Any]]:
        """
        Orchestrates the evaluation: fetches aggregated items, evaluates them
        with token-aware batching, generates an overall assessment, and stores results.

        Args:
            context: The shared ProcessingContext object.

        Returns:
            A dictionary summarizing evaluation results, or None if skipped/failed.
        """
        self._log_info(f"Starting evaluation phase for assessment: '{context.display_name}'", context)

        assessment_type = context.assessment_type
        summary_stats = {"items_evaluated": 0, "items_failed": 0, "batches_processed": 0}

        # --- Determine Data Type ---
        data_type = self._get_data_type_for_assessment(assessment_type)
        if not data_type:
            self._log_warning(f"Unsupported assessment type '{assessment_type}'. Skipping evaluation.", context)
            return None

        try:
            # --- Get Items to Evaluate (preferably from Aggregator) ---
            items_to_evaluate = self._get_data(context, "aggregator", data_type, default=None)

            if items_to_evaluate is None:
                self._log_warning(f"No aggregated '{data_type}' found. Trying extractor output.", context)
                items_to_evaluate = self._get_data(context, "extractor", data_type, default=[])

            if not isinstance(items_to_evaluate, list):
                self._log_error(f"Data for '{data_type}' is not a list (Type: {type(items_to_evaluate)}). Cannot evaluate.", context)
                raise TypeError(f"Input data for evaluation ('{data_type}') is not a list.")

            total_items = len(items_to_evaluate)
            if total_items == 0:
                self._log_info(f"No '{data_type}' items found to evaluate.", context)
                # Ensure empty list is stored
                self._store_data(context, data_type, [])
                return {"items_evaluated": 0, "items_failed": 0}

            self._log_info(f"Starting token-aware evaluation for {total_items} '{data_type}' items.", context)

            # --- Perform Token-Aware Evaluation ---
            evaluated_items, items_failed_count, batches_processed = await self._perform_token_aware_evaluation(
                context=context,
                items=items_to_evaluate,
                data_type=data_type,
                assessment_type=assessment_type
            )
            
            summary_stats["items_evaluated"] = total_items - items_failed_count
            summary_stats["items_failed"] = items_failed_count
            summary_stats["batches_processed"] = batches_processed

            # --- Store Evaluated Items ---
            self._store_data(context, data_type, evaluated_items)
            self._log_info(f"Stored {len(evaluated_items)} evaluated '{data_type}' items in context.", context)

            # --- Generate and Store Overall Assessment ---
            try:
                self._log_info("Generating overall assessment...", context)
                
                overall_assessment = await self._create_overall_assessment(
                    context=context,
                    evaluated_items=evaluated_items,
                    data_type=data_type,
                    assessment_type=assessment_type
                )
                
                # Store overall assessment under a specific key in the "evaluated" category
                self._store_data(context, "overall_assessment", overall_assessment)
                self._log_info("Generated and stored overall assessment.", context)
                
            except Exception as e_overall:
                self._log_error(f"Failed to generate overall assessment: {str(e_overall)}", context, exc_info=True)
                # Add warning to context, but don't fail the whole stage
                context.add_warning(f"Failed to generate overall assessment: {str(e_overall)}", stage="evaluation")

            self._log_info(f"Evaluation phase complete. Evaluated: {summary_stats['items_evaluated']}, Failed: {summary_stats['items_failed']}, Batches: {batches_processed}", context)
            return summary_stats

        except Exception as e:
            self._log_error(f"Evaluation phase failed: {str(e)}", context, exc_info=True)
            raise # Re-raise for orchestrator

    def _get_data_type_for_assessment(self, assessment_type: str) -> Optional[str]:
        """Determine the key/type of data being processed based on assessment type."""
        data_type_map = {
            "extract": "action_items",
            "assess": "issues",
            "distill": "key_points",
            "analyze": "evidence"
        }
        return data_type_map.get(assessment_type)

    async def _perform_token_aware_evaluation(self,
                                           context: ProcessingContext,
                                           items: List[Dict[str, Any]],
                                           data_type: str,
                                           assessment_type: str
                                           ) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Manages the evaluation process using token estimation for batch sizing.
        
        Returns:
            Tuple[List of evaluated items, count of failed items, number of batches processed]
        """
        # Estimate tokens needed per item (including evidence)
        total_items = len(items)
        
        # Estimate tokens with evidence factored in
        # This is important for evaluation since we need evidence context
        tokens_with_evidence = self._estimate_tokens_with_evidence(items, context)
        self._log_debug(f"Estimated ~{tokens_with_evidence} tokens needed for {total_items} items with evidence", context)
        
        # Calculate items per batch based on token estimation
        tokens_per_item = max(self.estimated_tokens_per_item, tokens_with_evidence / max(1, total_items))
        items_per_batch = max(1, int(self.max_tokens_per_batch / tokens_per_item))
        self._log_debug(f"Using ~{tokens_per_item} tokens per item estimate, {items_per_batch} items per batch", context)
        
        # If small enough, process in a single batch
        if total_items <= items_per_batch:
            self._log_info(f"Processing all {total_items} items in a single evaluation batch", context)
            batch_results, failed_count = await self._evaluate_batch(
                context, items, data_type, assessment_type, 0, 1
            )
            return batch_results, failed_count, 1
            
        # For larger sets, use batching
        return await self._execute_token_aware_evaluation_batching(
            context, items, data_type, assessment_type, items_per_batch
        )

    def _estimate_tokens_with_evidence(self, items: List[Dict[str, Any]], context: ProcessingContext) -> int:
        """
        Estimate token count including evidence for a sample of items.
        This helps determine appropriate batch sizes.
        """
        # Sample up to 5 items for estimation
        sample_size = min(5, len(items))
        if sample_size == 0:
            return 0
            
        sample_items = items[:sample_size]
        
        total_chars = 0
        for item in sample_items:
            # Count characters in item fields
            for value in item.values():
                if isinstance(value, str):
                    total_chars += len(value)
            
            # Sample evidence
            item_id = item.get("id")
            if item_id:
                evidence_list = context.get_evidence_for_item(item_id)
                evidence_chars = sum(len(e.get("text", "")) for e in evidence_list[:3])  # Sample first 3 evidence items
                total_chars += evidence_chars
        
        # Rough estimate: ~4 characters per token
        avg_chars_per_item = total_chars / sample_size
        avg_tokens_per_item = avg_chars_per_item / 4
        
        # Use the larger of our calculated value or the default
        return int(len(items) * max(self.estimated_tokens_per_item, avg_tokens_per_item))

    async def _execute_token_aware_evaluation_batching(self,
                                                    context: ProcessingContext,
                                                    items: List[Dict[str, Any]],
                                                    data_type: str,
                                                    assessment_type: str,
                                                    items_per_batch: int
                                                    ) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Executes the token-aware batching strategy for evaluation.
        
        Returns:
            Tuple[List of evaluated items, count of failed items, number of batches processed]
        """
        total_items = len(items)
        num_batches = (total_items + items_per_batch - 1) // items_per_batch
        
        self._log_info(f"Will evaluate {total_items} items in {num_batches} batches (~{items_per_batch} items per batch)", context)
        
        all_evaluated_items = []
        total_failed_count = 0
        batches_processed = 0
        
        # Process in batches
        for batch_num in range(num_batches):
            start_idx = batch_num * items_per_batch
            end_idx = min(start_idx + items_per_batch, total_items)
            batch_items = items[start_idx:end_idx]
            
            self._log_info(f"Evaluating batch {batch_num + 1}/{num_batches} with {len(batch_items)} items", context)
            
            try:
                batch_results, failed_count = await self._evaluate_batch(
                    context, batch_items, data_type, assessment_type, batch_num, num_batches
                )
                all_evaluated_items.extend(batch_results)
                total_failed_count += failed_count
                batches_processed += 1
                
                # Update progress
                progress = (batch_num + 1) / num_batches
                message = f"Evaluated batch {batch_num + 1}/{num_batches}"
                context.update_stage_progress(progress, message)
                
            except Exception as e:
                self._log_error(f"Error processing batch {batch_num + 1}: {str(e)}", context, exc_info=True)
                # Mark all items in this batch as failed
                batch_failed_items = []
                for item in batch_items:
                    failed_item = copy.deepcopy(item)
                    failed_item["_evaluation_status"] = "failed"
                    failed_item["_evaluation_error"] = f"Batch processing error: {str(e)}"
                    batch_failed_items.append(failed_item)
                
                all_evaluated_items.extend(batch_failed_items)
                total_failed_count += len(batch_items)
                batches_processed += 1
                
                # Continue with next batch despite the error
        
        self._log_info(f"Completed evaluation of {total_items} items. Failed: {total_failed_count}", context)
        return all_evaluated_items, total_failed_count, batches_processed
        
    async def _evaluate_batch(self,
                             context: ProcessingContext,
                             batch_items: List[Dict[str, Any]],
                             data_type: str,
                             assessment_type: str,
                             batch_num: int,
                             total_batches: int
                             ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Performs the core LLM call to evaluate a single batch of items.
        Uses enhanced evidence retrieval for better context.
        
        Returns:
            Tuple[List of processed items for this batch (evaluated or original+error), count of items failed in batch]
        """
        processed_batch_items = []  # Will hold items (original structure updated with evaluation)
        failed_count = 0
        item_name_singular = data_type.rstrip('s')

        # --- Get Evaluation Details ---
        # Determine the specific task and the schema for fields to be added/updated
        evaluation_task, item_update_schema = self._get_evaluation_details(assessment_type, context)
        if not evaluation_task:
            self._log_warning(f"No evaluation task defined for assessment type '{assessment_type}'", context)
            evaluation_task = f"Evaluate each {item_name_singular} item for relevance and importance."

        # --- Define LLM Output Schema ---
        # This schema describes what the LLM should return for each evaluated item
        llm_item_output_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "The unique ID of the item being evaluated."},
                **item_update_schema.get("properties", {})  # Evaluation fields to add/update
            },
            "required": ["id"] + item_update_schema.get("required", [])
        }
        
        # The overall schema expects a list of these items
        llm_batch_output_schema = {
            "type": "object",
            "properties": {
                "evaluated_items": {
                    "type": "array",
                    "description": f"List of evaluated {item_name_singular} items for this batch, containing ID and evaluation fields.",
                    "items": llm_item_output_schema
                }
            },
            "required": ["evaluated_items"]
        }

        # --- Construct Prompt ---
        prompt_items_section = self._build_evaluation_items_prompt(context, batch_items)
        if not prompt_items_section:
            self._log_warning("No valid items found in batch to construct prompt.", context)
            return [], 0
            
        # Get planner guidance
        planning_output = self._get_data(context, "planner", default={})
        planner_guidance = planning_output.get('evaluation_focus', f'Evaluate based on standard criteria for {item_name_singular}.')
        base_instructions = context.get_workflow_instructions(self.role) or "Evaluate each item according to the specific task based on its details and evidence."

        batch_info = f"Batch {batch_num + 1}/{total_batches}" if total_batches > 0 else "Single Batch Evaluation"
        items_in_prompt_count = len(batch_items)

        prompt = f"""
You are an AI 'Evaluator' agent. Your task is to assess a batch of '{item_name_singular}' items based on the assessment goal, specific instructions, item details, and associated evidence.

**Context:** {batch_info}
**Assessment Type:** {assessment_type}
**Planner's Guidance for Evaluation:** {planner_guidance}
**Base Instructions:** {base_instructions}
**Specific Evaluation Task for Each Item:** {evaluation_task}

**Items to Evaluate in this Batch ({items_in_prompt_count} items):**
{prompt_items_section}

**Instructions:**
1. For EACH item listed above (identified by 'Item ID'):
2. Review its details and associated evidence. Pay careful attention to the evidence which provides critical context.
3. Apply the 'Specific Evaluation Task' to determine the values for the required evaluation fields.
4. Generate a response containing ONLY the evaluated fields and the original 'id' for EACH item.
5. Ensure your evaluation is evidence-based and considers the full context available.

**Output Format:** Respond ONLY with a valid JSON object matching this schema. The object MUST contain the key 'evaluated_items', holding a list where each entry corresponds to an evaluated item and contains its 'id' and the requested evaluation fields.
```json
{json.dumps(llm_batch_output_schema, indent=2)}
```
"""

        # --- LLM Call ---
        try:
            max_tokens = self.options.get("evaluator_token_limit", self.DEFAULT_EVALUATOR_TOKENS)
            temperature = self.options.get("evaluator_temperature", self.DEFAULT_EVALUATOR_TEMP)

            structured_response = await self._generate_structured(
                prompt=prompt,
                output_schema=llm_batch_output_schema,
                context=context,
                temperature=temperature,
                max_tokens=max_tokens
            )

            llm_evaluated_items_list = structured_response.get("evaluated_items", [])

            # --- Process & Merge LLM Response ---
            if not isinstance(llm_evaluated_items_list, list):
                self._log_error(f"LLM evaluation response key 'evaluated_items' did not contain a list. Type: {type(llm_evaluated_items_list)}.", context)
                raise ValueError("LLM evaluation response format is invalid.")

            # Create a map of evaluation results by ID for efficient merging
            evaluation_results_map = {res.get("id"): res for res in llm_evaluated_items_list if isinstance(res, dict) and "id" in res}

            # Iterate through the ORIGINAL batch items and update them
            for original_item in batch_items:
                item_id = original_item.get("id")
                processed_item = copy.deepcopy(original_item)  # Start with a copy of the original

                if item_id and item_id in evaluation_results_map:
                    # Merge evaluation fields from LLM result into the item
                    evaluation_data = evaluation_results_map[item_id]
                    # Only update with keys present in the evaluation schema (plus 'id')
                    fields_to_update = list(item_update_schema.get("properties", {}).keys())

                    for key in fields_to_update:
                        if key in evaluation_data:
                            processed_item[key] = evaluation_data[key]

                    # Add a timestamp or flag
                    processed_item["_evaluation_status"] = "success"
                    processed_item["_evaluated_at"] = datetime.now(timezone.utc).isoformat()

                else:
                    # Item evaluation failed or missing from response
                    self._log_warning(f"Evaluation data not found for item ID '{item_id}' in LLM response.", context)
                    processed_item["_evaluation_status"] = "failed"
                    processed_item["_evaluation_error"] = "Evaluation data missing in LLM response."
                    failed_count += 1

                processed_batch_items.append(processed_item)

            self._log_info(f"Evaluation successful for batch ({batch_info}). {len(processed_batch_items) - failed_count} items evaluated, {failed_count} failed.", context)
            return processed_batch_items, failed_count

        except Exception as e:
            self._log_error(f"Evaluation failed for batch ({batch_info}): {e}", context, exc_info=True)
            # If batch fails, mark all items in this batch as failed and return them
            failed_count = len(batch_items)
            processed_batch_items = []
            for item in batch_items:
                processed_item = copy.deepcopy(item)
                processed_item["_evaluation_status"] = "failed"
                processed_item["_evaluation_error"] = f"Batch processing error: {str(e)}"
                processed_batch_items.append(processed_item)

            return processed_batch_items, failed_count

    def _build_evaluation_items_prompt(self, context: ProcessingContext, batch_items: List[Dict[str, Any]]) -> str:
        """
        Build the item details section of the evaluation prompt with enhanced evidence.
        """
        prompt_items_section = ""
        items_in_prompt_count = 0
        MAX_EVIDENCE_LENGTH = self.max_evidence_length

        for item in batch_items:
            item_id = item.get("id", f"temp_id_{uuid.uuid4().hex[:4]}")  # Ensure ID exists
            
            # Create a simplified view of the item for the prompt
            simplified_item = self._create_simplified_item(item)
            item_details_json = json.dumps(simplified_item, indent=2, default=str)
            
            # Get enhanced evidence with broader context
            evidence_text = self._get_enhanced_evidence_text(context, item, max_length=MAX_EVIDENCE_LENGTH)

            prompt_items_section += f"\n--- Item ID: {item_id} ---\n"
            prompt_items_section += f"Item Details:\n```json\n{item_details_json}\n```\n"
            prompt_items_section += f"Associated Evidence:\n{evidence_text}\n"
            prompt_items_section += "------\n"
            items_in_prompt_count += 1

        return prompt_items_section

    def _create_simplified_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a simplified version of an item for the prompt, removing internal fields
        and truncating long text to save tokens.
        """
        # Skip internal fields and metadata
        internal_prefixes = ["_", "merge_", "merged_"]
        simplified = {}
        
        for key, value in item.items():
            # Skip internal fields and metadata
            if any(key.startswith(prefix) for prefix in internal_prefixes):
                continue
                
            # Include the field
            if isinstance(value, str) and len(value) > 300:
                # Truncate long text fields
                simplified[key] = value[:300] + "... [truncated]"
            else:
                simplified[key] = value
                
        return simplified

    def _get_enhanced_evidence_text(self, context: ProcessingContext, item: Dict[str, Any], max_length: int = 1000) -> str:
        """
        Retrieve evidence text with enhanced context from surrounding chunks.
        This provides better context for evaluation than simple evidence retrieval.
        """
        evidence_texts = []
        total_len = 0
        item_id = item.get("id")
        chunk_index = item.get("chunk_index")  # Original chunk where item was found

        if not item_id:
            return "Item ID missing, cannot retrieve evidence."

        # Get evidence using the context method
        evidence_list = context.get_evidence_for_item(item_id)
        if not evidence_list:
            # Fallback if no linked evidence
            for field in ["description", "text", "evidence_text", "title"]:
                if field in item and item[field] and isinstance(item[field], str):
                    return f"Item Content Fallback: {item[field][:max_length]}"
            return "No evidence available for this item."

        # Process evidence list
        processed_chunks = set()  # Track which chunks we've already seen
        
        # First add direct evidence
        for evidence in evidence_list:
            evidence_text = evidence.get("text", "")
            if evidence_text:
                evidence_chunk = evidence.get("metadata", {}).get("chunk_index", "?")
                if evidence_chunk != "?":
                    processed_chunks.add(evidence_chunk)  # Track this chunk
                    
                confidence = evidence.get("confidence")
                conf_str = f" (Conf: {confidence:.2f})" if confidence is not None else ""
                snippet = f"- Src(Chunk {evidence_chunk}{conf_str}): {evidence_text}"

                # Add if we have space
                if total_len + len(snippet) < max_length:
                    evidence_texts.append(snippet)
                    total_len += len(snippet)
                else:
                    evidence_texts.append("- [...more evidence truncated...]")
                    total_len = max_length  # Mark as full
                    break
        
        # If we have a chunk_index but it wasn't in our evidence, try to add context from that chunk
        if chunk_index is not None and chunk_index not in processed_chunks and total_len < max_length:
            chunk = next((c for c in context.chunks if c.get("chunk_index") == chunk_index), None)
            if chunk and "text" in chunk:
                remaining_space = max_length - total_len
                if remaining_space > 100:  # Only if meaningful space is left
                    context_snippet = chunk["text"][:min(200, remaining_space)]  # Limit to 200 chars max
                    context_text = f"\nBroader Context from Chunk {chunk_index}:\n{context_snippet}"
                    evidence_texts.append(context_text)
                    total_len += len(context_text)
                    processed_chunks.add(chunk_index)
        
        # Try to add adjacent chunks for additional context if space permits
        if chunk_index is not None and total_len < max_length:
            for adj_idx in [chunk_index - 1, chunk_index + 1]:  # Try previous and next chunk
                if adj_idx >= 0 and adj_idx not in processed_chunks:  # Skip negative indices and already processed
                    chunk = next((c for c in context.chunks if c.get("chunk_index") == adj_idx), None)
                    if chunk and "text" in chunk:
                        remaining_space = max_length - total_len
                        if remaining_space > 100:  # Only if meaningful space is left
                            context_snippet = chunk["text"][:min(150, remaining_space)]  # Smaller snippet for adjacent
                            context_text = f"\nAdjacent Context (Chunk {adj_idx}):\n{context_snippet}"
                            evidence_texts.append(context_text)
                            total_len += len(context_text)
                            processed_chunks.add(adj_idx)
                            if total_len >= max_length:
                                break

        return "\n".join(evidence_texts) if evidence_texts else "No specific evidence text retrieved."

    def _get_evaluation_details(self, assessment_type: str, context: ProcessingContext) -> Tuple[str, Dict[str, Any]]:
        """
        Define the evaluation task and the schema for the fields to be added/updated
        by the evaluator based on the assessment type.
        """
        task = ""
        # Schema defines ONLY the fields the evaluator should ADD or UPDATE
        schema = {"type": "object", "properties": {}, "required": []}
        target_definition = context.get_target_definition() or {}

        base_rationale = {"rationale": {"type": "string", "description": "Brief explanation justifying the evaluation ratings/assignments."}}
        schema["properties"].update(base_rationale)
        schema["required"].append("rationale")

        if assessment_type == "extract": # Action Items
            task = "Evaluate clarity, assign priority (High, Medium, Low), assess feasibility (High, Medium, Low), and check actionability."
            schema["properties"].update({
                "evaluated_clarity": {"type": "string", "description": "Assessment of clarity (e.g., Clear, Needs Refinement)."},
                "evaluated_priority": {"type": "string", "enum": ["High", "Medium", "Low"], "description": "Assessed priority."},
                "evaluated_feasibility": {"type": "string", "enum": ["High", "Medium", "Low"], "description": "Estimated feasibility."},
                "is_actionable": {"type": "boolean", "description": "Is this item clear, specific, and actionable?"}
            })
            schema["required"].extend(["evaluated_priority", "is_actionable"])

        elif assessment_type == "assess": # Issues / Risks
            task = "Evaluate severity (Critical, High, Medium, Low), potential impact, and suggest brief recommendations. Validate initially extracted severity if present."
            schema["properties"].update({
                "evaluated_severity": {"type": "string", "enum": ["Critical", "High", "Medium", "Low"], "description": "Re-assessed severity based on evidence and impact."},
                "potential_impact": {"type": "string", "description": "Summarized potential impact if not addressed."},
                "suggested_recommendations": {"type": "array", "items": {"type": "string"}, "description": "1-2 brief suggestions for addressing the issue."}
            })
            schema["required"].extend(["evaluated_severity", "potential_impact"])

        elif assessment_type == "distill": # Key Points
            task = "Evaluate significance (High, Medium, Low) and relevance score (0.0-1.0) to the main topic. Verify the initially assigned importance if present."
            schema["properties"].update({
                "evaluated_significance": {"type": "string", "enum": ["High", "Medium", "Low"], "description": "Assessed significance of the point."},
                "evaluated_relevance_score": {"type": "number", "minimum":0.0, "maximum":1.0, "description": "Score indicating relevance to the overall document topic."}
            })
            schema["required"].extend(["evaluated_significance", "evaluated_relevance_score"])

        elif assessment_type == "analyze": # Framework Evidence
            rating_scale = context.get_rating_scale()
            if rating_scale and rating_scale.get("levels"):
                 # Dynamically build enum from rating scale config
                 level_values = [level.get("value") for level in rating_scale["levels"] if level.get("value") is not None]
                 # Determine type (integer or number)
                 value_type = "integer" if all(isinstance(v, int) for v in level_values) else "number"

                 task = f"Evaluate quality/relevance of evidence for the framework dimension/criteria. Assign a maturity rating using the scale ({min(level_values)}-{max(level_values)}) and confidence (0.0-1.0)."
                 schema["properties"].update({
                    "maturity_rating": {
                        "type": value_type,
                        "enum": level_values,
                        "description": f"Maturity rating based on this evidence. Scale: {json.dumps(rating_scale['levels'])}"
                    },
                    "evaluator_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Evaluator's confidence (0.0-1.0) in this rating based *only* on this specific evidence."}
                    # Rationale field covers rating rationale
                 })
                 schema["required"].extend(["maturity_rating", "evaluator_confidence"])
            else:
                 task = "Evaluate relevance and strength of evidence (1-5 scale)."
                 schema["properties"].update({
                      "evidence_strength": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Assessed strength/relevance of evidence (1=Weak, 5=Strong)."}
                 })
                 schema["required"].extend(["evidence_strength"])


        else: # Default/Unknown
            task = "Assess the overall relevance and importance of each item."
            schema["properties"].update({
                 "evaluated_relevance": {"type": "number", "minimum":0.0, "maximum":1.0, "description": "Score from 0.0 to 1.0 indicating relevance."},
                 "evaluated_importance": {"type": "string", "enum": ["High", "Medium", "Low"], "description": "Assessed importance."}
            })
            schema["required"].extend(["evaluated_relevance", "evaluated_importance"])


        return task, schema

    async def _create_overall_assessment(self,
                                        context: ProcessingContext,
                                        evaluated_items: List[Dict[str, Any]],
                                        data_type: str,
                                        assessment_type: str
                                        ) -> Dict[str, Any]:
        """
        Generates an overall assessment summary using the LLM, based on evaluated items.
        Uses strategic sampling of items to ensure representation across importance levels.
        """
        if not evaluated_items:
            self._log_warning("Cannot create overall assessment: No evaluated items provided.", context)
            return {"summary": "No items were available for overall assessment.", "status":"skipped"}

        self._log_info(f"Generating overall assessment based on {len(evaluated_items)} evaluated '{data_type}'.", context)

        # --- Prepare Data for Prompt ---
        # Create a strategic sampling of items
        sample_items = self._create_strategic_sample(evaluated_items, assessment_type, max_items=25)
        
        # Calculate statistics on the complete set
        stats = self._calculate_item_statistics(evaluated_items, assessment_type)

        # --- Define Output Schema ---
        overall_schema = {
            "type": "object",
            "properties": {
                "executive_summary": {"type": "string", "description": "High-level overview (3-5 sentences) of the main findings and overall status."},
                "key_findings": {"type": "array", "items": {"type": "string"}, "description": "3-5 bullet points highlighting the most critical evaluated items or trends."},
                "thematic_analysis": {"type": "string", "description": "Analysis of common themes, patterns, or critical areas observed across the evaluated items."},
                "strategic_recommendations": {"type": "array", "items": {"type": "string"}, "description": "3-5 actionable, high-level recommendations based on the overall evaluation."},
                "overall_status_rating": {"type": "string", "description": "A single qualitative rating summarizing the overall state (e.g., Critical, Warning, Fair, Good, Excellent)"}
            },
            "required": ["executive_summary", "key_findings", "thematic_analysis", "strategic_recommendations", "overall_status_rating"]
        }

        # --- Construct Prompt ---
        # Get information from the planning stage
        planning_results = self._get_data(context, "planner", None, {})
        doc_type = planning_results.get("document_type", "Document")
        
        prompt = f"""
You are an expert AI 'Evaluator' synthesizing an overall assessment based on previously evaluated items from a {doc_type} analysis.

Assessment Type: {assessment_type}
Total Number of Evaluated Items: {len(evaluated_items)} (Showing representative sample of {len(sample_items)})

Summary of Representative Sample of Evaluated '{data_type}':
```json
{json.dumps(sample_items, indent=2, default=str)}
```

Statistics Calculated from ALL Evaluated Items:
```json
{json.dumps(stats, indent=2)}
```

Document Analysis Context:
```json
{json.dumps({k: v for k, v in planning_results.items() if k in ["document_type", "key_topics_or_sections", "extraction_focus"]}, indent=2, default=str)}
```

Your Task:
Based on the provided sample of evaluated items, overall statistics, and document context, generate a concise yet insightful overall assessment. Provide analysis beyond just listing the items. Identify patterns, critical areas (especially items marked as failed or having high severity/priority), and strategic implications.

Output Format: Respond ONLY with a valid JSON object matching this schema:
```json
{json.dumps(overall_schema, indent=2)}
```
"""

        # --- LLM Call ---
        try:
            overall_assessment_result = await self._generate_structured(
                prompt=prompt,
                output_schema=overall_schema,
                context=context,
                temperature=self.options.get("overall_assessment_temperature", self.DEFAULT_OVERALL_ASSESSMENT_TEMP),
                max_tokens=self.options.get("overall_assessment_max_tokens", self.DEFAULT_OVERALL_ASSESSMENT_TOKENS)
            )
            
            # Add metadata
            overall_assessment_result["status"] = "success"
            overall_assessment_result["metadata"] = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_items_evaluated": len(evaluated_items),
                "assessment_type": assessment_type
            }
            
            return overall_assessment_result

        except Exception as e:
            self._log_error(f"LLM call for overall assessment failed: {str(e)}", context, exc_info=True)
            return {
                "summary": f"Failed to generate overall assessment: {str(e)}", 
                "status": "failed",
                "error": str(e)
            }
            
    def _create_strategic_sample(self, 
                               items: List[Dict[str, Any]], 
                               assessment_type: str, 
                               max_items: int = 25) -> List[Dict[str, Any]]:
        """
        Creates a strategic sample of items for the overall assessment,
        ensuring representation across severity/importance levels.
        """
        if len(items) <= max_items:
            return self._summarize_items_for_prompt(items, assessment_type)
        
        # Select items strategically based on assessment type
        selected_items = []
        
        if assessment_type == "assess":
            # Prioritize by severity
            severities = ["critical", "high", "medium", "low"]
            # Allocate slots for each severity level with emphasis on higher severities
            slots = {
                "critical": int(max_items * 0.4),
                "high": int(max_items * 0.3),
                "medium": int(max_items * 0.2),
                "low": int(max_items * 0.1)
            }
            
            # Ensure at least 1 slot for each severity if possible
            for sev in slots:
                if slots[sev] == 0:
                    slots[sev] = 1
                    
            # Adjust if total exceeds max_items
            total_slots = sum(slots.values())
            if total_slots > max_items:
                # Scale down proportionally
                scale = max_items / total_slots
                for sev in slots:
                    slots[sev] = max(1, int(slots[sev] * scale))
                # Final adjustment
                while sum(slots.values()) > max_items:
                    slots["low"] = max(0, slots["low"] - 1)
                    if sum(slots.values()) <= max_items:
                        break
                    slots["medium"] = max(0, slots["medium"] - 1)
            
            # Select items for each severity level
            for severity in severities:
                severity_items = [item for item in items if item.get("severity", "").lower() == severity or 
                                item.get("evaluated_severity", "").lower() == severity]
                
                # If more items than slots, prioritize failed items and then sample
                if len(severity_items) > slots[severity] and slots[severity] > 0:
                    # First include failed items
                    failed_items = [item for item in severity_items if item.get("_evaluation_status") == "failed"]
                    remaining_slots = slots[severity] - min(len(failed_items), slots[severity])
                    
                    # Add failed items up to the slot limit
                    selected_items.extend(failed_items[:slots[severity]])
                    
                    # Add non-failed items for remaining slots
                    if remaining_slots > 0:
                        non_failed = [item for item in severity_items if item.get("_evaluation_status") != "failed"]
                        # Spread selection across the list for diversity
                        step = max(1, len(non_failed) // remaining_slots)
                        sampled_indices = [i for i in range(0, len(non_failed), step)][:remaining_slots]
                        selected_items.extend([non_failed[i] for i in sampled_indices])
                else:
                    # If fewer items than slots, take all of them
                    selected_items.extend(severity_items[:slots[severity]])
        
        elif assessment_type == "extract":
            # Similar approach but prioritize by priority
            priorities = ["high", "medium", "low"]
            # Distribute slots
            slots_per_priority = max_items // len(priorities)
            
            for priority in priorities:
                priority_items = [item for item in items if 
                                item.get("priority", "").lower() == priority or 
                                item.get("evaluated_priority", "").lower() == priority]
                
                if len(priority_items) > slots_per_priority:
                    # Spread selection across the list
                    step = max(1, len(priority_items) // slots_per_priority)
                    indices = [i for i in range(0, len(priority_items), step)][:slots_per_priority]
                    selected_items.extend([priority_items[i] for i in indices])
                else:
                    selected_items.extend(priority_items)
                    
        else:
            # For other types, ensure diversity by taking items spread across the list
            step = max(1, len(items) // max_items)
            indices = [i for i in range(0, len(items), step)][:max_items]
            selected_items = [items[i] for i in indices]
            
        # If we ended up with fewer than max_items, add more items from the original list
        if len(selected_items) < max_items:
            remaining_items = [item for item in items if item not in selected_items]
            additional_needed = max_items - len(selected_items)
            if remaining_items and additional_needed > 0:
                step = max(1, len(remaining_items) // additional_needed)
                indices = [i for i in range(0, len(remaining_items), step)][:additional_needed]
                selected_items.extend([remaining_items[i] for i in indices])
        
        # Summarize the selected items
        return self._summarize_items_for_prompt(selected_items, assessment_type)
    
    def _summarize_items_for_prompt(self, items: List[Dict[str, Any]], assessment_type: str) -> List[Dict[str, Any]]:
        """Create summarized versions of items for the overall assessment prompt."""
        summary_items = []
        
        # Define key fields to include based on assessment type
        key_fields = {
            "assess": ["id", "title", "evaluated_severity", "potential_impact", "_evaluation_status"],
            "extract": ["id", "description", "owner", "evaluated_priority", "is_actionable", "_evaluation_status"],
            "distill": ["id", "text", "evaluated_significance", "evaluated_relevance_score", "_evaluation_status"],
            "analyze": ["id", "dimension", "criteria", "maturity_rating", "evaluator_confidence", "_evaluation_status"]
        }.get(assessment_type, ["id", "evaluated_relevance", "evaluated_importance", "_evaluation_status"])
        
        for item in items:
            summary = {k: item.get(k) for k in key_fields if k in item}
            
            # Add a unique ID if missing
            if "id" not in summary:
                summary["id"] = f"item_{uuid.uuid4().hex[:6]}"
                
            # Truncate any long text fields
            for field in ["title", "description", "text", "impact"]:
                if field in summary and isinstance(summary[field], str) and len(summary[field]) > 150:
                    summary[field] = summary[field][:150] + "..."
                    
            summary_items.append(summary)
            
        return summary_items
        
    def _calculate_item_statistics(self, items: List[Dict[str, Any]], assessment_type: str) -> Dict[str, Any]:
        """Calculate statistics on evaluated items to help with overall assessment."""
        stats = {"total_items": len(items), "items_evaluation_failed": 0}
        if not items: 
            return stats

        failed_items = [item for item in items if item.get("_evaluation_status") == "failed"]
        stats["items_evaluation_failed"] = len(failed_items)
        
        # Filter out failed items for calculating stats on evaluated fields
        valid_items = [item for item in items if item.get("_evaluation_status") == "success"]

        if not valid_items: 
            return stats # No successful items to analyze

        if assessment_type == "assess":
            severity_counts = {}
            for item in valid_items:
                severity = item.get("evaluated_severity", "unknown")
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            stats["severity_counts"] = severity_counts
            total_valid = len(valid_items)
            if total_valid > 0:
                critical_high = severity_counts.get("Critical", 0) + severity_counts.get("High", 0)
                stats["critical_high_percentage"] = round((critical_high / total_valid) * 100, 1)
                
            # Include category distribution if available
            categories = {}
            for item in valid_items:
                category = item.get("category", "unknown")
                categories[category] = categories.get(category, 0) + 1
            
            if len(categories) > 1:  # Only include if we have multiple categories
                stats["category_distribution"] = categories

        elif assessment_type == "extract":
            priority_counts = {}
            actionable_counts = {"actionable": 0, "not_actionable": 0}
            
            for item in valid_items:
                priority = item.get("evaluated_priority", "unknown")
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
                
                if item.get("is_actionable") == True:
                    actionable_counts["actionable"] += 1
                elif item.get("is_actionable") == False:
                    actionable_counts["not_actionable"] += 1
                    
            stats["priority_counts"] = priority_counts
            stats["actionable_counts"] = actionable_counts
            
            # Calculate percentage of items with owners
            items_with_owner = sum(1 for item in valid_items if item.get("owner") and item.get("owner") != "Unknown")
            stats["items_with_owner_percentage"] = round((items_with_owner / len(valid_items)) * 100, 1) if valid_items else 0

        elif assessment_type == "distill":
            significance_counts = {}
            total_relevance = 0
            count_relevance = 0
            
            for item in valid_items:
                significance = item.get("evaluated_significance", "unknown")
                significance_counts[significance] = significance_counts.get(significance, 0) + 1
                
                relevance = item.get("evaluated_relevance_score")
                if isinstance(relevance, (int, float)):
                    total_relevance += relevance
                    count_relevance += 1
                    
            stats["significance_counts"] = significance_counts
            stats["average_relevance_score"] = round(total_relevance / count_relevance, 2) if count_relevance > 0 else None
            
            # Get topic distribution
            topics = {}
            for item in valid_items:
                topic = item.get("topic", "unknown")
                if topic and topic.lower() != "unknown" and topic.lower() != "n/a":
                    topics[topic] = topics.get(topic, 0) + 1
            
            if topics:
                stats["topic_distribution"] = topics

        elif assessment_type == "analyze":
            dimension_ratings = {}
            total_confidence = 0
            count_confidence = 0
            
            for item in valid_items:
                dimension = item.get("dimension", "unknown")
                rating = item.get("maturity_rating")
                confidence = item.get("evaluator_confidence")

                if rating is not None:
                    if dimension not in dimension_ratings:
                        dimension_ratings[dimension] = {"ratings": [], "count": 0, "sum": 0, "average": None}
                    
                    dimension_ratings[dimension]["ratings"].append(rating)
                    dimension_ratings[dimension]["count"] += 1
                    if isinstance(rating, (int, float)):
                        dimension_ratings[dimension]["sum"] += rating

                if isinstance(confidence, (int, float)):
                    total_confidence += confidence
                    count_confidence += 1

            # Calculate averages for dimensions
            for dim_data in dimension_ratings.values():
                if dim_data["count"] > 0 and isinstance(dim_data["sum"], (int, float)):
                    dim_data["average"] = round(dim_data["sum"] / dim_data["count"], 2)
                del dim_data["sum"]  # Remove temporary sum

            stats["dimension_ratings_summary"] = dimension_ratings
            stats["average_evaluator_confidence"] = round(total_confidence / count_confidence, 2) if count_confidence > 0 else None

        # Common statistics across all types - rationale presence
        items_with_rationale = sum(1 for item in valid_items if item.get("rationale"))
        stats["items_with_rationale_percentage"] = round((items_with_rationale / len(valid_items)) * 100, 1) if valid_items else 0

        return stats