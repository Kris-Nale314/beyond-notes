# core/agents/evaluator.py
import logging
import json
import copy
import time # Keep for potential timing info if needed
import uuid # Keep if generating IDs within evaluator (unlikely now)
from typing import Dict, Any, List, Optional, Tuple

# Import base class and context/LLM types
from .base import BaseAgent
from core.llm.customllm import CustomLLM
from core.models.context import ProcessingContext

# Using BaseAgent's logger, but module-level can also be used
logger = logging.getLogger(__name__)

class EvaluatorAgent(BaseAgent):
    """
    Evaluates aggregated items based on assessment goals, criteria, evidence,
    and planner guidance. Assigns scores, ratings, or assessments using an LLM
    in batches, and generates an overall assessment summary.
    """

    DEFAULT_BATCH_SIZE = 20 # Smaller batch size might be needed due to richer context (evidence) per item
    DEFAULT_EVALUATOR_TEMP = 0.2
    DEFAULT_EVALUATOR_TOKENS = 4000 # May need adjustment based on batch size and item complexity
    DEFAULT_OVERALL_ASSESSMENT_TEMP = 0.4
    DEFAULT_OVERALL_ASSESSMENT_TOKENS = 1500

    def __init__(self, llm: CustomLLM, options: Optional[Dict[str, Any]] = None):
        """Initialize the EvaluatorAgent."""
        super().__init__(llm, options) # Pass options to BaseAgent
        self.role = "evaluator"
        # self.name is automatically set in BaseAgent
        self.logger = logging.getLogger(f"core.agents.{self.name}") # Get logger from BaseAgent
        self.logger.info(f"EvaluatorAgent initialized with options: {self.options}")

    async def process(self, context: ProcessingContext) -> Optional[Dict[str, Any]]:
        """
        Orchestrates the evaluation: fetches aggregated items, evaluates them
        in batches using LLM, generates an overall assessment, and stores results.

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
            # --- Get Items to Evaluate (from Aggregator ideally) ---
            # Use BaseAgent helper, specifying the role to get data from
            items_to_evaluate = self._get_data(context, "aggregator", data_type, default=None)

            if items_to_evaluate is None:
                self._log_warning(f"No aggregated '{data_type}' found. Trying extractor output.", context)
                items_to_evaluate = self._get_data(context, "extractor", data_type, default=[]) # Fallback

            if not isinstance(items_to_evaluate, list):
                 self._log_error(f"Data for '{data_type}' is not a list (Type: {type(items_to_evaluate)}). Cannot evaluate.", context)
                 raise TypeError(f"Input data for evaluation ('{data_type}') is not a list.")

            total_items = len(items_to_evaluate)
            if total_items == 0:
                self._log_info(f"No '{data_type}' items found to evaluate.", context)
                # Ensure empty list is stored if needed, though overall assessment will handle empty input
                self._store_data(context, data_type, []) # Store empty evaluated list
                return {"items_evaluated": 0, "items_failed": 0}

            self._log_info(f"Starting evaluation for {total_items} '{data_type}' items.", context)

            # --- Perform Evaluation (Handles Batching) ---
            evaluated_items, items_failed_count, batches_processed = await self._perform_evaluation(
                context=context,
                items=items_to_evaluate,
                data_type=data_type,
                assessment_type=assessment_type
            )
            summary_stats["items_evaluated"] = total_items - items_failed_count
            summary_stats["items_failed"] = items_failed_count
            summary_stats["batches_processed"] = batches_processed

            # --- Store Evaluated Items ---
            # Use BaseAgent helper - stores in context.data["evaluated"][data_type]
            self._store_data(context, data_type, evaluated_items)
            self._log_info(f"Stored {len(evaluated_items)} evaluated '{data_type}' items in context.", context)

            # --- Generate and Store Overall Assessment ---
            try:
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
                self._log_error(f"Failed to generate overall assessment: {e_overall}", context, exc_info=True)
                # Add warning to context, but don't fail the whole stage
                context.add_warning(f"Failed to generate overall assessment: {str(e_overall)}", stage="evaluation")


            self._log_info(f"Evaluation phase complete. Evaluated: {summary_stats['items_evaluated']}, Failed: {summary_stats['items_failed']}, Batches: {batches_processed}", context)
            return summary_stats

        except Exception as e:
            self._log_error(f"Evaluation phase failed: {str(e)}", context, exc_info=True)
            raise # Re-raise for orchestrator

    def _get_data_type_for_assessment(self, assessment_type: str) -> Optional[str]:
        """Determine the key/type of data being processed based on assessment type."""
        # Consistent with Extractor/Aggregator logic
        data_type_map = {
            "extract": "action_items",
            "assess": "issues",
            "distill": "key_points",
            "analyze": "evidence"
        }
        return data_type_map.get(assessment_type)

    async def _perform_evaluation(self,
                                  context: ProcessingContext,
                                  items: List[Dict[str, Any]],
                                  data_type: str,
                                  assessment_type: str
                                  ) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Manages the evaluation process, handling batching if necessary.

        Returns:
            Tuple[List of evaluated items, count of items failed, number of batches processed]
        """
        items_count = len(items)
        batch_size = self.options.get("evaluation_batch_size", self.DEFAULT_BATCH_SIZE)
        num_batches = (items_count + batch_size - 1) // batch_size

        all_evaluated_items = []
        total_failed_count = 0
        batches_processed_count = 0

        if num_batches <= 1:
            self._log_info(f"Processing {items_count} items in a single evaluation batch.", context)
            batch_results, failed_count = await self._evaluate_batch(context, items, data_type, assessment_type, 0, 1)
            all_evaluated_items = batch_results
            total_failed_count = failed_count
            batches_processed_count = 1
        else:
            self._log_info(f"Processing {items_count} items in {num_batches} evaluation batches (size: {batch_size}).", context)
            for batch_num in range(num_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, items_count)
                batch_items = items[start_idx:end_idx]

                self._log_info(f"Evaluating batch {batch_num + 1}/{num_batches} with {len(batch_items)} items.", context)
                batch_results, failed_count = await self._evaluate_batch(
                    context, batch_items, data_type, assessment_type, batch_num, num_batches
                )
                all_evaluated_items.extend(batch_results) # Append results from this batch
                total_failed_count += failed_count
                batches_processed_count += 1

                # Update context progress
                batch_progress = (batch_num + 1) / num_batches
                context.update_stage_progress(batch_progress, f"Evaluated batch {batch_num + 1}/{num_batches}")

        return all_evaluated_items, total_failed_count, batches_processed_count


    async def _evaluate_batch(self,
                              context: ProcessingContext,
                              batch_items: List[Dict[str, Any]],
                              data_type: str,
                              assessment_type: str,
                              batch_num: int,
                              total_batches: int
                             ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Performs the core LLM call to evaluate a single batch of items. Handles merging
        of results back into original items carefully.

        Returns:
            Tuple[List of processed items for this batch (evaluated or original+error), count of items failed in batch]
        """
        processed_batch_items = [] # Will hold items (original structure updated with evaluation)
        failed_count = 0
        item_name_singular = data_type.rstrip('s')

        # --- Get Evaluation Details ---
        # Determine the specific task and the schema for fields to be added/updated
        evaluation_task, item_update_schema = self._get_evaluation_details(assessment_type, context)
        if not evaluation_task:
            self._log_warning(f"No evaluation task defined for assessment type '{assessment_type}'. Items may not be fully evaluated.", context)
            # Return original items if no task defined? Or proceed with default? Let's proceed cautiously.
            # evaluation_task = "Assess relevance and importance." # Example fallback

        # --- Define LLM Output Schema ---
        # This schema describes the structure the LLM should return for *each item* in the batch.
        # It includes ONLY the fields the LLM needs to generate/evaluate.
        llm_item_output_schema = {
            "type": "object",
            "properties": {
                 "id": {"type": "string", "description": "The unique ID of the item being evaluated."},
                **item_update_schema.get("properties", {}) # Evaluation fields to be added/updated
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
        prompt_items_section = ""
        items_in_prompt_count = 0
        MAX_EVIDENCE_LENGTH = 500 # Limit evidence length per item in prompt

        for item in batch_items:
            item_id = item.get("id", f"temp_id_{uuid.uuid4().hex[:4]}") # Ensure ID exists for prompt reference
            item_details_json = json.dumps(item, indent=2, default=str)
            evidence_text = self._get_evidence_text(context, item, max_length=MAX_EVIDENCE_LENGTH)

            prompt_items_section += f"\n--- Item ID: {item_id} ---\n"
            prompt_items_section += f"Item Details:\n```json\n{item_details_json}\n```\n"
            prompt_items_section += f"Associated Evidence:\n{evidence_text}\n"
            prompt_items_section += "------\n"
            items_in_prompt_count += 1

        if items_in_prompt_count == 0:
            self._log_warning("No valid items found in batch to construct prompt.", context)
            return [], 0 # Return empty list and 0 failures


        # Get planner guidance
        planning_output = self._get_data(context, "planner", default={})
        planner_guidance = planning_output.get('evaluation_focus', f'Evaluate based on standard criteria for {item_name_singular}.')
        base_instructions = context.get_workflow_instructions(self.role) or "Evaluate each item according to the specific task based on its details and evidence."

        batch_info = f"Batch {batch_num + 1}/{total_batches}" if total_batches > 0 else "Single Batch Evaluation"

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
1.  For EACH item listed above (identified by 'Item ID'):
2.  Review its details and associated evidence.
3.  Apply the 'Specific Evaluation Task' to determine the values for the required evaluation fields (e.g., severity, priority, relevance, rationale).
4.  Generate a response containing ONLY the evaluated fields and the original 'id' for EACH item.

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
                 processed_item = copy.deepcopy(original_item) # Start with a copy of the original

                 if item_id and item_id in evaluation_results_map:
                      # Merge evaluation fields from LLM result into the item
                      evaluation_data = evaluation_results_map[item_id]
                      # Only update with keys present in the evaluation schema (plus 'id')
                      fields_to_update = list(item_update_schema.get("properties", {}).keys())

                      for key in fields_to_update:
                           if key in evaluation_data:
                                processed_item[key] = evaluation_data[key]
                           # else: # Optional: log if an expected eval field wasn't returned
                           #     self._log_debug(f"Evaluation field '{key}' missing for item {item_id}", context)

                      # Add a timestamp or flag
                      processed_item["_evaluation_status"] = "success"
                      processed_item["_evaluated_at"] = datetime.now(timezone.utc).isoformat()

                 else:
                      # Item evaluation failed or missing from response
                      self._log_warning(f"Evaluation data not found for item ID '{item_id}' in LLM response.", context)
                      processed_item["_evaluation_status"] = "failed"
                      processed_item["_evaluation_error"] = "Evaluation data missing in LLM response."
                      failed_count += 1

                 processed_batch_items.append(processed_item) # Add the updated/original item


            # Check if any items from the original batch were missed entirely
            if len(processed_batch_items) != len(batch_items):
                 self._log_warning(f"Mismatch in processed items count ({len(processed_batch_items)}) vs original batch size ({len(batch_items)}). Some items might have been lost.", context)
                 # This case shouldn't happen with the current logic, but good to note.


            self._log_info(f"Evaluation successful for batch ({batch_info}). {len(processed_batch_items) - failed_count} items evaluated, {failed_count} failed.", context)


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
            # Don't re-raise here, allow process to continue with failed items marked

        return processed_batch_items, failed_count

    def _get_evidence_text(self, context: ProcessingContext, item: Dict[str, Any], max_length=500) -> str:
        """Retrieve and concatenate evidence text for an item, respecting max_length."""
        # Adapted from old evaluator - uses context method directly
        evidence_texts = []
        total_len = 0
        item_id = item.get("id")

        if not item_id:
            # Fallback for items potentially missing ID temporarily during processing
            return "Item ID missing, cannot retrieve evidence."

        # Get evidence using the context method
        # This correctly handles merged items if Aggregator updated the links
        evidence_list = context.get_evidence_for_item(item_id)

        if evidence_list:
            # self._log_debug(f"Found {len(evidence_list)} evidence snippets for item {item_id}", context)
            for evidence in evidence_list:
                evidence_text = evidence.get("text", "")
                if evidence_text:
                    chunk_idx = evidence.get("metadata", {}).get("chunk_index", "?")
                    confidence = evidence.get("confidence")
                    conf_str = f" (Conf: {confidence:.2f})" if confidence is not None else ""
                    snippet = f"- Src(Chunk {chunk_idx}{conf_str}): {evidence_text}"

                    # Check length before adding
                    if total_len + len(snippet) < max_length:
                        evidence_texts.append(snippet)
                        total_len += len(snippet)
                    else:
                        # Add truncated indicator and stop
                        evidence_texts.append("- [...evidence truncated...]")
                        total_len = max_length # Prevent adding more
                        break
        else:
            # Fallback if no linked evidence
             for field in ["description", "text", "evidence_text", "title"]: # Check common fields
                if field in item and item[field] and isinstance(item[field], str):
                    return f"Item Content Fallback: {item[field][:max_length]}"
             return "No specific evidence text found or linked."

        return "\n".join(evidence_texts) if evidence_texts else "No evidence text retrieved."

    def _get_evaluation_details(self, assessment_type: str, context: ProcessingContext) -> Tuple[str, Dict[str, Any]]:
        """
        Define the evaluation task and the schema for the fields to be added/updated
        by the evaluator based on the assessment type.
        """
        # Adapted from previous suggestion - context added for potential config access
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
        """Generates an overall assessment summary using the LLM, based on evaluated items."""
        # Adapted from old evaluator
        if not evaluated_items:
            self._log_warning("Cannot create overall assessment: No evaluated items provided.", context)
            return {"summary": "No items were available for overall assessment.", "status":"skipped"}

        self._log_info(f"Generating overall assessment based on {len(evaluated_items)} evaluated '{data_type}'.", context)

        # --- Prepare Data for Prompt ---
        # Summarize evaluated items concisely
        summary_items = []
        max_items_in_prompt = self.options.get("overall_assessment_max_items", 25)
        # Prioritize showing items with potential issues or high importance/severity
        # Simple sort for now (e.g., by severity/priority if available) - can be improved
        # sorted_items = sorted(evaluated_items, key=lambda x: x.get("evaluated_severity_score", 0), reverse=True) # Example sort key needed
        item_sample = evaluated_items[:max_items_in_prompt] # Simple slice for now

        for item in item_sample:
            summary = {"id": item.get("id")}
            # Include key evaluated fields based on type
            if assessment_type == "assess":
                summary.update({k: item.get(k) for k in ["title", "evaluated_severity", "potential_impact", "_evaluation_status"] if k in item})
            elif assessment_type == "extract":
                summary.update({k: item.get(k) for k in ["description", "owner", "evaluated_priority", "_evaluation_status"] if k in item})
            elif assessment_type == "distill":
                 summary.update({k: item.get(k) for k in ["text", "evaluated_significance", "evaluated_relevance_score", "_evaluation_status"] if k in item})
            elif assessment_type == "analyze":
                summary.update({k: item.get(k) for k in ["dimension", "criteria", "maturity_rating", "evaluator_confidence", "rationale", "_evaluation_status"] if k in item})
            else:
                 summary.update({k: item.get(k) for k in ["text", "type", "evaluated_relevance", "evaluated_importance", "_evaluation_status"] if k in item})
            summary_items.append(summary)

        # Calculate statistics
        stats = self._calculate_item_statistics(evaluated_items, assessment_type)

        # --- Define Output Schema ---
        # Same schema as old evaluator
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
        prompt = f"""
You are an expert AI 'Evaluator' synthesizing an overall assessment based on previously evaluated items from a document analysis.

Assessment Type: {assessment_type}
Total Number of Evaluated Items: {len(evaluated_items)} (Showing sample of up to {max_items_in_prompt})

Summary of Sample Evaluated '{data_type}':
```json
{json.dumps(summary_items, indent=2, default=str)}
```

Statistics Calculated from ALL Evaluated Items:
```json
{json.dumps(stats, indent=2)}
```

Your Task:
Based on the provided sample of evaluated items and the overall statistics, generate a concise yet insightful overall assessment. Provide analysis beyond just listing the items. Identify patterns, critical areas (especially items marked as failed or having high severity/priority), and strategic implications.

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
            overall_assessment_result["status"] = "success" # Add status field
            return overall_assessment_result

        except Exception as e_overall:
            self._log_error(f"LLM call for overall assessment failed: {e_overall}", context, exc_info=True)
            # Return a failure indicator instead of raising, maybe?
            return {"summary": f"Failed to generate overall assessment: {str(e_overall)}", "status":"failed"}


    def _calculate_item_statistics(self, items: List[Dict[str, Any]], assessment_type: str) -> Dict[str, Any]:
        """Calculate statistics on evaluated items to help with overall assessment."""
        # Adapted directly from old evaluator - seems robust
        stats = {"total_items": len(items), "items_evaluation_failed": 0}
        if not items: return stats

        failed_items = [item for item in items if item.get("_evaluation_status") == "failed"]
        stats["items_evaluation_failed"] = len(failed_items)
        # Filter out failed items for calculating stats on evaluated fields
        valid_items = [item for item in items if item.get("_evaluation_status") == "success"]

        if not valid_items: return stats # No successful items to analyze

        if assessment_type == "assess":
            severity_counts = {}
            for item in valid_items:
                severity = item.get("evaluated_severity", "unknown")
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            stats["severity_counts"] = severity_counts
            total_valid = len(valid_items)
            if total_valid > 0:
                critical_high = severity_counts.get("Critical", 0) + severity_counts.get("High", 0) # Use capitalized enum values
                stats["critical_high_percentage"] = round((critical_high / total_valid) * 100, 1)

        elif assessment_type == "extract":
            priority_counts = {}
            actionable_counts = {"actionable": 0, "not_actionable": 0}
            for item in valid_items:
                priority = item.get("evaluated_priority", "unknown") # Should match enum values
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
                if item.get("is_actionable") == True:
                    actionable_counts["actionable"] += 1
                elif item.get("is_actionable") == False:
                    actionable_counts["not_actionable"] += 1
            stats["priority_counts"] = priority_counts
            stats["actionable_counts"] = actionable_counts

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
                    if isinstance(rating, (int, float)): # Handle potential non-numeric ratings if schema changes
                         dimension_ratings[dimension]["sum"] += rating

                if isinstance(confidence, (int, float)):
                    total_confidence += confidence
                    count_confidence += 1

            # Calculate averages for dimensions
            for dim_data in dimension_ratings.values():
                if dim_data["count"] > 0 and isinstance(dim_data["sum"], (int, float)):
                    dim_data["average"] = round(dim_data["sum"] / dim_data["count"], 2)
                del dim_data["sum"] # Remove temporary sum

            stats["dimension_ratings_summary"] = dimension_ratings # Renamed for clarity
            stats["average_evaluator_confidence"] = round(total_confidence / count_confidence, 2) if count_confidence > 0 else None

        return stats
