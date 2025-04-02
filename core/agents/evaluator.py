# core/agents/evaluator.py
import logging
import json
import copy
from typing import Dict, Any, List, Optional

# Import base class and context/LLM types
from .base import BaseAgent
from core.llm.customllm import CustomLLM
from core.models.context import ProcessingContext

class EvaluatorAgent(BaseAgent):
    """
    Assesses aggregated information based on assessment configuration,
    criteria, evidence, and potentially planner guidance. Assigns ratings,
    severities, priorities, etc., using an LLM.
    
    Uses the enhanced ProcessingContext for standardized data storage and retrieval.
    """

    def __init__(self, llm: CustomLLM, options: Optional[Dict[str, Any]] = None):
        """Initialize the EvaluatorAgent."""
        super().__init__(llm, options)
        self.role = "evaluator"
        self.name = "EvaluatorAgent"
        self.logger = logging.getLogger(f"core.agents.{self.name}")
        self.logger.info(f"EvaluatorAgent initialized.")

    async def process(self, context: ProcessingContext) -> Optional[Dict[str, Any]]:
        """
        Orchestrates the evaluation process based on assessment type, operating
        on aggregated data found in context.

        Args:
            context: The shared ProcessingContext object.

        Returns:
            A dictionary summarizing the evaluation results, or None.
        """
        self._log_info(f"Starting evaluation phase for assessment: '{context.display_name}'", context)

        assessment_type = context.assessment_type
        items_evaluated_count = 0
        items_failed_evaluation = 0

        try:
            # --- Determine which data type to evaluate based on assessment type ---
            data_type_map = {
                "extract": "action_items",
                "assess": "issues",
                "distill": "key_points",
                "analyze": "evidence"
            }
            
            data_type = data_type_map.get(assessment_type)
            if not data_type:
                self._log_warning(f"Unsupported assessment type: '{assessment_type}'. Skipping evaluation.", context)
                return None

            # --- Get items to evaluate using the enhanced context method ---
            items_to_evaluate = self._get_data(context, "aggregator", data_type, [])
            
            # If no aggregated items found, try getting extracted items directly
            if not items_to_evaluate:
                self._log_warning(f"No aggregated {data_type} found. Trying to get extracted items directly.", context)
                items_to_evaluate = self._get_data(context, "extractor", data_type, [])
            
            total_items_to_evaluate = len(items_to_evaluate)

            if not items_to_evaluate:
                self._log_info(f"No {data_type} found to evaluate.", context)
                # Store empty list in context
                self._store_data(context, data_type, [])
                return {"evaluated_count": 0, "failed_count": 0}

            self._log_info(f"Starting evaluation for {total_items_to_evaluate} {data_type}.", context)
            
            # Determine batch size for evaluation
            batch_size = self.options.get("evaluation_batch_size", 50)
            use_batching = total_items_to_evaluate > batch_size
            
            if use_batching:
                self._log_info(f"Using batch evaluation for {total_items_to_evaluate} items (batch size: {batch_size})", context)
                # Evaluate items in batches
                evaluated_items = []
                
                # Calculate number of batches
                num_batches = (total_items_to_evaluate + batch_size - 1) // batch_size
                
                for batch_num in range(num_batches):
                    start_idx = batch_num * batch_size
                    end_idx = min(start_idx + batch_size, total_items_to_evaluate)
                    batch_items = items_to_evaluate[start_idx:end_idx]
                    
                    self._log_info(f"Evaluating batch {batch_num+1}/{num_batches} with {len(batch_items)} items", context)
                    
                    # Process batch
                    batch_results, batch_success, batch_failed = await self._evaluate_items_batch(
                        context=context,
                        items=batch_items,
                        data_type=data_type,
                        batch_num=batch_num,
                        total_batches=num_batches
                    )
                    
                    # Add results to overall collection
                    evaluated_items.extend(batch_results)
                    items_evaluated_count += batch_success
                    items_failed_evaluation += batch_failed
                    
                    # Update progress
                    batch_progress = (batch_num + 1) / num_batches
                    context.update_stage_progress(batch_progress, f"Evaluated batch {batch_num+1}/{num_batches}")
                    
            else:
                # --- Evaluate Items Iteratively without batching ---
                evaluated_items = []
                for i, item in enumerate(items_to_evaluate):
                    if not isinstance(item, dict):
                        self._log_warning(f"Skipping invalid item (not a dict) at index {i}: {item}", context)
                        items_failed_evaluation += 1
                        continue

                    self._log_debug(f"Evaluating {data_type} item {i+1}/{total_items_to_evaluate} (ID: {item.get('id', 'N/A')})...", context)
                    try:
                        evaluated_item = await self._evaluate_single_item_llm(context, item, data_type)
                        evaluated_items.append(evaluated_item)
                        items_evaluated_count += 1
                    except Exception as e:
                        self._log_error(f"Failed to evaluate {data_type} item (ID: {item.get('id', 'N/A')}): {e}", context, exc_info=False)
                        items_failed_evaluation += 1
                        # Keep item in list but mark as failed
                        item["_evaluation_error"] = str(e)
                        evaluated_items.append(item)

                    # Update progress
                    context.update_stage_progress((i + 1) / total_items_to_evaluate)

            # --- Store evaluated items in context using the enhanced method ---
            self._store_data(context, data_type, evaluated_items)
            self._log_info(f"Stored {len(evaluated_items)} evaluated {data_type} in context", context)

            # --- Generate Overall Assessment (for all assessment types) ---
            try:
                overall_assessment = await self._create_overall_assessment_llm(context, evaluated_items, data_type)
                # Store using the enhanced context method
                self._store_data(context, "overall_assessment", overall_assessment)
                self._log_info("Generated and stored overall assessment.", context)
            except Exception as e:
                self._log_error(f"Failed to generate overall assessment: {e}", context, exc_info=True)
                context.add_warning(f"Failed to generate overall assessment: {str(e)}")

            # --- Final Summary ---
            summary = {
                "items_evaluated": items_evaluated_count,
                "items_failed_evaluation": items_failed_evaluation
            }
            self._log_info(f"Evaluation phase complete. Evaluated: {items_evaluated_count}, Failed: {items_failed_evaluation}", context)
            return summary

        except Exception as e:
            self._log_error(f"Evaluation phase failed: {str(e)}", context, exc_info=True)
            raise  # Re-raise to fail the stage

    async def _evaluate_items_batch(self,
                                  context: ProcessingContext,
                                  items: List[Dict[str, Any]],
                                  data_type: str,
                                  batch_num: int,
                                  total_batches: int) -> tuple:
        """
        Evaluate a batch of items.
        
        Returns:
            Tuple of (evaluated_items, success_count, failure_count)
        """
        batch_size = len(items)
        batch_succeeded = 0
        batch_failed = 0
        batch_results = []
        
        item_name = data_type.rstrip('s')  # Get singular name
        
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                self._log_warning(f"Batch {batch_num+1}: Skipping invalid item (not a dict) at index {i}", context)
                batch_failed += 1
                continue
                
            # Update more granular progress
            batch_item_progress = (batch_num + (i + 1) / batch_size) / total_batches
            if (i % 5) == 0:  # Update every 5 items to avoid too many updates
                progress_msg = f"Batch {batch_num+1}/{total_batches}: Item {i+1}/{batch_size}"
                context.update_stage_progress(batch_item_progress, progress_msg)
                
            try:
                evaluated_item = await self._evaluate_single_item_llm(context, item, data_type)
                batch_results.append(evaluated_item)
                batch_succeeded += 1
            except Exception as e:
                self._log_error(f"Failed to evaluate {item_name} (ID: {item.get('id', 'N/A')}): {e}", context, exc_info=False)
                batch_failed += 1
                # Keep item but mark as failed
                item["_evaluation_error"] = str(e)
                batch_results.append(item)
                
        return batch_results, batch_succeeded, batch_failed

    def _get_evidence_text(self, context: ProcessingContext, item: Dict[str, Any], max_length=1000) -> str:
        """Retrieve and concatenate evidence text for an item."""
        evidence_texts = []
        total_len = 0
        
        # Get the item ID for evidence lookup
        item_id = item.get("id")
        if not item_id:
            return "No item ID found for evidence lookup."
            
        # Get evidence using the enhanced context method
        evidence_list = context.get_evidence_for_item(item_id)
        
        if not evidence_list:
            # If no direct evidence, check merged_item_ids if available
            merged_ids = item.get("merged_item_ids", [])
            for merged_id in merged_ids:
                merged_evidence = context.get_evidence_for_item(merged_id)
                evidence_list.extend(merged_evidence)
                
        if evidence_list:
            self._log_debug(f"Found {len(evidence_list)} evidence snippets for item {item_id}", context)
            for evidence in evidence_list:
                evidence_text = evidence.get("text", "")
                if evidence_text:
                    chunk_info = evidence.get("metadata", {}).get("chunk_index", "?")
                    snippet = f"- Source (Chunk {chunk_info}): {evidence_text}"
                    
                    # Check if adding this would exceed max length
                    if total_len + len(snippet) < max_length:
                        evidence_texts.append(snippet)
                        total_len += len(snippet)
                    else:
                        evidence_texts.append(f"- [Additional evidence truncated due to length]")
                        break
        
        if not evidence_texts:
            # Fallback if no evidence links - try to get relevant text directly from the item
            for field in ["description", "text", "evidence_text"]:
                if field in item and item[field]:
                    return f"Item text: {item[field][:max_length]}"
            return "No specific evidence text found or linked for this item."
            
        return "\n".join(evidence_texts)

    async def _evaluate_single_item_llm(self,
                                    context: ProcessingContext,
                                    item: Dict[str, Any],
                                    data_type: str) -> Dict[str, Any]:
        """Uses LLM to evaluate a single item based on assessment type and config."""
        assessment_type = context.assessment_type
        
        # Get item name for better prompting (remove trailing 's' if present)
        item_name = data_type.rstrip('s')
        
        # --- Get Config/Guidance ---
        target_definition = context.get_target_definition()
        
        # Get planning data using the enhanced context method
        planning_output = self._get_data(context, "planner", "planning", {})
        planner_guidance = planning_output.get('evaluation_focus', f"Evaluate the {item_name} based on standard criteria.")
        
        base_evaluator_instructions = context.get_workflow_instructions(self.role) or "Evaluate the provided item according to the criteria."

        # --- Define Evaluation Output Schema Dynamically ---
        evaluation_properties = {
            "rationale": {"type": "string", "description": "Brief explanation justifying the evaluation ratings/assignments."}
        }
        required_fields = ["rationale"]

        if assessment_type == "extract":  # Action Items
            evaluation_properties["evaluated_priority"] = {"type": "string", "enum": ["high", "medium", "low"], "description": "Re-assessed priority."}
            evaluation_properties["is_actionable"] = {"type": "boolean", "description": "Is this item clear, specific, and actionable?"}
            evaluation_properties["evaluated_deadline"] = {"type": "string", "description": "Normalized deadline or timeframe if present."}
            required_fields.extend(["evaluated_priority", "is_actionable"])
        elif assessment_type == "assess":  # Issues
            evaluation_properties["evaluated_severity"] = {"type": "string", "enum": ["critical", "high", "medium", "low"], "description": "Re-assessed severity based on evidence and impact."}
            evaluation_properties["potential_impact"] = {"type": "string", "description": "Summarized potential impact if not addressed."}
            evaluation_properties["suggested_recommendations"] = {"type": "array", "items": {"type": "string"}, "description": "1-2 brief suggestions for addressing the issue."}
            required_fields.extend(["evaluated_severity", "potential_impact"])
        elif assessment_type == "distill":  # Key Points
            evaluation_properties["evaluated_importance"] = {"type": "string", "enum": ["High", "Medium", "Low"], "description": "Re-assessed importance of the key point."}
            evaluation_properties["topic_validation"] = {"type": "string", "description": "Does the assigned topic seem correct?"}
            required_fields.extend(["evaluated_importance"])
        elif assessment_type == "analyze":  # Evidence Snippets
            rating_scale = context.get_rating_scale()
            if rating_scale and rating_scale.get("levels"):
                level_values = [level.get("value") for level in rating_scale["levels"] if level.get("value") is not None]
                evaluation_properties["maturity_rating"] = {
                    "type": "integer" if all(isinstance(v, int) for v in level_values) else "number", 
                    "enum": level_values, 
                    "description": f"Maturity rating ({min(level_values)}-{max(level_values)}) based on this evidence snippet."
                }
                evaluation_properties["rating_rationale"] = {
                    "type": "string", 
                    "description": "Explanation of why this rating was assigned."
                }
                required_fields.extend(["maturity_rating", "rating_rationale"])
            else:
                self._log_warning("Rating scale not found for analyze assessment type.", context)
                # Add fallback properties
                evaluation_properties["maturity_rating"] = {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Maturity rating (1-5) based on this evidence snippet."
                }

        evaluation_schema = {
            "type": "object",
            "properties": evaluation_properties,
            "required": required_fields
        }

        # --- Get Evidence ---
        evidence_text = self._get_evidence_text(context, item)

        # --- Construct Prompt ---
        item_details_json = json.dumps(item, indent=2, default=str)  # Serialize item for prompt

        # Include definitions relevant to the evaluation
        definitions_prompt_section = ""
        if assessment_type == "assess":
            severity_defs = target_definition.get("properties", {}).get("severity", {}).get("descriptions", {})
            definitions_prompt_section = f"**Severity Definitions:**\n{json.dumps(severity_defs, indent=2)}\n"
        elif assessment_type == "analyze":
            rating_scale_defs = context.get_rating_scale()
            definitions_prompt_section = f"**Rating Scale Definitions:**\n{json.dumps(rating_scale_defs, indent=2)}\n"

        prompt = f"""
You are an expert AI 'Evaluator' agent. Your task is to assess the following '{item_name}' based on its content, associated evidence, and the provided criteria.

**Assessment Type:** {assessment_type}
**Item to Evaluate:**
```json
{item_details_json}
```

**Associated Evidence from Document:**
{evidence_text}

**Base Instructions for Evaluator:** {base_evaluator_instructions}
**Specific Evaluation Focus:** {planner_guidance or f"Apply standard evaluation criteria for this {item_name}."}
{definitions_prompt_section}

**Your Task:**
Carefully review the item and its evidence. Provide your evaluation by filling out the requested fields accurately and concisely. Justify your assessment in the 'rationale'.

**Output Format:** Respond only with a valid JSON object matching this schema:
```json
{json.dumps(evaluation_schema, indent=2)}
```
"""

        # --- LLM Call ---
        evaluation_result = await self._generate_structured(
            prompt=prompt,
            output_schema=evaluation_schema,
            context=context,
            temperature=self.options.get("evaluator_temperature", 0.2)
        )

        # --- Merge Results ---
        # Create a new dictionary with the original item and evaluation results
        evaluated_item = copy.deepcopy(item)
        evaluated_item.update(evaluation_result)  # Add evaluation fields
        
        # Add timestamp
        evaluated_item["evaluated_at"] = context.metadata.get("current_stage")
        
        self._log_debug(f"Evaluation completed for item {item.get('id','N/A')}.", context)
        return evaluated_item

    async def _create_overall_assessment_llm(self,
                                        context: ProcessingContext,
                                        evaluated_items: List[Dict[str, Any]],
                                        data_type: str) -> Dict[str, Any]:
        """Generates an overall assessment summary using the LLM."""
        if not evaluated_items:
            self._log_warning("Cannot create overall assessment: No evaluated items provided.", context)
            return {"summary": "No items were available for overall assessment."}

        self._log_info(f"Generating overall assessment based on {len(evaluated_items)} evaluated {data_type}.", context)
        assessment_type = context.assessment_type
        
        # --- Prepare Data for Prompt ---
        # Summarize evaluated items concisely for the prompt (to save tokens)
        summary_items = []
        max_items_in_prompt = self.options.get("overall_assessment_max_items", 25)
        item_sample = evaluated_items[:max_items_in_prompt]  # Take a sample if too many

        for item in item_sample:
            summary = {"id": item.get("id")}
            if assessment_type == "assess":
                summary.update({k: item.get(k) for k in ["title", "evaluated_severity", "potential_impact"] if k in item})
            elif assessment_type == "analyze":
                summary.update({k: item.get(k) for k in ["dimension", "criteria", "maturity_rating", "rating_rationale"] if k in item})
            elif assessment_type == "extract":
                summary.update({k: item.get(k) for k in ["description", "owner", "evaluated_priority"] if k in item})
            elif assessment_type == "distill":
                summary.update({k: item.get(k) for k in ["text", "evaluated_importance", "topic"] if k in item})

            summary_items.append(summary)

        # Calculate statistics for better analysis
        stats = self._calculate_item_statistics(evaluated_items, assessment_type)

        # --- Define Output Schema ---
        overall_schema = {
            "type": "object",
            "properties": {
                "executive_summary": {"type": "string", "description": "High-level overview (3-5 sentences) of the main findings and overall status."},
                "key_findings": {"type": "array", "items": {"type": "string"}, "description": "3-5 bullet points highlighting the most critical evaluated items or trends."},
                "thematic_analysis": {"type": "string", "description": "Analysis of common themes, patterns, or critical areas observed across the evaluated items."},
                "strategic_recommendations": {"type": "array", "items": {"type": "string"}, "description": "3-5 actionable, high-level recommendations based on the overall evaluation."},
                "overall_status_rating": {"type": "string", "description": "A single qualitative rating summarizing the overall state (e.g., Critical, Concerning, Moderate, Stable, Strong)"}
            },
            "required": ["executive_summary", "key_findings", "thematic_analysis", "strategic_recommendations", "overall_status_rating"]
        }

        # --- Construct Prompt ---
        prompt = f"""
You are an expert AI 'Evaluator' synthesizing an overall assessment based on previously evaluated items from a document analysis.

Assessment Type: {assessment_type}
Number of Evaluated Items: {len(evaluated_items)} (Showing sample of up to {max_items_in_prompt})

Summary of Evaluated {data_type}:
```json
{json.dumps(summary_items, indent=2, default=str)}
```

Statistics:
```json
{json.dumps(stats, indent=2)}
```

Your Task:
Based on the provided summary of evaluated items and statistics, generate a concise overall assessment. Provide insights beyond just listing the items. Identify patterns, critical areas, and strategic implications.

Output Format: Respond only with a valid JSON object matching this schema:
```json
{json.dumps(overall_schema, indent=2)}
```
"""

        # --- LLM Call ---
        overall_assessment_result = await self._generate_structured(
            prompt=prompt,
            output_schema=overall_schema,
            context=context,
            temperature=self.options.get("overall_assessment_temperature", 0.4),
            max_tokens=self.options.get("overall_assessment_max_tokens", 1500)
        )

        return overall_assessment_result

    def _calculate_item_statistics(self, items: List[Dict[str, Any]], assessment_type: str) -> Dict[str, Any]:
        """Calculate statistics on evaluated items to help with overall assessment."""
        stats = {}
        
        if assessment_type == "assess":
            # Count issues by severity
            severity_counts = {}
            category_counts = {}
            for item in items:
                severity = item.get("evaluated_severity", item.get("severity", "unknown"))
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                category = item.get("category", "uncategorized")
                category_counts[category] = category_counts.get(category, 0) + 1
                
            stats["severity_counts"] = severity_counts
            stats["category_counts"] = category_counts
            
            # Calculate critical+high percentage
            total = sum(severity_counts.values())
            if total > 0:
                critical_high = severity_counts.get("critical", 0) + severity_counts.get("high", 0)
                stats["critical_high_percentage"] = round((critical_high / total) * 100, 1)
            
        elif assessment_type == "extract":
            # Count action items by priority and owner
            priority_counts = {}
            owner_counts = {}
            has_dates = 0
            for item in items:
                priority = item.get("evaluated_priority", item.get("priority", "unknown"))
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
                
                owner = item.get("owner", "unassigned")
                owner_counts[owner] = owner_counts.get(owner, 0) + 1
                
                # Count items with dates
                if item.get("due_date") or item.get("evaluated_deadline"):
                    has_dates += 1
                
            stats["priority_counts"] = priority_counts
            stats["owner_counts"] = owner_counts
            stats["items_with_dates"] = has_dates
            stats["items_with_dates_percentage"] = round((has_dates / len(items)) * 100, 1) if items else 0
            
        elif assessment_type == "distill":
            # Count key points by importance and topic
            importance_counts = {}
            topic_counts = {}
            for item in items:
                importance = item.get("evaluated_importance", item.get("importance", "unknown"))
                importance_counts[importance] = importance_counts.get(importance, 0) + 1
                
                topic = item.get("topic", "uncategorized")
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
                
            stats["importance_counts"] = importance_counts
            stats["topic_counts"] = topic_counts
            
        elif assessment_type == "analyze":
            # Analyze maturity ratings across dimensions/criteria
            dimension_ratings = {}
            for item in items:
                dimension = item.get("dimension", "unknown")
                rating = item.get("maturity_rating")
                
                if rating is not None:
                    if dimension not in dimension_ratings:
                        dimension_ratings[dimension] = {"ratings": [], "average": None}
                    
                    dimension_ratings[dimension]["ratings"].append(rating)
            
            # Calculate averages
            for dimension, data in dimension_ratings.items():
                ratings = data["ratings"]
                if ratings:
                    data["average"] = sum(ratings) / len(ratings)
                    data["min"] = min(ratings)
                    data["max"] = max(ratings)
                    data["count"] = len(ratings)
                    
            stats["dimension_ratings"] = dimension_ratings
                    
        return stats
