# core/agents/extractor.py
import logging
import json
import uuid
from typing import Dict, Any, List, Optional

# Import base class and context/LLM types
from .base import BaseAgent
from core.llm.customllm import CustomLLM
from core.models.context import ProcessingContext

class ExtractorAgent(BaseAgent):
    """
    Extracts relevant information from document chunks based on the assessment
    configuration, planner guidance, and extraction criteria.
    
    Uses robust schema handling to ensure compatibility with the LLM function calling API.
    """

    def __init__(self, llm: CustomLLM, options: Optional[Dict[str, Any]] = None):
        """Initialize the ExtractorAgent."""
        super().__init__(llm, options)
        self.role = "extractor"
        self.name = "ExtractorAgent"
        self.logger = logging.getLogger(f"core.agents.{self.name}")
        self.logger.info(f"ExtractorAgent initialized.")

    async def process(self, context: ProcessingContext) -> Optional[Dict[str, Any]]:
        """
        Iterates through document chunks, extracts relevant information using the LLM,
        and updates the context with extracted items and evidence.

        Args:
            context: The shared ProcessingContext object.

        Returns:
            A dictionary summarizing the extraction results (e.g., counts), or None.
        """
        self._log_info(f"Starting extraction phase for assessment: '{context.display_name}'", context)

        # --- Get Config & Guidance from Context ---
        assessment_type = context.assessment_type
        base_extractor_instructions = context.get_workflow_instructions(self.role)
        target_definition = context.get_target_definition()
        extraction_criteria = context.get_extraction_criteria() or {}
        
        # Get planning data using the enhanced context method
        planning_output = self._get_data(context, "planner", "planning", {})
        planner_extraction_focus = planning_output.get('extraction_focus', '')

        if not target_definition:
            raise ValueError(f"ExtractorAgent cannot proceed without a target definition for type '{assessment_type}'.")

        # --- Determine what to extract based on type ---
        extraction_target_name = target_definition.get("name", assessment_type)
        list_key_for_results = self._get_list_key_for_assessment_type(assessment_type)
        item_schema = self._create_extraction_schema_for_type(assessment_type, target_definition)

        # --- Define the Schema for LLM's Structured Output ---
        llm_output_schema = {
            "type": "object",
            "properties": {
                list_key_for_results: {
                    "type": "array",
                    "description": f"List of {extraction_target_name} items extracted from the text chunk.",
                    "items": item_schema
                }
            },
            "required": [list_key_for_results]
        }

        # --- Process Chunks ---
        chunks = context.chunks
        total_chunks = len(chunks)
        items_extracted_count = 0
        chunks_failed = 0
        all_extracted_items = []  # Collects all extracted items

        self._log_info(f"Starting extraction from {total_chunks} chunks...", context)

        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "")
            chunk_index = chunk.get("chunk_index", i)
            
            if not chunk_text.strip():
                self._log_debug(f"Skipping empty chunk {chunk_index}.", context)
                context.update_stage_progress((i + 1) / total_chunks)  # Update progress even for skipped chunk
                continue

            self._log_debug(f"Processing chunk {chunk_index}/{total_chunks}...", context)

            # --- Construct Prompt for this Chunk ---
            prompt = self._build_extraction_prompt(
                chunk_text=chunk_text,
                chunk_index=chunk_index,
                assessment_type=assessment_type,
                extraction_target_name=extraction_target_name,
                target_definition=target_definition,
                list_key_for_results=list_key_for_results,
                item_schema=item_schema,
                base_instructions=base_extractor_instructions,
                planner_guidance=planner_extraction_focus,
                extraction_criteria=extraction_criteria
            )
            
            # --- LLM Call for Extraction ---
            try:
                # Use the structured output helper from BaseAgent
                extraction_result = await self._generate_structured(
                    prompt=prompt,
                    output_schema=llm_output_schema,
                    context=context,
                    temperature=self.options.get("extractor_temperature", 0.1)
                )

                extracted_items = extraction_result.get(list_key_for_results, [])
                if not isinstance(extracted_items, list):
                    self._log_warning(f"LLM response for chunk {chunk_index} did not contain a valid list under '{list_key_for_results}'. Found type: {type(extracted_items)}.", context)
                    extracted_items = []

                self._log_debug(f"Chunk {chunk_index}: Extracted {len(extracted_items)} potential items.", context)

                # --- Process Extracted Items ---
                for item_data in extracted_items:
                    if not isinstance(item_data, dict):
                        self._log_warning(f"Skipping invalid item data (not a dict) at index {i}: {item_data}", context)
                        continue

                    # Add chunk metadata to the item
                    item_data["chunk_index"] = chunk_index
                    item_data["id"] = f"{list_key_for_results[:-1]}-{uuid.uuid4().hex[:8]}"  # Generate a unique ID
                    
                    # Store evidence text for linking
                    self._add_evidence_for_item(context, assessment_type, item_data, chunk_index)
                    
                    # Add item to collection
                    all_extracted_items.append(item_data)
                    items_extracted_count += 1

            except Exception as chunk_e:
                chunks_failed += 1
                self._log_error(f"Failed to process chunk {chunk_index}: {str(chunk_e)}", context, exc_info=True)
                context.add_warning(f"Extraction failed for chunk {chunk_index}.", stage=self.role)
                # Continue to the next chunk

            # Update progress after processing each chunk
            context.update_stage_progress((i + 1) / total_chunks, f"Processed chunk {i+1}/{total_chunks}")

        # --- Store All Extracted Items in Context ---
        self._store_data(context, list_key_for_results, all_extracted_items)
        self._log_info(f"Stored {items_extracted_count} extracted {list_key_for_results} in context", context)

        # --- Final Summary ---
        summary = {
            "total_chunks_processed": total_chunks - chunks_failed,
            "chunks_with_errors": chunks_failed,
            f"total_{list_key_for_results}_found": items_extracted_count
        }
        
        self._log_info(f"Extraction phase complete. Found {items_extracted_count} items. {chunks_failed} chunks failed.", context)
        return summary

    def _get_list_key_for_assessment_type(self, assessment_type: str) -> str:
        """Get the appropriate list key for storing extracted items based on assessment type."""
        list_key_map = {
            "extract": "action_items",
            "assess": "issues",
            "distill": "key_points",
            "analyze": "evidence"
        }
        return list_key_map.get(assessment_type, "extracted_items")

    def _create_extraction_schema_for_type(self, assessment_type: str, target_definition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a schema for the items to be extracted based on assessment type.
        This method ensures the schema is compatible with OpenAI's function calling API.
        """
        if assessment_type == "extract":  # Action Items
            return {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "The action item description"
                    },
                    "owner": {
                        "type": "string",
                        "description": "Person or team responsible for the action item"
                    },
                    "due_date": {
                        "type": "string",
                        "description": "Due date or timeframe for the action item"
                    },
                    "priority": {
                        "type": "string",
                        "description": "Priority level (high, medium, low)"
                    }
                },
                "required": ["description"]
            }
            
        elif assessment_type == "assess":  # Issues
            return {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Brief title of the issue"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the issue"
                    },
                    "severity": {
                        "type": "string",
                        "description": "Severity level of the issue (critical, high, medium, low)"
                    },
                    "category": {
                        "type": "string",
                        "description": "Category of the issue (e.g., technical, process, resource)"
                    },
                    "impact": {
                        "type": "string",
                        "description": "Description of the potential impact of this issue"
                    }
                },
                "required": ["title", "description"]
            }
            
        elif assessment_type == "distill":  # Key Points
            return {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The extracted key point or statement"
                    },
                    "point_type": {
                        "type": "string",
                        "description": "Type of point (e.g., Fact, Decision, Question, Insight, Quote)"
                    },
                    "topic": {
                        "type": "string",
                        "description": "Main topic this point relates to"
                    },
                    "importance": {
                        "type": "string",
                        "description": "Assessed importance (High, Medium, Low)"
                    }
                },
                "required": ["text"]
            }
            
        elif assessment_type == "analyze":  # Framework Evidence
            return {
                "type": "object",
                "properties": {
                    "dimension": {
                        "type": "string",
                        "description": "The framework dimension the evidence relates to"
                    },
                    "criteria": {
                        "type": "string",
                        "description": "The specific criteria within the dimension"
                    },
                    "evidence_text": {
                        "type": "string",
                        "description": "The exact text snippet from the document providing evidence"
                    },
                    "commentary": {
                        "type": "string",
                        "description": "Brief explanation of why this text is evidence for the criteria"
                    }
                },
                "required": ["dimension", "criteria", "evidence_text"]
            }
        
        # Default generic schema if assessment type is not recognized
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string", 
                    "description": "The extracted text"
                },
                "type": {
                    "type": "string",
                    "description": "The type of the extracted item"
                }
            },
            "required": ["text"]
        }

    def _build_extraction_prompt(self, 
                               chunk_text: str,
                               chunk_index: int,
                               assessment_type: str,
                               extraction_target_name: str,
                               target_definition: Dict[str, Any],
                               list_key_for_results: str,
                               item_schema: Dict[str, Any],
                               base_instructions: Optional[str],
                               planner_guidance: str,
                               extraction_criteria: Dict[str, Any]) -> str:
        """Build the extraction prompt for a specific chunk."""
        # Get property names from schema for the prompt
        properties = list(item_schema.get("properties", {}).keys())
        property_names_str = ", ".join(properties)
        
        # Get default instructions based on assessment type if none provided
        if not base_instructions:
            if assessment_type == "extract":
                base_instructions = "Extract all action items mentioned in the text. Be thorough and identify both explicit and implied action items."
            elif assessment_type == "assess":
                base_instructions = "Identify all potential issues, risks, or problems mentioned in the text. Be thorough and cast a wide net."
            elif assessment_type == "distill":
                base_instructions = "Extract key points, decisions, facts and insights from the text. Focus on the most important information."
            elif assessment_type == "analyze":
                base_instructions = "Extract evidence related to the framework dimensions and criteria. Look for both positive and negative evidence."
            else:
                base_instructions = f"Extract all relevant {extraction_target_name} items from the text."
        
        # Build the prompt
        prompt = f"""
You are an AI 'Extractor' agent. Your task is to carefully read the following text chunk and extract all instances of '{extraction_target_name}' based on the provided definitions and criteria.

**Assessment Type:** {assessment_type}
**Target to Extract:** {extraction_target_name} ({target_definition.get('description', '')})
**Properties to Extract for each item:** {property_names_str}

**Base Instructions:** {base_instructions}
**Planner Guidance:** {planner_guidance or "Extract all relevant items."}
**Extraction Criteria:** {json.dumps(extraction_criteria.get('indicators', {}), indent=2) if extraction_criteria.get('indicators') else "N/A"}

**Text Chunk (Chunk {chunk_index}):**
{chunk_text}

**Your Task:**
Analyze the text chunk and extract all relevant '{extraction_target_name}' items. For each item found, provide the requested properties.

BE THOROUGH - it's better to extract too many items than to miss important ones. Later stages will refine and evaluate the extracted items.

**Output Format:** Respond with a valid JSON object containing a single key named "{list_key_for_results}", which holds a list of extracted items. Each item should contain the properties listed above. If no items are found, return an empty list.
"""
        return prompt

    def _add_evidence_for_item(self, context: ProcessingContext, assessment_type: str, item_data: Dict[str, Any], chunk_index: int) -> None:
        """Add evidence for an extracted item based on assessment type."""
        item_id = item_data.get("id")
        if not item_id:
            return
            
        # Determine evidence text based on assessment type
        evidence_text = None
        
        if assessment_type == "extract" or assessment_type == "assess":
            # For action items and issues, use description as evidence
            evidence_text = item_data.get("description")
            
        elif assessment_type == "distill":
            # For key points, use the text field
            evidence_text = item_data.get("text")
            
        elif assessment_type == "analyze":
            # For evidence snippets, use the evidence_text field
            evidence_text = item_data.get("evidence_text")
            
        # Add evidence if we have text
        if evidence_text and item_id:
            source_info = {"chunk_index": chunk_index}
            try:
                context.add_evidence(item_id, evidence_text, source_info)
            except Exception as e:
                self._log_warning(f"Failed to add evidence for item {item_id}: {str(e)}", context)