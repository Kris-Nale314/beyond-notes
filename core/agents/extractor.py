# core/agents/extractor.py
import logging
import uuid
from typing import Dict, Any, List, Optional

# Import base class and context/LLM types
from .base import BaseAgent
from core.llm.customllm import CustomLLM
from core.models.context import ProcessingContext

class ExtractorAgent(BaseAgent):
    """
    Extracts relevant information from document chunks based on the assessment
    configuration and planner guidance.
    
    Implements a "go wide" strategy to capture as much potential information as possible,
    which can be refined by downstream agents.
    """

    def __init__(self, llm: CustomLLM, options: Optional[Dict[str, Any]] = None):
        """Initialize the ExtractorAgent."""
        super().__init__(llm, options or {})
        self.role = "extractor"
        self.logger = logging.getLogger(f"core.agents.{self.name}")
        self.logger.info(f"ExtractorAgent initialized")

    async def process(self, context: ProcessingContext) -> Optional[Dict[str, Any]]:
        """
        Iterates through document chunks, extracts relevant information using the LLM,
        and updates the context with extracted items and evidence.

        Args:
            context: The shared ProcessingContext object.

        Returns:
            A dictionary summarizing the extraction results (e.g., counts).
        """
        self._log_info(f"Starting extraction phase for assessment: '{context.display_name}'", context)

        # --- Get Configuration & Guidance ---
        assessment_type = context.assessment_type
        extraction_instructions = context.get_workflow_instructions(self.role)
        target_definition = context.get_target_definition()
        extraction_criteria = context.get_extraction_criteria() or {}
        
        # Get planning data for focused extraction
        planning_output = self._get_data(context, "planner", "planning", {})
        planner_extraction_focus = planning_output.get('extraction_focus', '')
        document_type = planning_output.get('document_type', 'Unknown')

        if not target_definition:
            raise ValueError(f"ExtractorAgent cannot proceed without a target definition for type '{assessment_type}'.")

        # --- Determine what to extract based on assessment type ---
        extraction_target_name = target_definition.get("name", assessment_type)
        items_key = self._get_items_key_for_assessment_type(assessment_type)
        item_schema = self._create_extraction_schema_for_type(assessment_type, target_definition)

        # --- Define Schema for LLM's Structured Output ---
        output_schema = {
            "type": "object",
            "properties": {
                items_key: {
                    "type": "array",
                    "description": f"List of {extraction_target_name} items extracted from the text chunk.",
                    "items": item_schema
                }
            },
            "required": [items_key]
        }

        # --- Process Chunks ---
        chunks = context.chunks
        total_chunks = len(chunks)
        items_extracted_count = 0
        chunks_failed = 0
        all_extracted_items = []  # Collects all extracted items across chunks

        self._log_info(f"Processing {total_chunks} chunks for extraction...", context)

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
                items_key=items_key,
                extraction_instructions=extraction_instructions,
                planner_guidance=planner_extraction_focus,
                document_type=document_type,
                extraction_criteria=extraction_criteria
            )
            
            # --- LLM Call for Extraction ---
            try:
                # Use the structured output helper from BaseAgent
                extraction_result = await self._generate_structured(
                    prompt=prompt,
                    output_schema=output_schema,
                    context=context,
                    temperature=self.options.get("extractor_temperature", 0.1)
                )

                extracted_items = extraction_result.get(items_key, [])
                if not isinstance(extracted_items, list):
                    self._log_warning(f"LLM response for chunk {chunk_index} did not contain a valid list under '{items_key}'. Found type: {type(extracted_items)}", context)
                    extracted_items = []

                self._log_debug(f"Chunk {chunk_index}: Extracted {len(extracted_items)} potential items.", context)

                # --- Process Extracted Items ---
                for item_data in extracted_items:
                    if not isinstance(item_data, dict):
                        self._log_warning(f"Skipping invalid item data (not a dict): {item_data}", context)
                        continue

                    # Add chunk metadata and unique ID to the item
                    item_data["chunk_index"] = chunk_index
                    item_id = f"{items_key[:-1]}-{uuid.uuid4().hex[:8]}"  # Generate a unique ID
                    item_data["id"] = item_id
                    
                    # Store evidence text for linking
                    self._add_evidence_for_item(context, assessment_type, item_data, chunk_index)
                    
                    # Add item to collection
                    all_extracted_items.append(item_data)
                    items_extracted_count += 1

            except Exception as chunk_e:
                chunks_failed += 1
                self._log_error(f"Failed to process chunk {chunk_index}: {str(chunk_e)}", context, exc_info=True)
                context.add_warning(f"Extraction failed for chunk {chunk_index}.", "extraction")
                # Continue to the next chunk

            # Update progress after processing each chunk
            context.update_stage_progress((i + 1) / total_chunks, f"Processed chunk {i+1}/{total_chunks}")

        # --- Store All Extracted Items in Context ---
        self._store_data(context, items_key, all_extracted_items)
        self._log_info(f"Stored {items_extracted_count} extracted {items_key} in context", context)

        # --- Final Summary ---
        summary = {
            "total_chunks_processed": total_chunks - chunks_failed,
            "chunks_with_errors": chunks_failed,
            f"total_{items_key}_found": items_extracted_count
        }
        
        self._log_info(f"Extraction phase complete. Found {items_extracted_count} items. {chunks_failed} chunks failed.", context)
        return summary

    def _get_items_key_for_assessment_type(self, assessment_type: str) -> str:
        """Get the appropriate list key for storing extracted items based on assessment type."""
        items_key_map = {
            "extract": "action_items",
            "assess": "issues",
            "distill": "key_points",
            "analyze": "evidence"
        }
        return items_key_map.get(assessment_type, "extracted_items")

    def _create_extraction_schema_for_type(self, assessment_type: str, target_definition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a schema for the items to be extracted based on assessment type.
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
        
        # Default generic schema for unknown assessment types
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
                               items_key: str,
                               extraction_instructions: Optional[str],
                               planner_guidance: str,
                               document_type: str,
                               extraction_criteria: Dict[str, Any]) -> str:
        """Build the extraction prompt for a specific chunk."""
        # Get property names from schema for the prompt
        # Get default instructions based on assessment type if none provided
        if not extraction_instructions:
            if assessment_type == "extract":
                extraction_instructions = "Extract all action items mentioned in the text. Cast a wide net to identify both explicit and implied action items."
            elif assessment_type == "assess":
                extraction_instructions = "Identify all potential issues, risks, or problems mentioned in the text. Be thorough and err on the side of including borderline issues."
            elif assessment_type == "distill":
                extraction_instructions = "Extract key points, decisions, facts and insights from the text. Focus on the most important information."
            elif assessment_type == "analyze":
                extraction_instructions = "Extract evidence related to the framework dimensions and criteria. Look for both positive and negative evidence."
            else:
                extraction_instructions = f"Extract all relevant {extraction_target_name} items from the text."
        
        # Extract indicators if available
        indicators = extraction_criteria.get("indicators", [])
        indicators_text = "\n".join([f"- {indicator}" for indicator in indicators]) if indicators else "N/A"
        
        # Get target properties description
        properties = target_definition.get("properties", {})
        properties_text = ""
        if properties:
            for prop_name, prop_details in properties.items():
                prop_desc = prop_details.get("description", "")
                prop_required = "Required" if prop_name in target_definition.get("required", []) else "Optional"
                properties_text += f"- {prop_name}: {prop_desc} ({prop_required})\n"
        else:
            properties_text = "Capture all relevant information about each item."
        
        # Build the prompt
        prompt = f"""
You are an AI 'Extractor' agent. Your task is to carefully read the following text chunk and extract all instances of '{extraction_target_name}' based on the provided definitions and criteria.

**Assessment Type:** {assessment_type}
**Document Type:** {document_type}
**Target to Extract:** {extraction_target_name} ({target_definition.get('description', '')})

**Properties to Capture:**
{properties_text}

**Extraction Instructions:** {extraction_instructions}

**Planner's Guidance:** {planner_guidance}

**Extraction Indicators (patterns to look for):**
{indicators_text}

**Text Chunk (Chunk {chunk_index}):**
{chunk_text}

**Important Guidance:**
- GO WIDE: It's better to extract too many potential items than to miss important ones. Later agents will refine and evaluate.
- Be thorough in identifying ALL possible instances of '{extraction_target_name}' in this text chunk.
- Extract both explicit and implicit items.
- Provide as much detail as possible for each property.
- Each item should have all required properties and as many optional properties as you can identify.

**Output Format:** Respond with a valid JSON object containing ONLY a single key named "{items_key}", which holds a list of extracted items. Each item should contain the properties listed above. If no items are found, return an empty list.
"""
        return prompt

    def _add_evidence_for_item(self, context: ProcessingContext, assessment_type: str, item_data: Dict[str, Any], chunk_index: int) -> None:
        """Add evidence for an extracted item based on assessment type."""
        item_id = item_data.get("id")
        if not item_id:
            return
            
        # Determine evidence text based on assessment type
        evidence_text = None
        confidence = None
        
        if assessment_type == "extract":
            # For action items, use description as evidence
            evidence_text = item_data.get("description")
            
        elif assessment_type == "assess":
            # For issues, use title + description as evidence
            title = item_data.get("title", "")
            description = item_data.get("description", "")
            evidence_text = f"{title}: {description}" if title else description
            
        elif assessment_type == "distill":
            # For key points, use the text field
            evidence_text = item_data.get("text")
            
        elif assessment_type == "analyze":
            # For evidence snippets, use the evidence_text field
            evidence_text = item_data.get("evidence_text")
            
        # Add evidence if we have text
        if evidence_text and item_id:
            try:
                # Add evidence with confidence if available
                if "confidence" in item_data:
                    confidence = item_data.get("confidence")
                
                self._add_evidence(context, item_id, evidence_text, chunk_index, confidence)
            except Exception as e:
                self._log_warning(f"Failed to add evidence for item {item_id}: {str(e)}", context)