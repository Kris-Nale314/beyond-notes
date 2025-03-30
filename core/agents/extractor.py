# core/agents/extractor.py
from typing import Dict, Any, List, Optional
import json
import asyncio

from core.agents.base import Agent
from core.models.context import ProcessingContext

class ExtractorAgent(Agent):
    """
    The Extractor Agent examines document chunks to identify relevant information.
    
    Responsibilities:
    - Extract key information from document chunks
    - Follow focused instructions from the planner
    - Maintain source traceability for extracted information
    - Identify entities, action items, issues, etc.
    """
    
    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Extract information from document chunks.
        
        Args:
            context: The shared processing context
            
        Returns:
            Extraction results for all chunks
        """
        self.log_info(context, "Starting information extraction")
        
        try:
            # Get chunks from context
            chunks = context.chunks
            
            if not chunks:
                self.log_error(context, "No chunks available for extraction")
                raise ValueError("No chunks available for extraction")
            
            # Get planning information if available
            planning_results = context.results.get("planning", {})
            document_type = planning_results.get("document_type", "unknown")
            extraction_instructions = planning_results.get("extraction_instructions", {})
            focus_areas = planning_results.get("focus_areas", [])
            
            # Get custom instructions from assessment config
            agent_instructions = self.get_agent_instructions(context)
            
            # Process chunks in batches
            extraction_results = await self.process_in_batches(
                context=context,
                items=chunks,
                process_func=lambda chunk, i: self._extract_from_chunk(
                    context, chunk, i, document_type, extraction_instructions, focus_areas, agent_instructions
                ),
                batch_size=3,  # Process 3 chunks at a time to avoid rate limits
                delay=0.5  # Small delay between batches
            )
            
            # Compile extraction statistics
            stats = {
                "total_chunks_processed": len(extraction_results),
                "successful_extractions": sum(1 for r in extraction_results if "error" not in r),
                "issues_found": sum(len(r.get("issues", [])) for r in extraction_results if "error" not in r),
                "entities_found": sum(len(r.get("entities", [])) for r in extraction_results if "error" not in r),
                "key_points_found": sum(len(r.get("key_points", [])) for r in extraction_results if "error" not in r)
            }
            
            # Complete the stage
            extraction_output = {
                "chunk_extractions": extraction_results,
                "statistics": stats
            }
            
            self.log_info(context, f"Extraction complete. Found {stats['issues_found']} issues across {len(chunks)} chunks")
            
            return extraction_output
            
        except Exception as e:
            error_message = f"Extraction failed: {str(e)}"
            self.log_error(context, error_message)
            raise
    
    async def _extract_from_chunk(self, 
                                context: ProcessingContext,
                                chunk: Dict[str, Any], 
                                chunk_index: int,
                                document_type: str,
                                extraction_instructions: Dict[str, Any],
                                focus_areas: List[str],
                                agent_instructions: str) -> Dict[str, Any]:
        """
        Extract information from a single chunk.
        
        Args:
            context: The processing context
            chunk: The document chunk to process
            chunk_index: Index of the chunk
            document_type: Type of the document
            extraction_instructions: Instructions from the planner
            focus_areas: Focus areas from the planner
            agent_instructions: Instructions from the assessment config
            
        Returns:
            Extraction results for this chunk
        """
        # Build a schema for the extracted information
        json_schema = {
            "type": "object",
            "properties": {
                "chunk_index": {"type": "integer"},
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "severity": {"type": "string"},
                            "category": {"type": "string"},
                            "source_text": {"type": "string"}
                        },
                        "required": ["description"]
                    }
                },
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                            "mentions": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["name", "type"]
                    }
                },
                "key_points": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["chunk_index", "issues", "entities", "key_points"]
        }
        
        # Get extraction parameters from instructions
        key_entities = extraction_instructions.get("key_entities", [])
        important_sections = extraction_instructions.get("important_sections", [])
        data_points = extraction_instructions.get("data_points", [])
        
        # Build extraction prompt
        prompt = f"""
        You are an expert information extractor for an AI document analysis system.
        
        Please analyze the following document chunk ({chunk_index+1}) and extract relevant information.
        
        Document Type: {document_type}
        Focus Areas: {', '.join(focus_areas) if focus_areas else 'All relevant information'}
        
        {agent_instructions}
        
        Special Instructions:
        {', '.join(f"Look for {entity}" for entity in key_entities) if key_entities else ""}
        {', '.join(f"Pay special attention to {section}" for section in important_sections) if important_sections else ""}
        {', '.join(f"Extract {point}" for point in data_points) if data_points else ""}
        
        CHUNK TEXT:
        {chunk['text']}
        
        Extract the following information:
        1. Issues: Any problems, challenges, risks, or concerns mentioned
           - Provide a detailed description of each issue
           - Rate severity (Critical, High, Medium, Low)
           - Categorize the issue (Technical, Process, Resource, Quality, Risk, Compliance)
           - Include the source text where you found this issue
        
        2. Entities: People, teams, products, or systems mentioned
           - Include entity type (Person, Team, Product, System, etc.)
           - Include all mentions of this entity
        
        3. Key Points: Important information or insights (2-5 items)
        
        Format your response as a structured JSON object.
        """
        
        try:
            # Extract information from the chunk
            extraction = await self.generate_structured_output(
                prompt=prompt,
                json_schema=json_schema,
                max_tokens=2000,
                temperature=0.3
            )
            
            # Ensure the chunk_index is correct
            extraction["chunk_index"] = chunk_index
            
            # Add chunk metadata
            extraction["word_count"] = chunk.get("word_count", 0)
            extraction["start_position"] = chunk.get("start_position", 0)
            extraction["end_position"] = chunk.get("end_position", 0)
            
            # Add extractions to context
            self._add_extractions_to_context(context, extraction)
            
            return extraction
            
        except Exception as e:
            return {
                "chunk_index": chunk_index,
                "error": str(e),
                "issues": [],
                "entities": [],
                "key_points": []
            }
    
    def _add_extractions_to_context(self, context: ProcessingContext, extraction: Dict[str, Any]) -> None:
        """
        Add extracted elements to the processing context.
        
        Args:
            context: The processing context
            extraction: Extraction results for a chunk
        """
        chunk_index = extraction.get("chunk_index", -1)
        source_info = {"chunk_index": chunk_index}
        
        # Add issues
        for issue in extraction.get("issues", []):
            issue_data = {
                "description": issue.get("description", ""),
                "severity": issue.get("severity", "medium"),
                "category": issue.get("category", "unknown"),
                "source_chunks": [chunk_index]
            }
            
            # Add issue to context
            issue_id = context.add_issue(issue_data)
            
            # Add source reference if provided
            source_text = issue.get("source_text")
            if source_text:
                context.add_evidence(issue_id, source_text, source_info)
        
        # Add entities
        for entity in extraction.get("entities", []):
            entity_data = {
                "name": entity.get("name", ""),
                "type": entity.get("type", "unknown"),
                "mentions": entity.get("mentions", []),
                "source_chunks": [chunk_index]
            }
            
            # Add entity to context
            context.add_entity(entity_data)
        
        # Add key points
        for point in extraction.get("key_points", []):
            point_data = {
                "text": point,
                "source_chunks": [chunk_index]
            }
            
            # Add key point to context
            context.add_key_point(point_data)