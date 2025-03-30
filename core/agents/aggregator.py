# core/agents/aggregator.py
from typing import Dict, Any, List, Optional
import json
import asyncio

from core.agents.base import Agent
from core.models.context import ProcessingContext

class AggregatorAgent(Agent):
    """
    The Aggregator Agent combines findings from different document sections,
    eliminating redundancies and resolving conflicts.
    
    Responsibilities:
    - Merge information from multiple chunks
    - Remove duplicated items and issues
    - Resolve conflicting information
    - Maintain source traceability
    """
    
    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Aggregate extracted information.
        
        Args:
            context: The shared processing context
            
        Returns:
            Aggregated results
        """
        self.log_info(context, "Starting information aggregation")
        
        try:
            # Get extraction results from context
            extraction_results = context.results.get("extraction", {})
            chunk_extractions = extraction_results.get("chunk_extractions", [])
            
            if not chunk_extractions:
                self.log_error(context, "No extraction results available for aggregation")
                raise ValueError("No extraction results available for aggregation")
            
            # Get custom instructions from assessment config
            agent_instructions = self.get_agent_instructions(context)
            
            # Deduplicate issues
            self.log_info(context, "Deduplicating issues")
            issues = context.extracted_info["issues"]
            deduplicated_issues = await self._deduplicate_issues(context, issues, agent_instructions)
            
            # Update issues in context
            context.extracted_info["issues"] = deduplicated_issues
            
            # Combine and synthesize key points
            self.log_info(context, "Synthesizing key points")
            key_points = context.extracted_info["key_points"]
            synthesized_key_points = await self._synthesize_key_points(context, key_points, agent_instructions)
            
            # Update key points in context
            context.extracted_info["key_points"] = synthesized_key_points
            
            # Create the aggregated output
            aggregation_output = {
                "issues": deduplicated_issues,
                "entities": list(context.entities.values()),
                "key_points": synthesized_key_points,
                "statistics": {
                    "issues_raw": len(issues),
                    "issues_deduplicated": len(deduplicated_issues),
                    "entities": len(context.entities),
                    "key_points_raw": len(key_points),
                    "key_points_synthesized": len(synthesized_key_points)
                }
            }
            
            self.log_info(context, f"Aggregation complete. Found {len(deduplicated_issues)} unique issues")
            
            return aggregation_output
            
        except Exception as e:
            error_message = f"Aggregation failed: {str(e)}"
            self.log_error(context, error_message)
            raise
    
    async def _deduplicate_issues(self, 
                                context: ProcessingContext,
                                issues: List[Dict[str, Any]],
                                agent_instructions: str) -> List[Dict[str, Any]]:
        """
        Deduplicate issues using the LLM.
        
        Args:
            context: The processing context
            issues: List of issues to deduplicate
            agent_instructions: Instructions from the assessment config
            
        Returns:
            List of deduplicated issues
        """
        if not issues:
            return []
            
        if len(issues) <= 1:
            return issues
        
        # Define schema for deduplicated issues
        json_schema = {
            "type": "object",
            "properties": {
                "deduplicated_issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "description": {"type": "string"},
                            "severity": {"type": "string"},
                            "category": {"type": "string"},
                            "source_chunks": {
                                "type": "array",
                                "items": {"type": "integer"}
                            },
                            "related_issue_ids": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["description", "severity", "category", "source_chunks"]
                    }
                }
            },
            "required": ["deduplicated_issues"]
        }
        
        # Get assessment config info
        assessment_config = context.assessment_config
        issue_definition = assessment_config.get("issue_definition", {})
        severity_levels = issue_definition.get("severity_levels", {})
        categories = issue_definition.get("categories", [])
        
        # Create a prompt for deduplication
        prompt = f"""
        You are an expert information aggregator for an AI document analysis system.
        
        I have extracted {len(issues)} issues from different chunks of a document.
        Some of these might be duplicates or refer to the same problem from different parts of the document.
        
        {agent_instructions}
        
        Please deduplicate these issues by:
        1. Combining items that refer to the same underlying problem
        2. Using the most complete description
        3. Using the highest severity rating when there's a conflict
        4. Combining category information
        5. Tracking which chunks each issue came from
        
        Severity Levels:
        {json.dumps(severity_levels, indent=2)}
        
        Categories:
        {json.dumps(categories, indent=2)}
        
        Here are the issues:
        ```
        {json.dumps(issues, indent=2)}
        ```
        
        Create a deduplicated list of issues, where each item includes:
        - id: Preserve the original ID of the main issue you're keeping
        - description: The most complete description of the issue
        - severity: Severity level (use the highest one mentioned)
        - category: Category (use the most specific one)
        - source_chunks: List of chunk indices where this issue was found
        - related_issue_ids: List of IDs of the issues that were merged into this one
        
        Format your response as a structured JSON object with a "deduplicated_issues" array.
        """
        
        # Generate deduplicated issues
        try:
            result = await self.generate_structured_output(
                prompt=prompt,
                json_schema=json_schema,
                max_tokens=3000,
                temperature=0.3
            )
            
            # Get the deduplicated issues
            deduplicated = result.get("deduplicated_issues", [])
            
            # Log the results
            self.log_info(context, f"Deduplicated {len(issues)} issues into {len(deduplicated)} unique issues")
            
            return deduplicated
            
        except Exception as e:
            # If deduplication fails, log the error
            self.log_error(context, f"Error deduplicating issues: {str(e)}")
            
            # Create a basic deduplication by description
            seen_descriptions = {}
            deduplicated = []
            
            for item in issues:
                desc = item.get("description", "").lower()
                if desc and desc not in seen_descriptions:
                    seen_descriptions[desc] = item
                    if "source_chunks" not in item:
                        item["source_chunks"] = [item.get("source_chunks", [])]
                    deduplicated.append(item)
                elif desc:
                    # Add this source chunk to the existing item
                    source_chunks = item.get("source_chunks", [])
                    if source_chunks and isinstance(source_chunks, list):
                        if "source_chunks" not in seen_descriptions[desc]:
                            seen_descriptions[desc]["source_chunks"] = []
                        seen_descriptions[desc]["source_chunks"].extend(source_chunks)
            
            return list(seen_descriptions.values())
    
    async def _synthesize_key_points(self, 
                                    context: ProcessingContext,
                                    key_points: List[Dict[str, Any]],
                                    agent_instructions: str) -> List[Dict[str, Any]]:
        """
        Synthesize key points into a concise list.
        
        Args:
            context: The processing context
            key_points: List of key points to synthesize
            agent_instructions: Instructions from the assessment config
            
        Returns:
            List of synthesized key points
        """
        if not key_points:
            return []
        
        if len(key_points) <= 5:
            return key_points
        
        # Create a list of point texts
        point_texts = []
        for point in key_points:
            if isinstance(point, dict):
                point_texts.append(point.get("text", ""))
            else:
                point_texts.append(str(point))
        
        # Define schema for synthesized key points
        json_schema = {
            "type": "object",
            "properties": {
                "synthesized_points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "importance": {"type": "string"},
                            "related_point_indices": {
                                "type": "array",
                                "items": {"type": "integer"}
                            }
                        },
                        "required": ["text", "importance"]
                    }
                }
            },
            "required": ["synthesized_points"]
        }
        
        # Create a prompt for synthesizing key points
        prompt = f"""
        You are an expert information synthesizer for an AI document analysis system.
        
        I have extracted {len(key_points)} key points from different chunks of a document.
        Your task is to combine related points and create a concise list of the most important insights.
        
        {agent_instructions}
        
        Here are all the extracted key points:
        ```
        {json.dumps(point_texts, indent=2)}
        ```
        
        Please synthesize these into 5-10 comprehensive key points by:
        1. Combining related points into more comprehensive insights
        2. Preserving the most important information
        3. Eliminating redundancies
        4. Creating a concise yet complete list of the document's key insights
        
        For each synthesized point, include:
        - text: The synthesized key point text (comprehensive and clear)
        - importance: A rating of importance (high, medium, low)
        - related_point_indices: Indices of the original points that were combined into this one
        
        Format your response as a structured JSON object with a "synthesized_points" array.
        """
        
        # Generate synthesized key points
        try:
            result = await self.generate_structured_output(
                prompt=prompt,
                json_schema=json_schema,
                max_tokens=2000,
                temperature=0.4
            )
            
            # Get the synthesized points
            synthesized = result.get("synthesized_points", [])
            
            # Log the results
            self.log_info(context, f"Synthesized {len(key_points)} key points into {len(synthesized)} points")
            
            # Convert to the expected format
            formatted_points = []
            for point in synthesized:
                formatted_points.append({
                    "text": point.get("text", ""),
                    "importance": point.get("importance", "medium"),
                    "related_point_indices": point.get("related_point_indices", []),
                    "source_chunks": self._gather_source_chunks(key_points, point.get("related_point_indices", []))
                })
            
            return formatted_points
            
        except Exception as e:
            # If synthesis fails, log the error
            self.log_error(context, f"Error synthesizing key points: {str(e)}")
            
            # Return the original points if we have a reasonable number, or just the first 10
            if len(key_points) <= 10:
                return key_points
            else:
                return key_points[:10]
    
    def _gather_source_chunks(self, key_points: List[Dict[str, Any]], indices: List[int]) -> List[int]:
        """
        Gather source chunks from related key points.
        
        Args:
            key_points: List of key points
            indices: Indices of related key points
            
        Returns:
            List of source chunk indices
        """
        source_chunks = set()
        
        for idx in indices:
            if 0 <= idx < len(key_points):
                if isinstance(key_points[idx], dict):
                    chunks = key_points[idx].get("source_chunks", [])
                    if chunks and isinstance(chunks, list):
                        source_chunks.update(chunks)
        
        return list(source_chunks)