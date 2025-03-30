# core/agents/evaluator.py
from typing import Dict, Any, List, Optional
import json
import asyncio

from core.agents.base import Agent
from core.models.context import ProcessingContext

class EvaluatorAgent(Agent):
    """
    The Evaluator Agent assesses importance, assigns categories 
    and ratings using specialized frameworks.
    
    Responsibilities:
    - Assess severity and impact of issues
    - Apply consistent evaluation criteria
    - Prioritize findings
    - Add contextual assessment
    """
    
    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Evaluate aggregated information.
        
        Args:
            context: The shared processing context
            
        Returns:
            Evaluation results
        """
        self.log_info(context, "Starting evaluation")
        
        try:
            # Get aggregation results
            aggregation_results = context.results.get("aggregation", {})
            issues = aggregation_results.get("issues", [])
            key_points = aggregation_results.get("key_points", [])
            
            if not issues:
                self.log_warning(context, "No issues available for evaluation")
            
            # Get assessment config
            assessment_config = context.assessment_config
            issue_definition = assessment_config.get("issue_definition", {})
            severity_levels = issue_definition.get("severity_levels", {})
            categories = issue_definition.get("categories", [])
            
            # Get custom instructions from assessment config
            agent_instructions = self.get_agent_instructions(context)
            
            # Get evaluation criteria from config
            workflow = assessment_config.get("workflow", {})
            agent_roles = workflow.get("agent_roles", {})
            evaluator_config = agent_roles.get("evaluator", {})
            evaluation_criteria = evaluator_config.get("evaluation_criteria", [])
            
            # Evaluate issues
            self.log_info(context, f"Evaluating {len(issues)} issues")
            evaluated_issues = await self._evaluate_issues(
                context, issues, evaluation_criteria, severity_levels, categories, agent_instructions
            )
            
            # Analyze the overall situation
            self.log_info(context, "Creating overall assessment")
            overall_assessment = await self._create_overall_assessment(
                context, evaluated_issues, key_points, agent_instructions
            )
            
            # Create evaluation results
            evaluation_output = {
                "issues": evaluated_issues,
                "overall_assessment": overall_assessment,
                "statistics": {
                    "total_issues": len(evaluated_issues),
                    "by_severity": self._count_by_field(evaluated_issues, "severity"),
                    "by_category": self._count_by_field(evaluated_issues, "category")
                }
            }
            
            # Add to context evaluations
            context.add_evaluation("issues", {
                "evaluated_issues": evaluated_issues,
                "overall_assessment": overall_assessment
            })
            
            self.log_info(context, "Evaluation complete")
            
            return evaluation_output
            
        except Exception as e:
            error_message = f"Evaluation failed: {str(e)}"
            self.log_error(context, error_message)
            raise
    
    async def _evaluate_issues(self,
                             context: ProcessingContext,
                             issues: List[Dict[str, Any]],
                             evaluation_criteria: List[str],
                             severity_levels: Dict[str, str],
                             categories: List[str],
                             agent_instructions: str) -> List[Dict[str, Any]]:
        """
        Evaluate issues using the specified criteria.
        
        Args:
            context: The processing context
            issues: List of issues to evaluate
            evaluation_criteria: Criteria to apply
            severity_levels: Defined severity levels
            categories: Valid categories
            agent_instructions: Instructions from the assessment config
            
        Returns:
            List of evaluated issues
        """
        if not issues:
            return []
        
        # Process issues in batches to avoid rate limits
        evaluated_issues = await self.process_in_batches(
            context=context,
            items=issues,
            process_func=lambda issue, i: self._evaluate_single_issue(
                issue, evaluation_criteria, severity_levels, categories, agent_instructions
            ),
            batch_size=5,  # Process 5 issues at a time
            delay=0.2  # Small delay between batches
        )
        
        return evaluated_issues
    
    async def _evaluate_single_issue(self,
                                   issue: Dict[str, Any],
                                   evaluation_criteria: List[str],
                                   severity_levels: Dict[str, str],
                                   categories: List[str],
                                   agent_instructions: str) -> Dict[str, Any]:
        """
        Evaluate a single issue.
        
        Args:
            issue: Issue to evaluate
            evaluation_criteria: Criteria to apply
            severity_levels: Defined severity levels
            categories: Valid categories
            agent_instructions: Instructions from the assessment config
            
        Returns:
            Evaluated issue
        """
        # Define schema for evaluated issue
        json_schema = {
            "type": "object",
            "properties": {
                "issue_id": {"type": "string"},
                "title": {"type": "string"},
                "description": {"type": "string"},
                "severity": {"type": "string"},
                "category": {"type": "string"},
                "impact": {"type": "string"},
                "priority": {"type": "string"},
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "evaluation": {
                    "type": "object",
                    "properties": {
                        "rationale": {"type": "string"}
                    }
                }
            },
            "required": ["title", "description", "severity", "category", "impact", "priority"]
        }
        
        # Create a prompt for evaluation
        prompt = f"""
        You are an expert issue evaluator for an AI document analysis system.
        
        Please evaluate the following issue extracted from a document:
        
        ```
        {json.dumps(issue, indent=2)}
        ```
        
        {agent_instructions}
        
        Evaluation Criteria:
        {json.dumps(evaluation_criteria, indent=2)}
        
        Valid Severity Levels:
        {json.dumps(severity_levels, indent=2)}
        
        Valid Categories:
        {json.dumps(categories, indent=2)}
        
        Please evaluate this issue and provide:
        1. A concise title summarizing the issue
        2. A clear, comprehensive description
        3. The appropriate severity level based on impact
        4. The most specific category
        5. An impact assessment describing consequences if not addressed
        6. A priority level (high, medium, low) based on urgency and importance
        7. 1-3 specific recommendations for addressing the issue
        8. Evaluation rationale explaining your assessment
        
        Format your response as a structured JSON object.
        """
        
        try:
            # Generate evaluated issue
            evaluation = await self.generate_structured_output(
                prompt=prompt,
                json_schema=json_schema,
                max_tokens=1500,
                temperature=0.3
            )
            
            # Preserve original issue ID and source information
            evaluation["id"] = issue.get("id", "")
            evaluation["source_chunks"] = issue.get("source_chunks", [])
            
            return evaluation
            
        except Exception as e:
            # If evaluation fails, return the original issue with a note
            issue["evaluation_error"] = str(e)
            return issue
    
    async def _create_overall_assessment(self,
                                       context: ProcessingContext,
                                       evaluated_issues: List[Dict[str, Any]],
                                       key_points: List[Dict[str, Any]],
                                       agent_instructions: str) -> Dict[str, Any]:
        """
        Create an overall assessment based on the evaluated issues.
        
        Args:
            context: The processing context
            evaluated_issues: List of evaluated issues
            key_points: List of key points
            agent_instructions: Instructions from the assessment config
            
        Returns:
            Overall assessment
        """
        # Define schema for overall assessment
        json_schema = {
            "type": "object",
            "properties": {
                "executive_summary": {"type": "string"},
                "key_findings": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "critical_areas": {"type": "string"},
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "overall_severity": {"type": "string"}
            },
            "required": ["executive_summary", "key_findings", "recommendations", "overall_severity"]
        }
        
        # Get document info
        document_info = context.document_info
        
        # Get severity counts
        severity_counts = self._count_by_field(evaluated_issues, "severity")
        
        # Create a prompt for overall assessment
        prompt = f"""
        You are an expert evaluator for an AI document analysis system.
        
        Please create an overall assessment based on the issues and key points extracted from the document.
        
        Document Information:
        ```
        {json.dumps(document_info, indent=2)}
        ```
        
        Severity Distribution:
        ```
        {json.dumps(severity_counts, indent=2)}
        ```
        
        Evaluated Issues:
        ```
        {json.dumps(evaluated_issues, indent=2)}
        ```
        
        Key Points:
        ```
        {json.dumps(key_points, indent=2)}
        ```
        
        {agent_instructions}
        
        Please provide a comprehensive overall assessment with:
        1. An executive summary (around 3-5 sentences)
        2. 3-5 key findings highlighting the most important issues
        3. Analysis of critical areas needing attention
        4. 3-5 strategic recommendations
        5. An overall severity assessment (critical, high, medium, low)
        
        Format your response as a structured JSON object.
        """
        
        try:
            # Generate overall assessment
            assessment = await self.generate_structured_output(
                prompt=prompt,
                json_schema=json_schema,
                max_tokens=2000,
                temperature=0.4
            )
            
            return assessment
            
        except Exception as e:
            # If assessment fails, create a basic assessment
            return {
                "executive_summary": "Could not generate executive summary due to an error.",
                "key_findings": ["Error generating key findings"],
                "critical_areas": "Could not analyze critical areas due to an error.",
                "recommendations": ["Review issues manually"],
                "overall_severity": "unknown",
                "error": str(e)
            }
    
    def _count_by_field(self, items: List[Dict[str, Any]], field: str) -> Dict[str, int]:
        """
        Count items by a specific field value.
        
        Args:
            items: List of items to count
            field: Field to count by
            
        Returns:
            Dictionary with counts by field value
        """
        counts = {}
        
        for item in items:
            value = item.get(field, "unknown")
            if value:
                counts[value] = counts.get(value, 0) + 1
        
        return counts