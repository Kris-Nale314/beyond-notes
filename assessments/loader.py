# assessments/loader.py
import os
import json
import logging
import shutil
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class AssessmentLoader:
    """
    Enhanced loader for assessment configurations.
    
    Manages loading, validation, and creation of assessment definitions,
    providing access to assessment types and their configurations.
    """
    
    def __init__(self, assessment_dir: Optional[str] = None):
        """
        Initialize the assessment loader.
        
        Args:
            assessment_dir: Custom directory for assessment configurations
        """
        self.assessment_dir = assessment_dir or str(Path(__file__).parent)
        self.assessment_path = Path(self.assessment_dir)
        
        # Dictionary of loaded assessments
        self.assessments = {}
        
        # Dictionary of assessment schemas
        self.schemas = {}
        
        # Define standard assessment types
        self.standard_types = ["issues", "action_items", "insights"]
        
        # Load built-in assessments
        self._load_builtin_assessments()
    
    def _load_builtin_assessments(self) -> None:
        """Load built-in assessment configurations."""
        # Create assessments directory if it doesn't exist
        self.assessment_path.mkdir(exist_ok=True)
        
        # Load standard assessment types
        for assessment_type in self.standard_types:
            assessment_type_dir = self.assessment_path / assessment_type
            
            # Skip if directory doesn't exist
            if not assessment_type_dir.exists():
                logger.warning(f"Directory for {assessment_type} assessment not found")
                continue
                
            config_path = assessment_type_dir / "config.json"
            
            # Skip if config doesn't exist
            if not config_path.exists():
                logger.warning(f"Configuration file for {assessment_type} assessment not found")
                continue
            
            try:
                # Load the configuration
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Validate and register the assessment
                if self._validate_assessment_config(config):
                    self.assessments[assessment_type] = config
                    logger.info(f"Loaded {assessment_type} assessment configuration")
                else:
                    logger.error(f"Invalid configuration for {assessment_type} assessment")
            
            except Exception as e:
                logger.error(f"Error loading {assessment_type} assessment: {str(e)}")
    
    def _validate_assessment_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate an assessment configuration.
        
        Args:
            config: Assessment configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Required top-level keys
        required_keys = ["crew_type", "description", "workflow"]
        
        # Check required keys
        for key in required_keys:
            if key not in config:
                logger.error(f"Assessment configuration missing required key: {key}")
                return False
        
        # Validate workflow section
        workflow = config.get("workflow", {})
        if not isinstance(workflow, dict):
            logger.error("Workflow must be a dictionary")
            return False
        
        if "enabled_stages" not in workflow:
            logger.error("Workflow missing enabled_stages")
            return False
        
        if "agent_roles" not in workflow:
            logger.error("Workflow missing agent_roles")
            return False
        
        # Validate agent roles - at minimum we need these
        required_agents = ["planner", "extractor", "aggregator"]
        for agent in required_agents:
            if agent not in workflow.get("agent_roles", {}):
                logger.error(f"Assessment missing required agent role: {agent}")
                return False
        
        return True
    
    def load_assessment(self, assessment_type: str) -> Optional[Dict[str, Any]]:
        """
        Load an assessment configuration by type.
        
        Args:
            assessment_type: Type of assessment to load
            
        Returns:
            Assessment configuration or None if not found
        """
        # Check if already loaded
        if assessment_type in self.assessments:
            return self.assessments[assessment_type]
        
        # Try to load from file
        config_path = self.assessment_path / assessment_type / "config.json"
        
        if not config_path.exists():
            logger.error(f"No configuration file found for {assessment_type} assessment")
            return None
        
        try:
            # Load the configuration
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate and register the assessment
            if self._validate_assessment_config(config):
                self.assessments[assessment_type] = config
                logger.info(f"Loaded {assessment_type} assessment configuration")
                return config
            else:
                logger.error(f"Invalid configuration for {assessment_type} assessment")
                return None
        
        except Exception as e:
            logger.error(f"Error loading {assessment_type} assessment: {str(e)}")
            return None
    
    def get_assessment_types(self) -> List[str]:
        """
        Get a list of available assessment types.
        
        Returns:
            List of assessment type names
        """
        # Return loaded assessments
        loaded_types = list(self.assessments.keys())
        
        # Scan directory for additional types
        if self.assessment_path.exists():
            for item in self.assessment_path.iterdir():
                if item.is_dir():
                    config_path = item / "config.json"
                    if config_path.exists() and item.name not in loaded_types:
                        loaded_types.append(item.name)
        
        return loaded_types
    
    def create_assessment(self, assessment_type: str, config: Dict[str, Any]) -> bool:
        """
        Create a new assessment configuration.
        
        Args:
            assessment_type: Type name for the new assessment
            config: Assessment configuration
            
        Returns:
            True if successful, False otherwise
        """
        # Validate configuration
        if not self._validate_assessment_config(config):
            logger.error("Invalid assessment configuration")
            return False
        
        # Create directory if it doesn't exist
        assessment_dir = self.assessment_path / assessment_type
        assessment_dir.mkdir(exist_ok=True)
        
        # Write configuration file
        config_path = assessment_dir / "config.json"
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Add to loaded assessments
            self.assessments[assessment_type] = config
            logger.info(f"Created {assessment_type} assessment configuration")
            return True
            
        except Exception as e:
            logger.error(f"Error creating {assessment_type} assessment: {str(e)}")
            return False
    
    def get_agent_instructions(self, assessment_type: str, agent_role: str) -> Optional[Dict[str, Any]]:
        """
        Get instructions for a specific agent in an assessment.
        
        Args:
            assessment_type: Type of assessment
            agent_role: Role of the agent (planner, extractor, etc.)
            
        Returns:
            Agent instructions or None if not found
        """
        # Load assessment if not already loaded
        assessment = self.load_assessment(assessment_type)
        if not assessment:
            return None
        
        # Get agent roles from workflow
        workflow = assessment.get("workflow", {})
        agent_roles = workflow.get("agent_roles", {})
        
        # Return instructions for the requested role
        if agent_role in agent_roles:
            return agent_roles[agent_role]
        else:
            logger.warning(f"Agent role {agent_role} not found in {assessment_type} assessment")
            return None
    
    def get_assessment_schema(self, assessment_type: str) -> Optional[Dict[str, Any]]:
        """
        Get the output schema for an assessment type.
        
        Args:
            assessment_type: Type of assessment
            
        Returns:
            Schema dictionary or None if not found
        """
        # Check if schema is already loaded
        if assessment_type in self.schemas:
            return self.schemas[assessment_type]
        
        # Look for schema file
        schema_path = self.assessment_path / assessment_type / "schema.json"
        
        if not schema_path.exists():
            logger.warning(f"No schema found for {assessment_type} assessment")
            return None
        
        try:
            # Load the schema
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            
            # Cache and return
            self.schemas[assessment_type] = schema
            return schema
            
        except Exception as e:
            logger.error(f"Error loading schema for {assessment_type} assessment: {str(e)}")
            return None
    
    def update_assessment(self, assessment_type: str, config: Dict[str, Any]) -> bool:
            """
            Update an existing assessment configuration.
            
            Args:
                assessment_type: Assessment type to update
                config: New assessment configuration
                
            Returns:
                True if successful, False otherwise
            """
            # Validate configuration
            if not self._validate_assessment_config(config):
                logger.error("Invalid assessment configuration")
                return False
            
            # Check if assessment exists
            assessment_dir = self.assessment_path / assessment_type
            if not assessment_dir.exists():
                logger.error(f"Assessment type {assessment_type} does not exist")
                return False
            
            # Write configuration file
            config_path = assessment_dir / "config.json"
            try:
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                # Update loaded assessments
                self.assessments[assessment_type] = config
                logger.info(f"Updated {assessment_type} assessment configuration")
                return True
                
            except Exception as e:
                logger.error(f"Error updating {assessment_type} assessment: {str(e)}")
                return False
        
    def delete_assessment(self, assessment_type: str) -> bool:
        """
        Delete an assessment configuration.
        
        Args:
            assessment_type: Assessment type to delete
            
        Returns:
            True if successful, False otherwise
        """
        # Don't allow deletion of standard types
        if assessment_type in self.standard_types:
            logger.error(f"Cannot delete standard assessment type: {assessment_type}")
            return False
        
        # Check if assessment exists
        assessment_dir = self.assessment_path / assessment_type
        if not assessment_dir.exists():
            logger.error(f"Assessment type {assessment_type} does not exist")
            return False
        
        try:
            # Remove directory and all contents
            shutil.rmtree(assessment_dir)
            
            # Remove from loaded assessments
            if assessment_type in self.assessments:
                del self.assessments[assessment_type]
            
            if assessment_type in self.schemas:
                del self.schemas[assessment_type]
            
            logger.info(f"Deleted {assessment_type} assessment configuration")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting {assessment_type} assessment: {str(e)}")
            return False
    
    def create_assessment_directories(self) -> None:
        """Create directories for standard assessment types with sample configurations."""
        for assessment_type in self.standard_types:
            assessment_dir = self.assessment_path / assessment_type
            
            # Skip if already exists
            if assessment_dir.exists():
                continue
            
            # Create directory
            assessment_dir.mkdir(exist_ok=True)
            logger.info(f"Created directory for {assessment_type} assessment")
            
            # Determine which template to use
            if assessment_type == "issues":
                config = self._get_issues_template()
                schema = self._get_issues_schema()
            elif assessment_type == "action_items":
                config = self._get_action_items_template()
                schema = self._get_action_items_schema()
            elif assessment_type == "insights":
                config = self._get_insights_template()
                schema = self._get_insights_schema()
            else:
                # Generic template
                config = self._get_generic_template(assessment_type)
                schema = self._get_generic_schema()
            
            # Write config file
            config_path = assessment_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Write schema file
            schema_path = assessment_dir / "schema.json"
            with open(schema_path, 'w') as f:
                json.dump(schema, f, indent=2)
            
            logger.info(f"Created configuration files for {assessment_type} assessment")
    
    def get_user_options(self, assessment_type: str) -> Dict[str, Any]:
        """
        Get user-configurable options for an assessment type.
        
        Args:
            assessment_type: Assessment type
            
        Returns:
            Dictionary of user options
        """
        assessment = self.load_assessment(assessment_type)
        if not assessment:
            return {}
        
        return assessment.get("user_options", {})
    
    def _get_generic_template(self, assessment_type: str) -> Dict[str, Any]:
        """Create a generic assessment template."""
        return {
            "crew_type": assessment_type,
            "description": f"Generic {assessment_type} assessment",
            
            "workflow": {
                "enabled_stages": ["document_analysis", "chunking", "planning", "extraction", "aggregation", "evaluation", "formatting", "review"],
                "agent_roles": {
                    "planner": {
                        "description": "Plans the analysis approach",
                        "primary_task": "Create tailored instructions",
                        "instructions": "Study the document to identify its structure and purpose. Create extraction instructions."
                    },
                    "extractor": {
                        "description": "Extracts relevant information",
                        "primary_task": "Find relevant information in chunks",
                        "instructions": "Extract relevant information from each document chunk."
                    },
                    "aggregator": {
                        "description": "Combines extracted information",
                        "primary_task": "Consolidate findings",
                        "instructions": "Combine findings from different chunks and remove duplicates."
                    }
                },
                "stage_weights": {
                    "document_analysis": 0.1,
                    "chunking": 0.1,
                    "planning": 0.1,
                    "extraction": 0.3,
                    "aggregation": 0.2,
                    "evaluation": 0.1,
                    "formatting": 0.1
                }
            }
        }
    
    def _get_generic_schema(self) -> Dict[str, Any]:
        """Create a generic schema template."""
        return {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Summary of findings"
                },
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "category": {"type": "string"},
                            "importance": {"type": "string"}
                        }
                    }
                },
                "metadata": {
                    "type": "object"
                }
            }
        }
    
    def _get_issues_template(self) -> Dict[str, Any]:
        """Get the template for issues assessment."""
        return {
            "crew_type": "issues",
            "description": "Identifies problems, challenges, risks, and concerns in documents",
            
            "issue_definition": {
                "description": "Any problem, challenge, risk, or concern that may impact objectives, efficiency, or quality",
                "severity_levels": {
                    "critical": "Immediate threat requiring urgent attention",
                    "high": "Significant impact requiring prompt attention",
                    "medium": "Moderate impact that should be addressed",
                    "low": "Minor impact with limited consequences"
                },
                "categories": [
                    "technical", "process", "resource", "quality", "risk", "compliance"
                ]
            },
            
            "workflow": {
                "enabled_stages": ["document_analysis", "chunking", "planning", "extraction", "aggregation", "evaluation", "formatting", "review"],
                "agent_roles": {
                    "planner": {
                        "description": "Plans the analysis approach and creates instructions for other agents",
                        "primary_task": "Create tailored instructions for each stage based on document type and user preferences",
                        "instructions": "Study the beginning of the document to identify its type and purpose. Then create specific extraction and evaluation instructions for identifying issues and problems."
                    },
                    "extractor": {
                        "description": "Identifies issues from document chunks",
                        "primary_task": "Find all issues, assign initial severity, and provide relevant context",
                        "output_schema": {
                            "title": "Concise issue label",
                            "description": "Detailed explanation of the issue",
                            "severity": "Initial severity assessment (critical/high/medium/low)",
                            "category": "Issue category from the defined list",
                            "context": "Relevant information from the document"
                        },
                        "instructions": "Carefully analyze each document chunk to identify issues, challenges, risks, or concerns. Focus on problems that might impact objectives, efficiency, or quality."
                    },
                    "aggregator": {
                        "description": "Combines and deduplicates issues from all chunks",
                        "primary_task": "Consolidate similar issues while preserving important distinctions",
                        "instructions": "Combine similar issues from different chunks, ensuring that unique details and context are preserved. Remove duplicate issues and reconcile any conflicting severity ratings."
                    },
                    "evaluator": {
                        "description": "Assesses issue severity and priority",
                        "primary_task": "Analyze each issue's impact and assign final severity and priority",
                        "evaluation_criteria": [
                            "Impact scope (how many people/systems affected)",
                            "Urgency (how soon it needs to be addressed)",
                            "Consequence (what happens if not addressed)",
                            "Complexity (how difficult it might be to solve)"
                        ],
                        "instructions": "Evaluate each issue against the assessment criteria. Consider both immediate and long-term impacts. Assign final severity ratings and prioritize the issues."
                    },
                    "formatter": {
                        "description": "Creates the structured report",
                        "primary_task": "Organize issues by severity and category into a clear report",
                        "instructions": "Format the issues into a structured report, organizing them by severity and category. Create an executive summary highlighting the most critical issues."
                    },
                    "reviewer": {
                        "description": "Ensures quality and alignment with user needs",
                        "primary_task": "Verify report quality and alignment with user preferences",
                        "instructions": "Review the generated report for accuracy, completeness, and alignment with the user's needs. Check for logical inconsistencies or missing important issues."
                    }
                },
                "stage_weights": {
                    "document_analysis": 0.05,
                    "chunking": 0.05,
                    "planning": 0.10,
                    "extraction": 0.30,
                    "aggregation": 0.20,
                    "evaluation": 0.15,
                    "formatting": 0.10,
                    "review": 0.05
                }
            },
            
            "user_options": {
                "detail_levels": {
                    "essential": "Focus only on the most significant issues",
                    "standard": "Balanced analysis of important issues",
                    "comprehensive": "In-depth analysis of all potential issues"
                },
                "focus_areas": {
                    "technical": "Implementation, architecture, technology issues",
                    "process": "Workflow, procedure, methodology issues",
                    "resource": "Staffing, budget, time, materials constraints",
                    "quality": "Standards, testing, performance concerns",
                    "risk": "Compliance, security, strategic risks"
                }
            },
            
            "report_format": {
                "sections": [
                    "Executive Summary",
                    "Critical Issues",
                    "High-Priority Issues", 
                    "Medium-Priority Issues",
                    "Low-Priority Issues"
                ],
                "issue_presentation": {
                    "title": "Clear, descriptive title",
                    "severity": "Visual indicator of severity",
                    "description": "Full issue description",
                    "impact": "Potential consequences",
                    "category": "Issue category"
                }
            }
        }
    
    def _get_issues_schema(self) -> Dict[str, Any]:
        """Get the schema for issues assessment."""
        return {
            "type": "object",
            "properties": {
                "executive_summary": {
                    "type": "string",
                    "description": "A concise summary of the key issues found"
                },
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Short, descriptive title of the issue"
                            },
                            "description": {
                                "type": "string",
                                "description": "Detailed explanation of the issue"
                            },
                            "severity": {
                                "type": "string",
                                "enum": ["critical", "high", "medium", "low"],
                                "description": "Assessment of the issue's severity"
                            },
                            "category": {
                                "type": "string",
                                "enum": ["technical", "process", "resource", "quality", "risk", "compliance"],
                                "description": "Category of the issue"
                            },
                            "impact": {
                                "type": "string",
                                "description": "Potential consequences if not addressed"
                            },
                            "evidence": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "text": {
                                            "type": "string",
                                            "description": "Source text supporting this issue"
                                        },
                                        "chunk_index": {
                                            "type": "integer",
                                            "description": "Index of the chunk containing this evidence"
                                        }
                                    },
                                    "required": ["text"]
                                }
                            },
                            "recommendations": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "description": "Potential approaches to address the issue"
                                }
                            }
                        },
                        "required": ["title", "description", "severity", "category"]
                    }
                },
                "statistics": {
                    "type": "object",
                    "properties": {
                        "total_issues": {
                            "type": "integer",
                            "description": "Total number of issues identified"
                        },
                        "by_severity": {
                            "type": "object",
                            "properties": {
                                "critical": {"type": "integer"},
                                "high": {"type": "integer"},
                                "medium": {"type": "integer"},
                                "low": {"type": "integer"}
                            }
                        },
                        "by_category": {
                            "type": "object",
                            "properties": {
                                "technical": {"type": "integer"},
                                "process": {"type": "integer"},
                                "resource": {"type": "integer"},
                                "quality": {"type": "integer"},
                                "risk": {"type": "integer"},
                                "compliance": {"type": "integer"}
                            }
                        }
                    }
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "document_name": {"type": "string"},
                        "word_count": {"type": "integer"},
                        "processing_time": {"type": "number"},
                        "date_analyzed": {"type": "string"},
                        "assessment_type": {"type": "string"},
                        "user_options": {"type": "object"}
                    }
                }
            },
            "required": ["executive_summary", "issues", "statistics", "metadata"]
        }
    
    def _get_action_items_template(self) -> Dict[str, Any]:
        """Get the template for action items assessment."""
        # Similar to the JSON we already defined
        return {
            "crew_type": "action_items",
            "description": "Identifies concrete follow-up tasks and commitments from meeting transcripts",
            
            "action_item_definition": {
                "description": "A specific, actionable task that requires follow-up after the meeting, typically with an owner and timeframe",
                "priority_levels": {
                    "critical": "Must be completed urgently",
                    "high": "Important task requiring prompt attention",
                    "medium": "Standard priority task",
                    "low": "Task to be done when time permits"
                },
                "status_types": [
                    "assigned", "pending", "in_progress", "blocked", "completed"
                ]
            },
            
            "workflow": {
                "enabled_stages": ["document_analysis", "chunking", "planning", "extraction", "aggregation", "evaluation", "formatting", "review"],
                "agent_roles": {
                    "planner": {
                        "description": "Plans the analysis approach for action item identification",
                        "primary_task": "Create tailored instructions for identifying true action items vs general statements",
                        "instructions": "Study the document to identify its structure and purpose. Then create specific extraction and evaluation instructions for identifying real action items that require follow-up."
                    },
                    "extractor": {
                        "description": "Identifies potential action items from document chunks",
                        "primary_task": "Extract action-oriented statements with ownership and timing details",
                        "output_schema": {
                            "description": "Description of the action item",
                            "owner": "Person or group responsible",
                            "due_date": "When it should be completed",
                            "priority": "Initial priority assessment",
                            "confidence": "Confidence that this is a true action item requiring follow-up",
                            "context": "Surrounding context from the document"
                        },
                        "instructions": "Extract action-oriented statements that appear to be tasks, commitments, or follow-ups. Look for indicators like assigned ownership, deadlines, or explicit commitments. Include both clear action items and potential ones for later filtering."
                    },
                    "aggregator": {
                        "description": "Combines and deduplicates action items",
                        "primary_task": "Consolidate similar action items while preserving ownership and timing details",
                        "instructions": "Combine similar action items from different document sections. Preserve the most specific owner, due date, and priority information. Eliminate true duplicates but keep similar items that might represent different tasks."
                    },
                    "evaluator": {
                        "description": "Assesses if extracted items are true action items requiring follow-up",
                        "primary_task": "Filter out statements that aren't actual action items and assign final priorities",
                        "evaluation_criteria": [
                            "Actionability (is it clear what needs to be done)",
                            "Ownership (is someone responsible)",
                            "Timing (is there a timeframe)",
                            "Commitment (was it agreed upon)",
                            "Follow-up required (needs post-meeting attention)"
                        ],
                        "instructions": "Evaluate each potential action item to determine if it's a true action item requiring follow-up. Filter out general statements, hypotheticals, or discussion of past actions. A true action item should be clear, have an owner or way to assign ownership, and ideally have a timeframe. Assess priority based on urgency and importance mentioned."
                    },
                    "formatter": {
                        "description": "Creates a structured action item list",
                        "primary_task": "Format action items into a clear, actionable list organized by owner and priority",
                        "instructions": "Create a well-structured action item list that can be easily shared after a meeting. Organize by owner and then priority. Include all relevant details like due dates and context for each item."
                    },
                    "reviewer": {
                        "description": "Ensures action items are clear, actionable, and properly assigned",
                        "primary_task": "Verify action items have clear owners, actions, and timelines where possible",
                        "instructions": "Review the action item list for completeness and clarity. Ensure each item clearly states what needs to be done, who should do it, and when it should be completed (if specified). Check that no important action items from the document were missed."
                    }
                },
                "stage_weights": {
                    "document_analysis": 0.05,
                    "chunking": 0.05,
                    "planning": 0.10,
                    "extraction": 0.30,
                    "aggregation": 0.15,
                    "evaluation": 0.20,
                    "formatting": 0.10,
                    "review": 0.05
                }
            },
            
            "user_options": {
                "detail_levels": {
                    "essential": "Include only clearly defined action items with owners",
                    "standard": "Include most action items with reasonable confidence",
                    "comprehensive": "Include all potential action items, even with lower confidence"
                },
                "format_options": {
                    "by_owner": "Group action items by owner",
                    "by_priority": "Group action items by priority",
                    "by_due_date": "Group action items by timeline",
                    "sequential": "List action items in the order they appeared"
                }
            },
            
            "report_format": {
                "sections": [
                    "Summary",
                    "Action Items by Owner",
                    "Action Items by Due Date"
                ],
                "action_item_presentation": {
                    "owner": "Person responsible",
                    "description": "What needs to be done",
                    "due_date": "When it should be completed",
                    "priority": "Visual indicator of priority",
                    "context": "Meeting context"
                }
            }
        }
    
    def _get_action_items_schema(self) -> Dict[str, Any]:
        """Get the schema for action items assessment."""
        return {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A brief summary of the key action items"
                },
                "action_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "Clear description of what needs to be done"
                            },
                            "owner": {
                                "type": "string",
                                "description": "Person or group responsible for this action item"
                            },
                            "due_date": {
                                "type": "string",
                                "description": "When this action item should be completed"
                            },
                            "priority": {
                                "type": "string",
                                "enum": ["critical", "high", "medium", "low"],
                                "description": "Priority level of the action item"
                            },
                            "status": {
                                "type": "string",
                                "enum": ["assigned", "pending", "in_progress", "blocked", "completed"],
                                "description": "Current status of the action item"
                            },
                            "context": {
                                "type": "string",
                                "description": "Meeting context where this action item was discussed"
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "Confidence score that this is a true action item (0-1)"
                            },
                            "evidence": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "text": {
                                            "type": "string",
                                            "description": "Source text supporting this action item"
                                        },
                                        "chunk_index": {
                                            "type": "integer",
                                            "description": "Index of the chunk containing this evidence"
                                        }
                                    },
                                    "required": ["text"]
                                }
                            }
                        },
                        "required": ["description", "owner", "priority"]
                    }
                },
                "statistics": {
                    "type": "object",
                    "properties": {
                        "total_action_items": {
                            "type": "integer",
                            "description": "Total number of action items identified"
                        },
                        "by_owner": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "integer"
                            },
                            "description": "Count of action items by owner"
                        },
                        "by_priority": {
                            "type": "object",
                            "properties": {
                                "critical": {"type": "integer"},
                                "high": {"type": "integer"},
                                "medium": {"type": "integer"},
                                "low": {"type": "integer"}
                            }
                        },
                        "by_status": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "integer"
                            }
                        }
                    }
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "document_name": {"type": "string"},
                        "meeting_date": {"type": "string"},
                        "word_count": {"type": "integer"},
                        "processing_time": {"type": "number"},
                        "date_analyzed": {"type": "string"},
                        "assessment_type": {"type": "string"},
                        "user_options": {"type": "object"}
                    }
                }
            },
            "required": ["summary", "action_items", "statistics", "metadata"]
        }
    
    def _get_insights_template(self) -> Dict[str, Any]:
        """Get the template for insights assessment."""
        return {
            "crew_type": "insights",
            "description": "Performs SWOT analysis (Strengths, Weaknesses, Opportunities, Threats) on meeting transcripts with client context",
            
            "insight_definition": {
                "description": "Meaningful observations about strengths, weaknesses, opportunities, and threats derived from the discussion",
                "categories": {
                    "strength": "Positive internal factors or capabilities mentioned in the discussion",
                    "weakness": "Internal limitations, challenges, or areas for improvement",
                    "opportunity": "External factors that could benefit the client or situation",
                    "threat": "External challenges or risks that could negatively impact the client"
                },
                "impact_levels": {
                    "high": "Major impact on business outcomes or strategy",
                    "medium": "Moderate impact requiring attention",
                    "low": "Minor impact worth noting"
                }
            },
            
            "workflow": {
                "enabled_stages": ["document_analysis", "chunking", "planning", "extraction", "aggregation", "evaluation", "formatting", "review"],
                "agent_roles": {
                    "planner": {
                        "description": "Plans the SWOT analysis approach with client context in mind",
                        "primary_task": "Create tailored instructions for identifying SWOT elements relevant to client context",
                        "instructions": "Study the document and client context to identify the most relevant insights. Create specific extraction and evaluation instructions that focus on SWOT elements particularly relevant to this client's situation."
                    },
                    "extractor": {
                        "description": "Identifies SWOT elements from document chunks",
                        "primary_task": "Extract statements that indicate strengths, weaknesses, opportunities, or threats",
                        "output_schema": {
                            "type": "SWOT category (strength, weakness, opportunity, threat)",
                            "description": "Description of the insight",
                            "impact": "Initial impact assessment",
                            "confidence": "Confidence level in this assessment",
                            "context": "Relevant context from the document"
                        },
                        "instructions": "Extract statements that indicate strengths, weaknesses, opportunities, or threats. Use the client context to guide your identification - what might be a strength for one organization could be a weakness for another. Look for explicit statements as well as implied insights."
                    },
                    "aggregator": {
                        "description": "Combines and organizes SWOT elements",
                        "primary_task": "Consolidate similar insights and organize by SWOT category",
                        "instructions": "Combine similar insights while preserving nuance and detail. Organize insights into proper SWOT categories. Ensure insights are aligned with the client context provided."
                    },
                    "evaluator": {
                        "description": "Assesses impact and relevance of SWOT elements",
                        "primary_task": "Evaluate the impact and strategic importance of each insight given client context",
                        "evaluation_criteria": [
                            "Relevance to client context",
                            "Strategic importance",
                            "Actionability",
                            "Time sensitivity",
                            "Magnitude of potential impact"
                        ],
                        "instructions": "Evaluate each insight for its impact and strategic importance given the client context. Consider how actionable the insight is, its time sensitivity, and the magnitude of its potential impact. Prioritize insights that are most relevant to the client's specific situation and goals."
                    },
                    "formatter": {
                        "description": "Creates a structured SWOT analysis report",
                        "primary_task": "Format insights into a clear SWOT analysis organized by category and impact",
                        "instructions": "Create a well-structured SWOT analysis report that clearly presents strengths, weaknesses, opportunities, and threats. Organize by impact level within each category. Include an executive summary that highlights the most important insights across categories."
                    },
                    "reviewer": {
                        "description": "Ensures the SWOT analysis is balanced, insightful, and relevant to client context",
                        "primary_task": "Verify insights are properly categorized and provide strategic value",
                        "instructions": "Review the SWOT analysis for balance, accuracy, and strategic relevance. Ensure each insight is properly categorized and provides value. Check that the analysis is aligned with the client context and captures the most important insights from the discussion."
                    }
                },
                "stage_weights": {
                    "document_analysis": 0.05,
                    "chunking": 0.05,
                    "planning": 0.15,
                    "extraction": 0.25,
                    "aggregation": 0.15,
                    "evaluation": 0.20,
                    "formatting": 0.10,
                    "review": 0.05
                }
            },
            
            "user_options": {
                "detail_levels": {
                    "essential": "Focus only on high-impact insights",
                    "standard": "Include high and medium impact insights",
                    "comprehensive": "Include all insights regardless of impact level"
                },
                "client_context": {
                    "required": true,
                    "description": "Background information about the client to ground the analysis",
                    "fields": {
                        "client_name": "Name of the organization or client",
                        "industry": "Client's industry or sector",
                        "size": "Organization size (revenue, employees)",
                        "key_challenges": "Known challenges or issues",
                        "objectives": "Client's goals or objectives for this engagement",
                        "current_systems": "Current technology or process landscape",
                        "additional_context": "Any other relevant background information"
                    }
                }
            },
            
            "report_format": {
                "sections": [
                    "Executive Summary",
                    "Strengths",
                    "Weaknesses",
                    "Opportunities",
                    "Threats",
                    "Strategic Recommendations"
                ],
                "insight_presentation": {
                    "description": "Clear description of the insight",
                    "impact": "Visual indicator of impact level",
                    "evidence": "Supporting evidence from the transcript",
                    "strategic_implication": "What this means for the client's strategy"
                }
            }
        }
    
    def _get_insights_schema(self) -> Dict[str, Any]:
        """Get the schema for insights assessment."""
        return {
            "type": "object",
            "properties": {
                "executive_summary": {
                    "type": "string",
                    "description": "A concise summary of the key insights from the SWOT analysis"
                },
                "client_context": {
                    "type": "object",
                    "properties": {
                        "client_name": {"type": "string"},
                        "industry": {"type": "string"},
                        "size": {"type": "string"},
                        "key_challenges": {"type": "string"},
                        "objectives": {"type": "string"},
                        "current_systems": {"type": "string"},
                        "additional_context": {"type": "string"}
                    }
                },
                "swot_analysis": {
                    "type": "object",
                    "properties": {
                        "strengths": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "description": {"type": "string"},
                                    "impact": {"type": "string", "enum": ["high", "medium", "low"]},
                                    "strategic_implication": {"type": "string"},
                                    "evidence": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "text": {"type": "string"},
                                                "chunk_index": {"type": "integer"}
                                            },
                                            "required": ["text"]
                                        }
                                    }
                                },
                                "required": ["description", "impact"]
                            }
                        },
                        "weaknesses": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "description": {"type": "string"},
                                    "impact": {"type": "string", "enum": ["high", "medium", "low"]},
                                    "strategic_implication": {"type": "string"},
                                    "evidence": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "text": {"type": "string"},
                                                "chunk_index": {"type": "integer"}
                                            },
                                            "required": ["text"]
                                        }
                                    }
                                },
                                "required": ["description", "impact"]
                            }
                        },
                        "opportunities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "description": {"type": "string"},
                                    "impact": {"type": "string", "enum": ["high", "medium", "low"]},
                                    "strategic_implication": {"type": "string"},
                                    "evidence": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "text": {"type": "string"},
                                                "chunk_index": {"type": "integer"}
                                            },
                                            "required": ["text"]
                                        }
                                    }
                                },
                                "required": ["description", "impact"]
                            }
                        },
                        "threats": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "description": {"type": "string"},
                                    "impact": {"type": "string", "enum": ["high", "medium", "low"]},
                                    "strategic_implication": {"type": "string"},
                                    "evidence": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "text": {"type": "string"},
                                                "chunk_index": {"type": "integer"}
                                            },
                                            "required": ["text"]
                                        }
                                    }
                                },
                                "required": ["description", "impact"]
                            }
                        }
                    },
                    "required": ["strengths", "weaknesses", "opportunities", "threats"]
                },
                "strategic_recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "recommendation": {"type": "string"},
                            "rationale": {"type": "string"},
                            "related_insights": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "priority": {"type": "string", "enum": ["high", "medium", "low"]}
                        },
                        "required": ["recommendation", "rationale"]
                    }
                },
                "statistics": {
                    "type": "object",
                    "properties": {
                        "strengths_count": {"type": "integer"},
                        "weaknesses_count": {"type": "integer"},
                        "opportunities_count": {"type": "integer"},
                        "threats_count": {"type": "integer"},
                        "by_impact": {
                            "type": "object",
                            "properties": {
                                "high": {"type": "integer"},
                                "medium": {"type": "integer"},
                                "low": {"type": "integer"}
                            }
                        },
                        "recommendations_count": {"type": "integer"}
                    }
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "document_name": {"type": "string"},
                        "meeting_date": {"type": "string"},
                        "word_count": {"type": "integer"},
                        "processing_time": {"type": "number"},
                        "date_analyzed": {"type": "string"},
                        "assessment_type": {"type": "string"},
                        "user_options": {"type": "object"}
                    }
                }
            },
            "required": ["executive_summary", "swot_analysis", "strategic_recommendations", "statistics", "metadata"]
        }
   