# `beyond-notes`: Technical Architecture Overview

## 🏗️ Architecture Overview

`beyond-notes` is a scalable, multi-agent document analysis system built to transform meeting transcripts and documents into structured, actionable insights. The architecture employs a pipeline of specialized AI agents that collaborate through a shared context object to progressively analyze, extract, refine, and format information from unstructured text.

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Document &   │     │               │     │               │     │               │
│  Assessment   │────▶│ Orchestrator  │────▶│ Agent Pipeline│────▶│  Final Output │
│  Configuration│     │               │     │               │     │               │
└───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
```

## 🧩 Core Components

### Assessment Types

The system supports four primary assessment types, each with its own specialized workflow and output structure:

1. **Distill** (`distill`) - Creates concise summaries and extracts key points from documents
   - Focuses on: summary generation, key point extraction, topic categorization
   - Output: document summary, categorized key points by topic

2. **Extract** (`extract`) - Identifies and organizes action items from text
   - Focuses on: action item descriptions, owners, due dates, priority levels
   - Output: structured action items with metadata for tracking

3. **Assess** (`assess`) - Uncovers issues, challenges, and risks
   - Focuses on: issue identification, severity rating, impact analysis
   - Output: categorized issues with severity ratings and recommendations

4. **Analyze** (`analyze`) - Evaluates content against structured frameworks
   - Focuses on: evidence collection, criteria assessment, maturity ratings
   - Output: dimension and criteria assessments with supporting evidence

### Configuration System

The assessment pipeline is driven by a flexible JSON configuration system:

```
assessments/
├── types/                 # Base assessment type configurations
│   ├── base_distill_summary_v1.json
│   ├── base_extract_action_items_v1.json
│   ├── base_assess_issue_v1.json
│   └── base_analyze_readiness_v1.json
└── templates/             # User-created assessment templates
    └── [custom templates].json
```

#### Assessment Configuration Structure

Each assessment configuration JSON defines:

```json
{
  "assessment_id": "base_assess_issue_v1",
  "assessment_type": "assess",
  "display_name": "Issue Assessment",
  "description": "Identifies issues, challenges, and risks within documents",
  "version": "1.0",
  
  "entity_definition": {
    "name": "issue",
    "plural": "issues",
    "description": "A problem, challenge, or risk that needs attention",
    "properties": {
      "title": { "type": "string", "description": "Brief title of the issue" },
      "description": { "type": "string", "description": "Detailed description of the issue" },
      "severity": { 
        "type": "string", 
        "description": "Severity level of the issue",
        "options": ["critical", "high", "medium", "low"],
        "descriptions": {
          "critical": "Must be addressed immediately; significant negative impact",
          "high": "Should be addressed soon; substantial negative impact",
          "medium": "Should be addressed in the near term; moderate negative impact",
          "low": "Can be addressed when convenient; minor negative impact"
        }
      },
      "category": { 
        "type": "string", 
        "description": "Category of the issue",
        "options": ["technical", "process", "resource", "quality", "risk", "compliance"]
      }
    }
  },
  
  "extraction_criteria": {
    "indicators": {
      "phrases": [
        "problem", "issue", "concern", "challenge", "risk", 
        "obstacle", "blocker", "difficulty", "trouble"
      ],
      "contexts": [
        "needs to be addressed", 
        "needs attention",
        "preventing progress"
      ]
    }
  },
  
  "workflow": {
    "enabled_stages": [
      "document_analysis", "chunking", "planning", 
      "extraction", "aggregation", "evaluation", "formatting"
    ],
    "stage_weights": {
      "document_analysis": 0.05,
      "chunking": 0.05,
      "planning": 0.1,
      "extraction": 0.3,
      "aggregation": 0.2,
      "evaluation": 0.2,
      "formatting": 0.1
    },
    "halt_on_error": false,
    "agent_instructions": {
      "planner": "Analyze the document to identify key themes and create a strategic plan for extracting issues...",
      "extractor": "Identify all potential issues, challenges, risks, and problems mentioned in the text...",
      "aggregator": "Consolidate similar issues and remove duplicates while preserving unique details...",
      "evaluator": "Assess each issue's severity, impact, and categorize appropriately...",
      "formatter": "Format the evaluated issues into a structured report following the output schema..."
    }
  },
  
  "output_schema": {
    "type": "object",
    "properties": {
      "executive_summary": {
        "type": "string",
        "description": "Brief summary of the assessment findings"
      },
      "issues": {
        "type": "array",
        "description": "List of identified issues",
        "items": {
          "type": "object",
          "properties": {
            "title": { "type": "string" },
            "description": { "type": "string" },
            "severity": { "type": "string" },
            "category": { "type": "string" },
            "impact": { "type": "string" },
            "recommendations": {
              "type": "array",
              "items": { "type": "string" }
            }
          },
          "required": ["title", "description", "severity", "category"]
        }
      },
      "statistics": {
        "type": "object",
        "properties": {
          "total_issues": { "type": "integer" },
          "by_severity": { "type": "object" },
          "by_category": { "type": "object" }
        }
      },
      "metadata": {
        "type": "object",
        "properties": {
          "document_name": { "type": "string" },
          "word_count": { "type": "integer" },
          "processing_time": { "type": "number" },
          "date_analyzed": { "type": "string" },
          "assessment_id": { "type": "string" },
          "assessment_type": { "type": "string" }
        }
      }
    },
    "required": ["executive_summary", "issues", "statistics", "metadata"]
  },
  
  "output_format": {
    "presentation": {
      "group_by": "severity",
      "sort_by": ["severity", "category"],
      "sort_order": "desc"
    },
    "sections": [
      "executive_summary", "issues_by_severity", "statistics"
    ]
  },
  
  "user_options": {
    "properties": {
      "model": {
        "type": "string",
        "title": "LLM Model",
        "description": "Language model to use for assessment",
        "default": "gpt-3.5-turbo",
        "enum": ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"]
      },
      "chunk_size": {
        "type": "integer",
        "title": "Chunk Size",
        "description": "Size of document chunks (in characters)",
        "default": 8000,
        "minimum": 1000,
        "maximum": 16000
      }
    }
  }
}
```

## 🔄 Processing Pipeline

### Orchestrator

The `Orchestrator` class is the central controller that:

1. Loads assessment configurations
2. Initializes the Processing Context
3. Manages document chunking
4. Coordinates the agent pipeline
5. Handles errors and progress reporting

```python
orchestrator = Orchestrator(assessment_id="base_assess_issue_v1", options={
    "model": "gpt-3.5-turbo",
    "chunk_size": 8000,
    "chunk_overlap": 300
})

result = await orchestrator.process_document(document)
```

### Agent Pipeline

The system uses a sequential pipeline of specialized AI agents, each with a distinct role:

1. **🧭 PlannerAgent** - Analyzes document preview and creates a strategic plan
   - Assesses document type and key topics
   - Provides guidance for extraction and evaluation focus
   - Identifies special considerations (e.g., jargon, technical content)

2. **🔍 ExtractorAgent** - Processes each document chunk to identify relevant items
   - Extracts items based on assessment type (action items, issues, key points)
   - Links each item to its source evidence
   - Casts a wide net to capture all potential candidates

3. **🧩 AggregatorAgent** - Consolidates and deduplicates extracted items
   - Merges similar/duplicate items while preserving unique details
   - Groups related items when appropriate
   - Maintains evidence links across merged items

4. **⚖️ EvaluatorAgent** - Assesses and rates consolidated items
   - Assigns severity/priority ratings
   - Validates categorization
   - Generates recommendations or impact assessments
   - Creates an overall assessment summary

5. **📊 FormatterAgent** - Transforms evaluated data into final output
   - Structures data according to the output schema
   - Applies formatting preferences (grouping, sorting)
   - Generates summary statistics and metadata

6. **🔍 ReviewerAgent** - (Optional) Performs quality control on results
   - Checks for completeness and consistency
   - Identifies potential improvements
   - Provides feedback on overall quality

## 🧠 ProcessingContext

The `ProcessingContext` class serves as the shared "brain" of the system, maintaining state across agent boundaries and providing a consistent interface for data access.

### Core Responsibilities

- **Document Management** - Stores document text and metadata
- **Chunk Management** - Maintains document chunks with position tracking
- **Data Storage** - Structured storage for each processing phase
- **Evidence Tracking** - Links extracted items to source text
- **Progress Tracking** - Monitors pipeline stage progress
- **Token Usage** - Tracks LLM token consumption

### Context Structure

```
ProcessingContext
├── document_text: str                 # Original document text
├── assessment_config: Dict            # Assessment configuration
├── options: Dict                      # Runtime options
├── assessment_id: str                 # Assessment identifier
├── assessment_type: str               # Assessment type
├── display_name: str                  # Human-readable name
├── run_id: str                        # Unique run identifier
├── document_info: Dict                # Document metadata
├── chunks: List[Dict]                 # Document chunks
├── chunk_mapping: Dict                # Maps positions to chunks
├── pipeline_state: Dict               # Processing state
│   ├── current_stage: str             # Active pipeline stage
│   ├── stages: Dict                   # Stage status & progress
│   ├── progress: float                # Overall progress (0-1)
│   └── warnings/errors: List          # Issues encountered
├── data: Dict                         # Central data store
│   ├── planning: Dict                 # Planning results
│   ├── extracted: Dict                # Extraction results
│   ├── aggregated: Dict               # Aggregation results
│   ├── evaluated: Dict                # Evaluation results
│   └── formatted: Dict                # Formatting results
├── entities: Dict                     # Entity registry
├── relationships: List                # Entity relationships
├── evidence_store: Dict               # Evidence tracking
│   ├── references: Dict               # Item → Evidence links
│   └── sources: Dict                  # Evidence source text
└── usage_metrics: Dict                # Token usage statistics
```

### Data Flow Between Agents

```
                  ┌───────────────┐
                  │ Planning Data │
                  └───────┬───────┘
                          │
                          ▼
┌───────────┐     ┌───────────────┐     ┌────────────────┐
│ Document  │────▶│   Extracted   │────▶│   Aggregated   │
│  Chunks   │     │     Items     │     │     Items      │
└───────────┘     └───────┬───────┘     └────────┬───────┘
                          │                      │
                          │                      ▼
                 ┌────────┴──────┐     ┌────────────────┐     ┌────────────────┐
                 │Evidence Store │────▶│   Evaluated    │────▶│    Formatted   │
                 └───────────────┘     │     Items      │     │     Output     │
                                       └────────────────┘     └────────────────┘
```

### Key Context Methods

```python
# Storing and retrieving agent data
context.store_agent_data("extractor", "action_items", extracted_items)
items = context.get_data_for_agent("extractor", "action_items")

# Evidence tracking
evidence_id = context.add_evidence(item_id, evidence_text, source_info)
evidence_list = context.get_evidence_for_item(item_id)

# Stage management
context.set_stage("extraction")
context.update_stage_progress(0.5, "Processed 50% of chunks")
context.complete_stage("extraction", results)

# Token tracking
context.track_token_usage(tokens)

# Final result assembly
result = context.get_final_result()
```

## 📈 Customization Points

Beyond Notes supports multiple customization approaches:

1. **Configuration-Based** - Adjust assessment parameters via JSON
   - Modify extraction criteria (keywords, indicators)
   - Customize agent instructions
   - Adjust workflow stages and weights
   - Define output schemas and formatting

2. **Template Creation** - Build specialized assessment templates
   - Inherit from base assessment types
   - Customize for specific meeting types or domains
   - Add domain-specific extraction criteria
   - Tailor output formats for specific use cases

3. **Agent Extension** - Create specialized agent implementations
   - Subclass BaseAgent for custom processing logic
   - Implement domain-specific extraction techniques
   - Add specialized post-processing stages
   - Integrate external knowledge or tools

4. **Runtime Options** - Configure behavior at runtime
   - Select LLM model (balancing cost vs. quality)
   - Adjust chunking parameters
   - Enable/disable specific processing stages
   - Set debug and logging levels

## 🔧 Technical Implementation Details

### LLM Integration

The system uses a custom LLM interface that:

- Supports OpenAI's function calling API for structured outputs
- Handles token counting and usage tracking
- Manages retry logic for API rate limits
- Includes schema validation and sanitization

```python
# CustomLLM provides a standardized interface with token tracking
result_dict, usage_dict = await self.llm.generate_structured_output(
    prompt=prompt,
    output_schema=schema,
    temperature=0.2
)

# Track token usage in the context
context.track_token_usage(usage_dict.get("total_tokens", 0))
```

### Document Chunking

Documents are processed in manageable chunks:

- Configurable chunk size (default: 8000 characters)
- Overlap between chunks (default: 300 characters)
- Position tracking for source evidence linking
- Metadata preservation across chunks

```python
chunks = chunk_document(
    document,
    target_chunk_size=8000,
    overlap=300
)

context.set_chunks(chunks)
```

### Error Handling & Resilience

The system prioritizes pipeline resilience:

- Graceful handling of LLM errors
- Continuation despite individual stage failures
- Comprehensive error and warning tracking
- Fallback strategies for critical components

### Asynchronous Processing

All operations use Python's asyncio for efficient processing:

- Asynchronous LLM API calls
- Concurrent processing where possible
- Progress callback support for real-time updates

## 🔍 Debugging & Testing

The system includes comprehensive debugging tools:

- Detailed logging at all pipeline stages
- Token usage tracking by stage
- Performance metrics collection
- Test runner with visualization
- Schema validation for all outputs

## 📋 Integration Notes

To integrate Beyond Notes into other applications:

1. **API-Based Integration**
   - Use the Orchestrator as an API endpoint
   - Pass documents and receive structured results
   - Configure assessment types via API parameters

2. **Embedded Integration**
   - Import the processing pipeline directly
   - Use the Context object to share state with host application
   - Extend the agent pipeline with application-specific agents

3. **Output Consumption**
   - Results follow predictable schemas for each assessment type
   - JSON output can be imported into other systems
   - Evidence links enable tracing back to source text