# `beyond-notes`

> Transform meeting transcripts into structured insights with collaborative AI agents

## Overview

`beyond-notes` is an AI-powered application that processes meeting transcripts to extract meaningful insights, action items, and issues. It employs a multi-agent architecture where specialized AI agents collaborate to analyze documents and produce structured, consistent outputs.

The application serves two primary purposes:
1. **Practical utility** - Creating useful summaries and analyses of meeting transcripts for business teams
2. **Experimental platform** - Providing a playground for exploring advanced AI techniques and agent-based architectures

## Key Features

- **Meeting Transcript Processing** - Clean summaries and automated follow-up actions
- **Issue Extraction and Categorization** - Identify problems with severity ratings
- **Cross-Transcript Analysis** - Discover patterns across multiple calls
- **Customizable Assessment Types** - Configure and extend analysis approaches
- **Structured Outputs** - Consistent formats for easy consumption
- **Interactive UI** - Post-analysis conversation with documents

## Project Structure

```
beyond-notes/
├── app.py                      # Main Streamlit entry point
├── orchestrator.py             # Main workflow coordinator
├── core/
│   ├── processors/             # Document processing pipelines
│   ├── agents/                 # Agent definitions
│   │   ├── base.py             # Base agent class
│   │   ├── planner.py          # Orchestration agent
│   │   ├── extractor.py        # Information extraction
│   │   └── ...                 # Other specialized agents
│   ├── models/                 # Data models
│   │   ├── document.py         # Document representation
│   │   ├── analysis.py         # Analysis results
│   │   └── context.py          # Processing context
│   └── llm/                    # LLM integration
│       ├── provider.py         # Abstract LLM provider
│       └── openai.py           # OpenAI implementation
├── assessments/                # Assessment library modules
│   ├── base.py                 # Base assessment class
│   ├── registry.py             # Assessment type registry
│   ├── issues/                 # Issues assessment
│   │   ├── config.json         # Configuration
│   │   ├── schemas.py          # Data schemas
│   │   ├── processor.py        # Specialized processing
│   │   └── renderer.py         # UI components
│   ├── action_items/           # Action items assessment
│   └── insights/               # Cross-document insights
├── ui/
│   ├── pages/                  # Page components
│   ├── components/             # Reusable UI components
│   └── state.py                # UI state management
├── utils/
│   ├── paths.py                # Path management
│   ├── chunking.py             # Text chunking strategies
│   ├── formatting.py           # Output formatting
│   └── storage.py              # Data persistence
├── config/
│   ├── app.py                  # Application settings
│   └── logging.py              # Logging configuration
├── tests/                      # Testing infrastructure
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   ├── fixtures/               # Test data fixtures
│   └── conftest.py             # Test configuration
├── data/                       # Data storage
│   ├── uploads/                # User uploaded files
│   ├── cache/                  # Processing cache
│   └── samples/                # Sample transcripts for demo
├── docs/                       # Documentation
│   ├── architecture/           # System architecture docs
│   ├── user_guide/             # End-user documentation
│   └── development/            # Developer documentation
├── output/                     # Processing output
│   ├── issues/                 # Issues assessments
│   ├── action_items/           # Action item assessments
│   └── insights/               # Insights assessments
├── temp/                       # Temporary files
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Docker Compose configuration
└── requirements.txt            # Python dependencies
```

## Core Architectural Concepts

### Multi-Agent Architecture

Beyond Notes uses a team of specialized AI agents that collaborate on document analysis:

1. **Planner Agent** - Analyzes documents and creates tailored instructions for other agents
2. **Extractor Agent** - Identifies relevant information from document chunks
3. **Aggregator Agent** - Combines similar findings and eliminates duplicates
4. **Evaluator Agent** - Determines importance, severity, and relationships between findings
5. **Formatter Agent** - Creates structured, navigable reports
6. **Reviewer Agent** - Performs quality control on the final output

### Assessment Libraries

The system uses a modular "assessment library" approach where each analysis type is a self-contained module with:

- **Configuration** - Definitions, ratings, and categories
- **Schemas** - Data structures for the assessment
- **Processing** - Specialized analysis logic
- **Rendering** - UI components for displaying results

This modular approach allows for easy extension and customization.

### Processing Pipeline

Documents flow through a structured pipeline:

1. **Document Intake** - Upload and preprocessing
2. **Planning** - Analysis of document structure and content
3. **Chunking** - Division into manageable segments
4. **Extraction** - Identification of relevant information
5. **Aggregation** - Consolidation of findings
6. **Evaluation** - Assessment of importance and relevance
7. **Formatting** - Creation of structured outputs
8. **Review** - Quality assurance and verification

## Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API key

### Installation

```bash
# Clone repository
git clone https://github.com/username/beyond-notes.git
cd beyond-notes

# Set up environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env file to add your OpenAI API key

# Run the application
streamlit run app.py
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d
```

## Development Principles

1. **Structural Integrity** - Maintain the defined project structure
2. **Modular Design** - Build self-contained components with clear interfaces
3. **Progressive Enhancement** - Start with core functionality, add complexity gradually
4. **Test-Driven Development** - Write tests for core components
5. **Documentation First** - Document concepts before implementation

## License

MIT License

## Contributing

Contributions welcome! Please follow our development principles and coding standards.