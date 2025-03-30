# `beyond-notes` 📝

> *When your meeting transcripts deserve more than just a summary*

`beyond-notes` transforms meeting transcripts into structured, actionable insights using multi-agent AI. It's what happens when specialized AI agents collaborate on understanding your meetings instead of asking a single model to do everything at once.

## Key Features of `beyond-notes`

- **Multiple Assessment Types** - Extract issues, action items, or perform SWOT analysis
- **Multi-Agent Processing** - Specialized agents for planning, extraction, aggregation, and evaluation
- **Progressive Enhancement** - Each agent builds upon previous work for deeper understanding
- **Configurable Assessments** - Customize extraction targets through JSON configuration
- **Clean Output Formats** - Structured reports in HTML, Markdown, or JSON

## How It Works

`beyond-notes` uses a "progressive enhancement" approach where specialized agents collaborate:

1. **🧠 Planner Agent** - Creates a tailored analysis strategy based on document content
2. **🔍 Extractor Agent** - Identifies relevant information from document chunks
3. **🧩 Aggregator Agent** - Combines findings and removes duplicates
4. **⚖️ Evaluator Agent** - Assesses importance and assigns ratings
5. **📊 Formatter Agent** - Transforms insights into structured reports

The shared `ProcessingContext` enables agents to build upon each other's work, tracking metrics throughout processing.

## Assessment Library

`beyond-notes` includes three assessment types:

- **Issues** - Identify problems, challenges, and risks with severity ratings
- **Action Items** - Extract concrete follow-up tasks with ownership and timing
- **Insights** - Perform SWOT analysis with client context integration

Each assessment is configurable through JSON files in the `assessments` directory.

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
│   ├── loader.py                 # Base assessment class
│   ├── issues/                 # Issues assessment
│   │   ├── config.json         # Configuration
│   │   └── schemas.py          # Data schemas
│   ├── action_items/           # Action items assessment
│   └── insights/               # Cross-document insights
├── ui/
│   ├── components/             # Reusable UI components
│   └── state.py                # UI state management
├── utils/
│   ├── paths.py                # Path management
│   ├── chunking.py             # Text chunking strategies
│   ├── formatting.py           # Output formatting
│   └── storage.py              # Data persistence
├── config/
├── tests/                      # Testing infrastructure
├── pages/                      # Streamlit pages
├── data/                       # Data storage
│   ├── uploads/                # User uploaded files
│   ├── cache/                  # Processing cache
│   └── samples/                # Sample transcripts for demo
├── docs/                       # Documentation
│   ├── architecture/           # System architecture docs
│   ├── user_guide/             # End-user documentation
│   └── development/            # Developer documentation
├── output/                     # Processing output
├── temp/                       # Temporary files
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Docker Compose configuration
└── requirements.txt            # Python dependencies
```


## Getting Started

```bash
# Clone repository
git clone https://github.com/username/beyond-notes.git
cd beyond-notes

# Set up environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Set API key (never store in code)
export OPENAI_API_KEY=your_api_key_here

# Run the application
streamlit run app.py
```

## Built For Learning & Exploration

Beyond Notes demonstrates several advanced AI concepts:

- **Multi-Agent Architectures** - Specialized agents that collaborate effectively
- **Context Management** - Maintaining state across complex AI workflows
- **Progressive Enhancement** - Building deeper understanding through staged analysis
- **Configurable Assessments** - Adapting extraction for different document types
- **Metrics Tracking** - Measuring performance throughout processing

---

*Beyond Notes - Because your meetings contained actual insights (they were just hiding)*