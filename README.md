# `beyond-notes` 📝✨

> *""Built by humans. Reviewed by AI. Then corrected by more AI. It’s collaborative chaos - that works."*

Beyond Notes transforms documents into structured insights using a multi-agent AI system. It's an educational journey into how specialized AI agents can collaborate to extract deeper meaning than any single model could achieve on its own.

![Beyond Notes Architecture](https://raw.githubusercontent.com/kris-nale314/beyond-notes/main/docs/images/beyondLogic.png)

## Why Beyond Notes? 🤔

Standard AI approaches often miss the nuanced understanding that comes from specialized analysis. Beyond Notes addresses this through four core capabilities:

- **🔍 Distill** - Create focused summaries with progressive detail levels
- **📋 Extract** - Identify and organize action items, owners, and deadlines
- **⚠️ Assess** - Uncover issues, challenges, and risks with severity ratings
- **📊 Analyze** - Evaluate content against structured assessment frameworks

## Technical Approach 🧠

Beyond Notes implements several advanced AI techniques:

### Multi-Agent Orchestration

Five specialized AI agents form a sequential processing pipeline:

```python
async def _execute_pipeline(self, document: Document) -> None:
    # 1. Document Analysis
    await self._analyze_document(document)
    
    # 2. Document Chunking
    await self._chunk_document(document)
    
    # 3. Execute enabled stages
    for stage_name in enabled_stages:
        await self._execute_stage(stage_name)
```

- **🧭 Planner Agent** - Analyzes document structure and creates extraction strategies
- **🔍 Extractor Agent** - Identifies relevant information from document chunks
- **🧩 Aggregator Agent** - Combines findings and eliminates duplications
- **⚖️ Evaluator Agent** - Assesses information quality and importance
- **📊 Formatter Agent** - Creates the final structured output
- **🥇 Reviewer Agent** - Reviews output to validate against instructions

### Processing Context

The `ProcessingContext` class serves as the "brain" of the pipeline:

```python
class ProcessingContext:
    """
    Context manager for multi-agent document processing with improved
    data management and monitoring.
    """
```

This provides:
- Centralized data storage accessible to all agents
- Evidence tracing to link findings with source material
- Progress tracking and performance metrics
- Document chunking and metadata management

### Configuration-Driven Design

Beyond Notes uses a JSON-based configuration system to define assessment types:

```json
{
  "assessment_type": "assess",
  "assessment_id": "base_assess_issue_v1",
  "version": "1.0",
  "display_name": "Issue Assessment",
  "description": "Identifies problems, challenges, risks, and concerns in documents"
}
```

This enables customization without code changes and supports multiple workflow types.

## Features ✨

- **🧪 Progressive Enhancement** - Each agent adds deeper layers of understanding
- **🔎 Evidence-Based Analysis** - All findings are linked to source text
- **🔄 Multi-Perspective Analysis** - Examine documents from multiple angles
- **📚 Document Intelligence** - Extract structured knowledge from unstructured text
- **🧩 Modular Architecture** - Easily extensible to new assessment types
- **📊 Performance Tracking** - Detailed metrics on processing stages

## Getting Started 🚀

```bash
# Clone repository
git clone https://github.com/Kris-Nale314/beyond-notes.git
cd beyond-notes

# Set up environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY=your_api_key_here

# Run the application
streamlit run app.py
```

## Application Structure 📑

- **🏠 `app.py`** - Entry point with navigation to different tools
- **⚙️ `pages/01_Settings.py`** - Configuration management (coming soon)
- **🔍 `pages/02_Summarizer.py`** - Document summarization interface
- **⚠️ `pages/03_Issues.py`** - Issue assessment interface
- **📋 `pages/04_Chat.py`** - Chat with the document and output (coming soon)
- **📊 `pages/05_Refine.py`** - Framework for refining output with AI (coming soon)

## Core Components

- **🧠 `core/orchestrator.py`** - Coordinates the multi-agent pipeline
- **🔄 `core/models/context.py`** - Shared processing context management
- **👥 `core/agents/`** - Specialized agent implementations
- **📝 `assessments/`** - Assessment type definitions and templates
- **🛠️ `utils/`** - Shared utilities and UI components

## Educational Value 🎓

Beyond Notes demonstrates several advanced AI concepts:

- **Pipeline Architecture** - Sequential processing with specialized stages
- **Service-Based Organization** - Clear separation of responsibilities
- **Evidence Tracing** - Connecting insights to source material
- **Configuration-Driven Design** - Flexible systems without code changes
- **Document Chunking** - Handling content beyond token limits

## Use Cases & Applications 🔮

- **Meeting Transcript Analysis** - Extract key points, decisions, and action items
- **Research Paper Summarization** - Distill core findings and implications
- **Project Documentation Review** - Identify issues and recommendations
- **Contract Analysis** - Highlight important terms and considerations
- **Knowledge Management** - Transform unstructured content into structured insights

---

*Beyond Notes is both a practical tool and an educational journey into multi-agent AI systems—demonstrating how breaking complex problems into specialized tasks yields deeper insights than monolithic approaches.*