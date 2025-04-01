# `beyond-notes` 📝✨

> *"The difference between ordinary and extraordinary is that little extra."* — Jimmy Johnson

Beyond Notes transforms meeting transcripts into structured insights by orchestrating specialized AI agents to analyze your meetings. It's what happens when you get specialized experts collaborating on understanding your meetings instead of asking a single model to do everything at once.

## Why Beyond Notes? 🤔

Meetings are packed with valuable information that often gets lost. Beyond Notes extracts this value using advanced AI techniques to create actionable insights:

- **🔍 Distill** - Create concise, focused summaries of meeting content
- **📋 Extract** - Identify and organize action items, owners, and deadlines
- **⚠️ Assess** - Uncover issues, challenges, and risks with severity ratings
- **📊 Analyze** - Evaluate content against structured frameworks like readiness assessments

## How It Works 🧠

Beyond Notes brings together several cutting-edge AI techniques:

### Multi-Agent Architecture

Multiple specialized AI agents collaborate on your document, each bringing unique expertise:

- **🧭 Planner Agent** - Orchestrates the analysis strategy based on your document
- **🔍 Extractor Agent** - Identifies relevant information from document chunks
- **🧩 Aggregator Agent** - Combines findings and removes duplicates
- **⚖️ Evaluator Agent** - Assesses importance and assigns ratings
- **📊 Formatter Agent** - Transforms insights into structured reports

### Shared Processing Context

A sophisticated `ProcessingContext` class maintains state across the agent workflow:

- Tracks extracted entities, relationships, and evidence
- Preserves source references to original text
- Manages workflow progress and metrics
- Enables agents to build upon each other's work

### Customizable Assessment Framework

A flexible configuration system allows tailoring analysis to your specific needs:

- Base assessment types for common scenarios
- User-created templates for repeated use cases
- Configurable output formats and rating systems

## Features ✨

- **🎯 Focused Analysis Types** - Choose the right tool for your needs
- **🔄 Progressive Enhancement** - Each agent builds upon previous work
- **💬 Interactive Exploration** - Chat with your analyzed documents
- **🛠️ Customizable Templates** - Create reusable configurations
- **📱 Streamlit Interface** - Clean, responsive UI for easy use
- **🔗 Evidence Tracing** - Connect insights back to source text

## Getting Started 🚀

```bash
# Clone repository
git clone https://github.com/username/beyond-notes.git
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

## Pages 📑

- **🏠 Home** - App overview and capabilities
- **⚙️ Settings** - Manage assessment types and templates
- **🔍 Analyze** - Upload and analyze documents
- **💬 Chat** - Interactive Q&A with analyzed documents
- **✏️ Refine** - Edit and export assessment results

## Built For Learning & Exploration 🔭

Beyond Notes demonstrates several advanced AI concepts:

- **🤖 Multi-Agent Orchestration** - Specialized agents collaborating effectively
- **🧠 Context Management** - Maintaining state across complex AI workflows
- **📈 Progressive Enhancement** - Building deeper understanding through staged analysis
- **🧩 Configuration-Driven Design** - Adapting analysis through structured configurations
- **🔍 Evidence-Based Analysis** - Tracing insights to source material

---

*Beyond Notes - Because your meetings deserve better than your hastily scribbled notes*