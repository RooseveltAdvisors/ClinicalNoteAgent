---
name: rag-retrieval
description: Hybrid search (embedding + BM25) for retrieving relevant clinical note passages. Use for finding source evidence to support claims in summaries and recommendations.
---

# RAG Retrieval Skill

## Overview

Retrieves relevant text passages from clinical notes using hybrid search combining dense embeddings (semantic similarity) and BM25 (keyword matching).

## When to Use

- Find source passages for clinical claims
- Retrieve evidence for treatment recommendations
- Support citation generation with relevant context

## Installation

**IMPORTANT**: This skill has its own isolated virtual environment (`.venv`) managed by `uv`. Do NOT use system Python.

Initialize the skill's environment:
```bash
# From the skill directory
cd .agent/skills/rag-retrieval
uv sync  # Creates .venv and installs dependencies from pyproject.toml
```

## Usage

**CRITICAL**: Always use `uv run` to execute code with this skill's `.venv`, NOT system Python.

```python
# From .agent/skills/rag-retrieval/ directory
# Run with: uv run python -c "..."
from rag_retrieval import RAGRetriever

retriever = RAGRetriever(chroma_client, collection_name="session_123")

# Query for relevant passages
results = retriever.retrieve(
    query="cardiovascular symptoms",
    n_results=5
)

for result in results:
    print(f"Text: {result['text']}")
    print(f"Score: {result['score']}")
    print(f"Offset: {result['start_offset']}-{result['end_offset']}")
```

## Implementation

See `rag_retrieval.py`.
