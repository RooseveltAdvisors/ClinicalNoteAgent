---
name: structured-logging
description: JSON-based structured logging for audit trails and debugging. Use for logging all agent operations, quality metrics, errors, and execution times with daily rotation and automatic cleanup.
---

# Structured Logging Skill

## Overview

This skill provides JSON-formatted logging for audit trails, debugging, and compliance monitoring. Logs are written as newline-delimited JSON to daily files (`logs/YYYY-MM-DD.json`) with automatic 30-day retention.

## When to Use

Use this skill to:
- Log agent invocations and iterations
- Track quality metrics over time
- Record errors with full stack traces
- Monitor execution times and performance
- Create audit trails for compliance
- Debug agent behavior and refinement loops

## Installation

**IMPORTANT**: This skill has its own isolated virtual environment (`.venv`) managed by `uv`. Do NOT use system Python.

Initialize the skill's environment:
```bash
# From the skill directory
cd .agent/skills/structured-logging
uv sync  # Creates .venv (no external dependencies, uses Python stdlib)
```

No external dependencies - uses Python standard library.

## Usage

**CRITICAL**: Always use `uv run` to execute code with this skill's `.venv`, NOT system Python.

### Initialize Logger

```python
# From .agent/skills/structured-logging/ directory
# Run with: uv run python -c "..."
from structured_logging import StructuredLogger

# Initialize with defaults
logger = StructuredLogger(
    log_dir="logs",          # Directory for log files
    retention_days=30        # Auto-delete logs older than 30 days
)
```

### Log Agent Operations

```python
from src.models.evaluator_schema import QualityMetrics

# Log successful operation
logger.log(
    log_level="INFO",
    agent_or_skill_name="summary_subagent",
    operation_type="invoke",
    input_summary="Clinical note: 2500 words, cardiology",
    output_summary="Summary generated: 5 key problems, 12 citations",
    execution_time_ms=45000,
    quality_metrics=QualityMetrics(
        citation_coverage=0.92,
        hallucination_rate=0.03,
        jaccard_overlap=0.75
    )
)
```

### Log Errors

```python
from src.models.evaluator_schema import ErrorDetails

# Log error with full context
logger.log(
    log_level="ERROR",
    agent_or_skill_name="ollama_client",
    operation_type="error",
    input_summary="Prompt: Generate clinical summary...",
    execution_time_ms=300500,
    error_details=ErrorDetails(
        error_reference_id="ERR-2025-A3F",
        stack_trace="Traceback (most recent call last)...",
        context={"model": "phi4:14b", "timeout": 300},
        file_paths=["src/skills/ollama_client.py"]
    )
)
```

### Log Iterative Refinement

```python
# Log each iteration in refinement loop
for iteration in range(1, 6):
    logger.log(
        log_level="INFO",
        agent_or_skill_name="main_orchestrator",
        operation_type="iterate",
        input_summary=f"Iteration {iteration}: Refining based on evaluator feedback",
        output_summary=f"Status: {'pass' if metrics_pass else 'fail'}",
        execution_time_ms=iteration_time,
        quality_metrics=current_metrics
    )
```

### Read and Filter Logs

```python
from datetime import datetime

# Read today's logs
entries = logger.read_logs()

# Read specific date
entries = logger.read_logs(date=datetime(2025, 10, 24))

# Filter by log level
errors = logger.read_logs(log_level="ERROR")

# Filter by agent
agent_logs = logger.read_logs(agent_name="evaluator_agent")
```

### Cleanup Old Logs

```python
# Manually trigger cleanup (also runs automatically)
deleted_count = logger.cleanup_old_logs()
print(f"Deleted {deleted_count} expired log files")
```

## Log Format

**File Path**: `logs/YYYY-MM-DD.json`

**Format**: Newline-delimited JSON (one entry per line)

**Example Entry**:
```json
{
  "timestamp": "2025-10-24T14:30:22Z",
  "log_level": "INFO",
  "agent_or_skill_name": "summary_subagent",
  "operation_type": "invoke",
  "input_summary": "Clinical note: 2500 words",
  "output_summary": "Summary: 5 problems, 12 citations",
  "execution_time_ms": 45000,
  "quality_metrics": {
    "citation_coverage": 0.92,
    "hallucination_rate": 0.03,
    "jaccard_overlap": 0.75
  },
  "error_details": null
}
```

## Querying Logs with jq

```bash
# Show all errors from today
cat logs/$(date +%Y-%m-%d).json | jq 'select(.log_level == "ERROR")'

# Show quality metrics for iterations ≥3
cat logs/*.json | jq 'select(.operation_type == "iterate" and .iteration_number >= 3) | .quality_metrics'

# Find error by reference ID
cat logs/*.json | jq 'select(.error_details.error_reference_id == "ERR-2025-A3F")'

# Calculate average execution time
cat logs/$(date +%Y-%m-%d).json | jq '[.execution_time_ms] | add/length'
```

## Best Practices

1. **Sanitize PHI**: Never log actual clinical content - use summaries only
2. **Include Execution Time**: Always track performance metrics
3. **Use Error References**: Generate unique error IDs (ERR-YYYY-NNN) for user-facing messages
4. **Log Quality Metrics**: Track citation coverage, hallucination rate, Jaccard overlap
5. **Rotation**: Rely on daily rotation, not manual log management
6. **Retention**: 30 days is suitable for debugging and compliance

## Integration with Agents

All agents and skills should log:
- **Start**: Before operation begins (input summary)
- **End**: After operation completes (output summary, execution time)
- **Errors**: With full stack trace and error reference ID
- **Metrics**: Quality scores for validation operations

## Implementation

See `structured_logging.py` for the full Python implementation.
