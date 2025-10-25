---
name: json-validation
description: Pydantic schema validation for agent outputs. Use to validate JSON against TableOfContents, ClinicalSummary, TreatmentPlan, and EvaluatorFeedback schemas before accepting agent responses.
---

# JSON Validation Skill

## Overview

This skill provides utilities for validating JSON outputs against Pydantic schemas to ensure data integrity and schema compliance. It handles LLM output parsing, JSON extraction, and detailed error reporting.

## When to Use

Use this skill to:
- Validate agent outputs before accepting them
- Parse JSON from LLM text responses (handles embedded JSON)
- Ensure schema compliance for TableOfContents, ClinicalSummary, TreatmentPlan
- Get detailed validation error messages
- Enforce data constraints (offsets, citations, confidence scores)

## Installation

**IMPORTANT**: This skill has its own isolated virtual environment (`.venv`) managed by `uv`. Do NOT use system Python.

Initialize the skill's environment:
```bash
# From the skill directory
cd .agent/skills/json-validation
uv sync  # Creates .venv and installs dependencies from pyproject.toml
```

Dependencies are in `pyproject.toml`:
- `pydantic` - Schema validation

## Usage

**CRITICAL**: Always use `uv run` to execute code with this skill's `.venv`, NOT system Python.

### Validate Python Dictionary

```python
# From .agent/skills/json-validation/ directory
# Run with: uv run python -c "..."
from json_validation import JSONValidator

# You'll need to import schemas separately
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src/models"))
from toc_schema import TableOfContents

# Data to validate
toc_data = {
    "sections": [
        {
            "title": "History of Present Illness",
            "start_offset": 0,
            "end_offset": 450,
            "is_explicit": True,
            "confidence": 0.98
        }
    ],
    "total_sections": 1,
    "navigation_enabled": True
}

# Validate against schema
result = JSONValidator.validate(toc_data, TableOfContents)

if result['valid']:
    validated_toc = result['validated_data']
    print("Validation passed!")
    print(validated_toc)
else:
    print("Validation failed:")
    for error in result['errors']:
        print(f"  - {error}")
```

### Validate JSON String

```python
# JSON as string
toc_json = '{"sections": [...], "total_sections": 1, "navigation_enabled": true}'

result = JSONValidator.validate_json_string(toc_json, TableOfContents)

if result['valid']:
    print("Valid JSON and schema")
else:
    print(f"Errors: {result['errors']}")
```

### Parse LLM Output with Embedded JSON

```python
# LLM often returns JSON embedded in explanatory text
llm_output = """
Here's the clinical summary:

{
  "patient_snapshot": {"age": 45, "sex": "M", "chief_complaint": "Chest pain"},
  "key_problems": [...]
}

This summary covers the main clinical findings.
"""

# Extract and validate JSON automatically
from src.models.summary_schema import ClinicalSummary

result = JSONValidator.validate_llm_output(llm_output, ClinicalSummary)

if result['valid']:
    summary = result['validated_data']
    print(f"Extracted and validated summary: {summary['patient_snapshot']}")
else:
    print(f"Could not extract valid JSON: {result['errors']}")
```

### Validation with All Schemas

```python
from src.models.toc_schema import TableOfContents
from src.models.summary_schema import ClinicalSummary
from src.models.plan_schema import TreatmentPlan
from src.models.evaluator_schema import EvaluatorFeedback

# Validate different agent outputs
schemas = {
    "toc": TableOfContents,
    "summary": ClinicalSummary,
    "plan": TreatmentPlan,
    "evaluator": EvaluatorFeedback
}

for output_type, schema in schemas.items():
    result = JSONValidator.validate_llm_output(agent_outputs[output_type], schema)
    
    if not result['valid']:
        print(f"{output_type} validation failed:")
        for error in result['errors']:
            print(f"  - {error}")
```

## Common Validation Rules

**TableOfContents**:
- Sections must not overlap
- `start_offset` < `end_offset`
- Sections ordered by `start_offset`

**ClinicalSummary**:
- All citations must have `jaccard_overlap` ≥ 0.7
- `key_problems` cannot be empty
- Citation offsets must be valid

**TreatmentPlan**:
- At least 3 recommendations required
- Recommendations ordered by `priority_level`
- If `confidence_score` < 0.6, `hallucination_guard_note` required

**EvaluatorFeedback**:
- If `status == "fail"`, `issues` and `suggestions` required
- `iteration_number` must be 1-5

## Error Messages

Validation errors include field paths:

```
sections -> 0 -> end_offset: end_offset must be greater than start_offset
total_sections: total_sections must equal len(sections)
key_problems: ensure this value has at least 1 items
```

## Best Practices

1. **Always Validate**: Never accept agent outputs without validation
2. **Handle Errors Gracefully**: Retry with feedback if validation fails
3. **Log Validation Failures**: Use `structured-logging` skill to track issues
4. **Extract from LLM**: Use `validate_llm_output()` for robustness against LLM formatting
5. **Provide Feedback**: Pass validation errors to Evaluator Agent for refinement

## Integration with Iterative Refinement

```python
from src.skills.json_validation.json_validation import JSONValidator
from src.skills.ollama_client.ollama_client import OllamaClient
from src.models.summary_schema import ClinicalSummary

client = OllamaClient()
max_retries = 5

for attempt in range(1, max_retries + 1):
    # Get LLM output
    llm_response = client.generate(prompt=summary_prompt)
    
    # Validate
    result = JSONValidator.validate_llm_output(
        llm_response["response"],
        ClinicalSummary
    )
    
    if result['valid']:
        print(f"Valid summary on attempt {attempt}")
        summary = result['validated_data']
        break
    else:
        print(f"Attempt {attempt} failed validation:")
        for error in result['errors']:
            print(f"  - {error}")
        
        # Add errors to next prompt for refinement
        summary_prompt += f"\n\nPrevious attempt had errors: {result['errors']}"
else:
    print("Failed to get valid summary after max retries")
```

## Debugging Tips

1. **Check Field Paths**: Error messages include exact field location
2. **Review Schema**: Look at Pydantic models in `src/models/` for constraints
3. **Test with Minimal Data**: Start with simple valid examples
4. **Incremental Complexity**: Add fields one at a time to isolate issues

## Implementation

See `json_validation.py` for the full Python implementation.
