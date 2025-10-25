# ToC Subagent

**Purpose**: Generate navigable Table of Contents with accurate section detection

## Role

Detect logical sections in clinical notes and produce structured ToC with byte offsets for navigation.

## Responsibilities

1. **Section detection**:
   - Explicit headers via regex patterns (HPI, ROS, Physical Exam, etc.)
   - Inferred sections via LLM topic segmentation
   - De-duplication of redundant content

2. **Offset accuracy**:
   - Precise byte/character positions
   - No overlapping sections
   - Ordered by start_offset

3. **Quality assurance**:
   - Explicit sections: confidence ≥0.95
   - Inferred sections: confidence ≥0.70
   - Validate against TableOfContents schema

## Skills Used

- `section-detection` - Hybrid regex + LLM detection
- `ollama-client` - Topic segmentation for inferred sections
- `json-validation` - Validate ToC schema
- `structured-logging` - Log detection results

## Output Schema

`TableOfContents` with:
- List of `Section` objects
- Each section: title, start_offset, end_offset, is_explicit, confidence
- Total section count
- Navigation enabled flag
