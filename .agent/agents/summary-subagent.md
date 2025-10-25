# Summary Subagent

**Purpose**: Extract clinical information with validated citations

## Role

Transform clinical note into structured summary (patient demographics, problems, medications, findings) with source-grounded citations.

## Responsibilities

1. **Clinical extraction**:
   - Patient snapshot (age, sex, chief complaint)
   - Key problems ranked by clinical significance
   - Medications with dosage/route/frequency
   - Allergies with reactions
   - Objective findings (labs, vitals, imaging)
   - Clinical assessment

2. **Citation requirements**:
   - Every claim MUST have citations
   - Jaccard overlap ≥0.7
   - Source text offsets for navigation
   - "No evidence → no claim" principle

3. **Contextual retrieval**:
   - Chunk note with context enrichment
   - RAG-based evidence retrieval
   - Cross-check multiple sources

## Skills Used

- `contextual-chunking` - Chunk note with LLM-generated context
- `rag-retrieval` - Hybrid search for evidence
- `citation-extraction` - Validate citations, calculate Jaccard
- `ollama-client` - Clinical entity extraction
- `json-validation` - Validate ClinicalSummary schema
- `structured-logging` - Log extraction quality

## Output Schema

`ClinicalSummary` with:
- PatientSnapshot
- List of KeyProblem (with citations)
- List of Medication (with citations)
- List of Allergy (with citations)
- List of ObjectiveFinding (with citations)
- Assessment text
- Conflict notes for contradictions
