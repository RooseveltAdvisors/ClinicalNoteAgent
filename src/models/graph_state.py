"""
Graph state management for clinical notes multi-agent processing.

This module defines the state schema used by the LangGraph orchestration system
to track clinical note processing across multiple agents and iterations.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState


class ClinicalNoteState(MessagesState):
    """
    State schema for clinical notes processing graph.

    Extends MessagesState to maintain conversation history between agents
    while adding clinical note-specific tracking fields.

    Attributes:
        clinical_note_path: Path to uploaded clinical note file
        clinical_note_text: Full text content of the note
        collection_name: ChromaDB collection name for this session (for cleanup)

        current_phase: Processing phase - one of:
            - "toc_generation" - Table of Contents creation
            - "summary_generation" - Clinical summary extraction
            - "recommendation_generation" - Treatment plan creation
            - "evaluation" - Quality validation
            - "complete" - Processing finished

        iteration_count: Current refinement iteration (0-5)
        max_iterations: Maximum allowed iterations (default: 5)

        toc_output: Generated Table of Contents (JSON)
        summary_output: Generated Clinical Summary (JSON)
        plan_output: Generated Treatment Plan (JSON)

        evaluator_feedback: Latest feedback from Evaluator agent
        evaluation_status: "pass" or "fail" from latest evaluation
        quality_metrics: Latest quality metrics (citation coverage, hallucination rate, etc.)

        temp_file_paths: List of temporary file paths to cleanup after processing
        error_occurred: Flag indicating if processing failed
        error_message: Error details if error_occurred is True
    """

    # Clinical note input
    clinical_note_path: str = Field(..., description="Path to uploaded .txt file")
    clinical_note_text: str = Field(..., description="Full note text content")
    collection_name: str = Field(..., description="ChromaDB collection name for session")
    session_id: Optional[str] = Field(None, description="Session ID for observability")

    # Processing state
    current_phase: str = Field(
        default="toc_generation",
        description="Current processing phase"
    )
    iteration_count: int = Field(default=0, ge=0, le=5, description="Current iteration")
    max_iterations: int = Field(default=5, description="Maximum iterations")

    # Generated outputs (JSON strings)
    toc_output: Optional[str] = Field(None, description="Table of Contents JSON")
    summary_output: Optional[str] = Field(None, description="Clinical Summary JSON")
    plan_output: Optional[str] = Field(None, description="Treatment Plan JSON")

    # Evaluation tracking
    evaluator_feedback: Optional[Dict[str, Any]] = Field(
        None,
        description="Latest evaluator feedback"
    )
    evaluation_status: Optional[str] = Field(None, description="pass or fail")
    quality_metrics: Optional[Dict[str, float]] = Field(
        None,
        description="Quality metrics from evaluator"
    )

    # Session management
    temp_file_paths: List[str] = Field(
        default_factory=list,
        description="Temporary files to cleanup"
    )
    error_occurred: bool = Field(default=False, description="Processing error flag")
    error_message: Optional[str] = Field(None, description="Error details")


class SessionMetadata(BaseModel):
    """
    Session metadata for tracking and cleanup.

    Attributes:
        session_id: Unique session identifier
        upload_timestamp: When file was uploaded (ISO format)
        collection_name: ChromaDB collection name
        input_file_path: Path to uploaded clinical note
        output_file_paths: Paths to generated output files
        log_file_path: Path to session log file
        processing_time_ms: Total processing time in milliseconds
    """

    session_id: str = Field(..., description="Unique session ID")
    upload_timestamp: str = Field(..., description="Upload time (ISO format)")
    collection_name: str = Field(..., description="ChromaDB collection name")
    input_file_path: str = Field(..., description="Uploaded note path")
    output_file_paths: List[str] = Field(
        default_factory=list,
        description="Generated output files"
    )
    log_file_path: Optional[str] = Field(None, description="Session log path")
    processing_time_ms: Optional[int] = Field(None, description="Processing duration")
