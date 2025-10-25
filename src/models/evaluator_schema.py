"""
Evaluator Agent Pydantic schemas.

This module defines the data models for quality validation feedback,
quality metrics, and iteration state tracking.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class QualityMetrics(BaseModel):
    """
    Measured quality metrics for validation.

    Attributes:
        citation_coverage: Percentage of claims with citations (0.0-1.0)
        hallucination_rate: Percentage of claims without evidence (0.0-1.0)
        jaccard_overlap: Average Jaccard index across all citations (0.0-1.0)
    """
    citation_coverage: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="% claims with citations"
    )
    hallucination_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="% claims without evidence"
    )
    jaccard_overlap: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Avg Jaccard index"
    )


class EvaluatorIssue(BaseModel):
    """
    Field-specific problem identified during validation.

    Attributes:
        field_path: JSON path to problematic field (e.g., "summary.key_problems[0].citations")
        issue_description: What's wrong with this field
        severity: Severity level - "critical", "warning", "info"
    """
    field_path: str = Field(..., description="JSON path to problematic field")
    issue_description: str = Field(..., description="What's wrong")
    severity: str = Field(
        ...,
        regex="^(critical|warning|info)$",
        description="Severity level"
    )


class EvaluatorFeedback(BaseModel):
    """
    Quality validation results with improvement suggestions.

    Attributes:
        status: Validation result - "pass" or "fail"
        iteration_number: Current iteration (1-5)
        issues: Field-specific problems found
        suggestions: Improvement recommendations
        quality_metrics: Measured metrics
    """
    status: str = Field(..., regex="^(pass|fail)$", description="Validation result")
    iteration_number: int = Field(..., ge=1, le=5, description="Current iteration")
    issues: List[EvaluatorIssue] = Field(default_factory=list, description="Problems found")
    suggestions: List[str] = Field(default_factory=list, description="Improvements")
    quality_metrics: QualityMetrics = Field(..., description="Measured metrics")

    @validator('issues', 'suggestions')
    def validate_fail_requirements(cls, v, values, field):
        """Ensure issues and suggestions are present when status is 'fail'."""
        if 'status' in values and values['status'] == 'fail' and not v:
            raise ValueError(f"{field.name} required when status='fail'")
        return v


class IterationState(BaseModel):
    """
    Tracking state for feedback loop refinement.

    Attributes:
        current_iteration: Current iteration number (0-5)
        max_iterations: Maximum iterations (always 5)
        feedback_history: All feedback from previous iterations
        refinement_actions: Actions taken in response to feedback

    State Transitions:
        - Initial: current_iteration = 0
        - After each evaluation: current_iteration += 1
        - Terminal: current_iteration == 5 OR status == "pass"
    """
    current_iteration: int = Field(..., ge=0, le=5, description="Current iteration")
    max_iterations: int = Field(5, description="Max iterations")
    feedback_history: List[EvaluatorFeedback] = Field(
        default_factory=list,
        description="Feedback"
    )
    refinement_actions: List[str] = Field(
        default_factory=list,
        description="Actions taken"
    )

    @validator('current_iteration')
    def validate_iteration_limit(cls, v, values):
        """Ensure current_iteration does not exceed max_iterations."""
        if 'max_iterations' in values and v > values['max_iterations']:
            raise ValueError("current_iteration cannot exceed max_iterations")
        return v


class ErrorDetails(BaseModel):
    """
    Error information for logging purposes.

    Attributes:
        error_reference_id: Unique error ID in format ERR-YYYY-NNN
        stack_trace: Full stack trace for debugging
        context: Request context (sanitized, no PHI)
        file_paths: Involved files (for debugging)
    """
    error_reference_id: str = Field(..., description="ERR-YYYY-NNN format")
    stack_trace: str = Field(..., description="Full stack trace")
    context: Dict[str, Any] = Field(..., description="Request context")
    file_paths: List[str] = Field(default_factory=list, description="Involved files")


class LogEntry(BaseModel):
    """
    Structured audit and debugging record.

    Attributes:
        timestamp: When logged (UTC)
        log_level: "INFO", "ERROR", "DEBUG"
        agent_or_skill_name: Source of log
        operation_type: "invoke", "iterate", "error"
        input_summary: Abbreviated input (sanitized)
        output_summary: Abbreviated output (sanitized)
        execution_time_ms: Duration in milliseconds
        quality_metrics: If applicable (for agent operations)
        error_details: If ERROR level
    """
    timestamp: str = Field(..., description="UTC timestamp")
    log_level: str = Field(..., regex="^(INFO|ERROR|DEBUG)$", description="Log level")
    agent_or_skill_name: str = Field(..., description="Source")
    operation_type: str = Field(
        ...,
        regex="^(invoke|iterate|error)$",
        description="Operation"
    )
    input_summary: Optional[str] = Field(None, description="Abbreviated input")
    output_summary: Optional[str] = Field(None, description="Abbreviated output")
    execution_time_ms: Optional[int] = Field(None, ge=0, description="Duration")
    quality_metrics: Optional[QualityMetrics] = Field(None, description="Metrics if applicable")
    error_details: Optional[ErrorDetails] = Field(None, description="Error info")

    @validator('error_details')
    def validate_error_details(cls, v, values):
        """Ensure error_details is present when log_level is ERROR."""
        if 'log_level' in values and values['log_level'] == 'ERROR' and not v:
            raise ValueError("error_details required when log_level='ERROR'")
        return v
