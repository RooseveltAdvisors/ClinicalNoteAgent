"""
Treatment Plan Pydantic schemas.

This module defines the data models for treatment recommendations including
diagnostics, therapeutics, follow-ups, with confidence scoring and hallucination guards.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator
from .summary_schema import Citation


class TreatmentRecommendation(BaseModel):
    """
    Actionable care plan item (diagnostic, therapeutic, follow-up).

    Attributes:
        recommendation_text: The recommendation
        category: Type - "diagnostic", "therapeutic", "follow-up", "risk-benefit"
        rationale: Explanation for the recommendation
        citations: Source evidence from original note
        confidence_score: Strength of evidence (0.0-1.0)
        hallucination_guard_note: Warning if weak evidence (required when confidence < 0.6)
        conflict_note: Warning if based on contradictory information
        priority_level: Priority ranking (1=highest)
    """
    recommendation_text: str = Field(..., min_length=1, description="The recommendation")
    category: str = Field(
        ...,
        regex="^(diagnostic|therapeutic|follow-up|risk-benefit)$",
        description="Type"
    )
    rationale: str = Field(..., min_length=1, description="Explanation")
    citations: List[Citation] = Field(..., min_items=1, description="Source evidence")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Evidence strength")
    hallucination_guard_note: Optional[str] = Field(None, description="Weak evidence warning")
    conflict_note: Optional[str] = Field(None, description="Contradiction warning")
    priority_level: int = Field(..., gt=0, description="Priority (1=highest)")

    @validator('hallucination_guard_note', always=True)
    def validate_hallucination_guard(cls, v, values):
        """Ensure hallucination_guard_note is present when confidence is low."""
        if 'confidence_score' in values and values['confidence_score'] < 0.6 and not v:
            raise ValueError("hallucination_guard_note required when confidence_score < 0.6")
        return v


class TreatmentPlan(BaseModel):
    """
    Collection of prioritized treatment recommendations.

    Attributes:
        recommendations: Prioritized recommendations (ordered by priority_level)
        total_recommendations: Count of recommendations
        high_confidence_ratio: Ratio of recommendations with confidence > 0.8
    """
    recommendations: List[TreatmentRecommendation] = Field(
        ...,
        min_items=3,
        description="Recommendations"
    )
    total_recommendations: int = Field(..., gt=0, description="Count")
    high_confidence_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ratio >0.8 confidence"
    )

    @validator('total_recommendations')
    def validate_count(cls, v, values):
        """Ensure total_recommendations matches the actual count."""
        if 'recommendations' in values and v != len(values['recommendations']):
            raise ValueError("total_recommendations must equal len(recommendations)")
        return v

    @validator('recommendations')
    def validate_ordering(cls, v):
        """
        Ensure recommendations are ordered by priority_level (ascending).

        Returns:
            List[TreatmentRecommendation]: Sorted recommendations by priority_level.
        """
        sorted_recs = sorted(v, key=lambda r: r.priority_level)
        return sorted_recs
