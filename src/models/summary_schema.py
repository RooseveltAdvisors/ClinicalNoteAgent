"""
Clinical Summary Pydantic schemas.

This module defines the data models for clinical summaries including patient demographics,
key problems, medications, allergies, objective findings, and citations.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class Citation(BaseModel):
    """
    Reference linking a claim to source text in the original clinical note.

    Attributes:
        claim_text: The claim being cited
        source_text: Actual text from original note that supports the claim
        start_offset: Source text starting position in original note
        end_offset: Source text ending position in original note
        confidence: Citation quality score (0.0-1.0)
        jaccard_overlap: Jaccard index measuring intersection over union (0.0-1.0)
    """
    claim_text: str = Field(..., min_length=1, description="The claim")
    source_text: str = Field(..., min_length=1, description="Source text from note")
    start_offset: int = Field(..., ge=0, description="Source start position")
    end_offset: int = Field(..., ge=0, description="Source end position")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Citation quality")
    jaccard_overlap: float = Field(..., ge=0.0, le=1.0, description="Overlap score")

    @validator('end_offset')
    def validate_offsets(cls, v, values):
        """Ensure end_offset is greater than start_offset."""
        if 'start_offset' in values and v <= values['start_offset']:
            raise ValueError("end_offset must be greater than start_offset")
        return v


class PatientSnapshot(BaseModel):
    """
    Brief summary of patient demographics.

    Attributes:
        age: Patient age (if mentioned in note)
        sex: Patient sex - values: "M", "F", "Unknown"
        chief_complaint: Primary reason for visit
    """
    age: Optional[int] = Field(None, gt=0, description="Patient age")
    sex: Optional[str] = Field(None, regex="^(M|F|Unknown)$", description="Patient sex")
    chief_complaint: Optional[str] = Field(None, description="Primary reason for visit")


class KeyProblem(BaseModel):
    """
    Clinically significant diagnosis, symptom, or condition.

    Attributes:
        problem_description: Description of the problem
        clinical_significance: Priority ranking (1=highest priority)
        citations: Source evidence from original note
    """
    problem_description: str = Field(..., min_length=1, description="Problem description")
    clinical_significance: int = Field(..., gt=0, description="Priority ranking (1=highest)")
    citations: List[Citation] = Field(..., min_items=1, description="Source evidence")


class Medication(BaseModel):
    """
    Prescribed or mentioned drug.

    Attributes:
        drug_name: Medication name
        dosage: Dose amount (e.g., "100mg")
        route: Administration route (e.g., "PO", "IV")
        frequency: Frequency (e.g., "BID", "QD")
        citations: Source evidence from original note
    """
    drug_name: str = Field(..., min_length=1, description="Medication name")
    dosage: Optional[str] = Field(None, description="Dose amount")
    route: Optional[str] = Field(None, description="Administration route")
    frequency: Optional[str] = Field(None, description="Frequency")
    citations: List[Citation] = Field(..., min_items=1, description="Source evidence")


class Allergy(BaseModel):
    """
    Documented patient allergy.

    Attributes:
        allergen: Allergen name (drug, food, environmental)
        reaction_type: Type of reaction (e.g., "rash", "anaphylaxis")
        severity: Severity level (e.g., "mild", "severe")
        citations: Source evidence from original note
    """
    allergen: str = Field(..., min_length=1, description="Allergen name")
    reaction_type: Optional[str] = Field(None, description="Reaction type")
    severity: Optional[str] = Field(None, description="Severity")
    citations: List[Citation] = Field(..., min_items=1, description="Source evidence")


class ObjectiveFinding(BaseModel):
    """
    Clinical observations, lab results, imaging findings, vital signs.

    Attributes:
        finding_description: Description of the finding
        measurement_values: Structured measurements if applicable (e.g., {"BP": "120/80"})
        citations: Source evidence from original note
    """
    finding_description: str = Field(..., min_length=1, description="Finding description")
    measurement_values: Optional[Dict[str, Any]] = Field(None, description="Structured measurements")
    citations: List[Citation] = Field(..., min_items=1, description="Source evidence")


class ClinicalSummary(BaseModel):
    """
    Hierarchical distillation of key clinical information from a clinical note.

    Attributes:
        patient_snapshot: Demographics and basic information
        key_problems: Main diagnoses/symptoms
        pertinent_history: Relevant medical history
        medications: Current medications
        allergies: Known allergies
        objective_findings: Observations/labs/imaging
        assessment: Clinical assessment summary
        conflict_notes: Warnings about contradictory information
    """
    patient_snapshot: PatientSnapshot = Field(..., description="Demographics")
    key_problems: List[KeyProblem] = Field(..., min_items=1, description="Main problems")
    pertinent_history: Optional[str] = Field(None, description="Medical history")
    medications: List[Medication] = Field(default_factory=list, description="Medications")
    allergies: List[Allergy] = Field(default_factory=list, description="Allergies")
    objective_findings: List[ObjectiveFinding] = Field(default_factory=list, description="Findings")
    assessment: str = Field(..., min_length=1, description="Clinical assessment")
    conflict_notes: List[str] = Field(default_factory=list, description="Contradictions")
