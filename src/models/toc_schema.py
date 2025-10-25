"""
Table of Contents (ToC) Pydantic schemas.

This module defines the data models for structured navigation of clinical notes
through section detection and offset tracking.
"""

from typing import List
from pydantic import BaseModel, Field, validator


class Section(BaseModel):
    """
    A logically distinct portion of a clinical note (e.g., HPI, Assessment, Plan).

    Attributes:
        title: Section name (e.g., "History of Present Illness", "Assessment and Plan")
        start_offset: Starting byte/character position in the original note
        end_offset: Ending byte/character position in the original note
        is_explicit: True if detected via explicit header, False if inferred via topic segmentation
        confidence: Detection confidence score (0.0-1.0)
    """
    title: str = Field(..., min_length=1, description="Section name")
    start_offset: int = Field(..., ge=0, description="Starting byte position")
    end_offset: int = Field(..., ge=0, description="Ending byte position")
    is_explicit: bool = Field(..., description="Detected via header vs inferred")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")

    @validator('end_offset')
    def validate_offsets(cls, v, values):
        """Ensure end_offset is greater than start_offset."""
        if 'start_offset' in values and v <= values['start_offset']:
            raise ValueError("end_offset must be greater than start_offset")
        return v


class TableOfContents(BaseModel):
    """
    Structured index of sections within a clinical note.

    Attributes:
        sections: Ordered list of sections (sorted by start_offset)
        total_sections: Count of sections
        navigation_enabled: Whether offsets enable direct navigation (should always be True)
    """
    sections: List[Section] = Field(..., min_items=1, description="Ordered sections")
    total_sections: int = Field(..., gt=0, description="Section count")
    navigation_enabled: bool = Field(True, description="Offsets enable navigation")

    @validator('total_sections')
    def validate_count(cls, v, values):
        """Ensure total_sections matches the actual count of sections."""
        if 'sections' in values and v != len(values['sections']):
            raise ValueError("total_sections must equal len(sections)")
        return v

    @validator('sections')
    def validate_no_overlap(cls, v):
        """
        Ensure sections do not overlap and are ordered by start_offset.

        Returns:
            List[Section]: Sorted sections by start_offset.

        Raises:
            ValueError: If overlapping sections are detected.
        """
        sorted_sections = sorted(v, key=lambda s: s.start_offset)
        for i in range(len(sorted_sections) - 1):
            if sorted_sections[i].end_offset > sorted_sections[i+1].start_offset:
                raise ValueError(
                    f"Overlapping sections detected: {sorted_sections[i].title} "
                    f"and {sorted_sections[i+1].title}"
                )
        return sorted_sections
