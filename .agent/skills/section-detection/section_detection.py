"""Section Detection Skill - ToC generation with regex + LLM."""

import re
from typing import List, Dict, Any


class SectionDetector:
    """Detect clinical note sections using hybrid regex + LLM approach."""

    # Common clinical section headers
    SECTION_PATTERNS = [
        (r'\b(Chief Complaint|CC):\s*', 'Chief Complaint'),
        (r'\b(History of Present Illness|HPI):\s*', 'History of Present Illness'),
        (r'\b(Review of Systems|ROS):\s*', 'Review of Systems'),
        (r'\b(Past Medical History|PMH):\s*', 'Past Medical History'),
        (r'\b(Medications?):\s*', 'Medications'),
        (r'\b(Allergies):\s*', 'Allergies'),
        (r'\b(Physical Exam|PE):\s*', 'Physical Exam'),
        (r'\b(Assessment and Plan|A&P|Assessment|Plan):\s*', 'Assessment and Plan'),
        (r'\b(Impression):\s*', 'Impression'),
    ]

    def __init__(self, ollama_client):
        """
        Initialize section detector.

        Args:
            ollama_client: OllamaClient for LLM-based section detection
        """
        self.ollama_client = ollama_client

    def detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect sections in clinical note.

        Args:
            text: Clinical note text

        Returns:
            List of section dictionaries with title, offsets, confidence
        """
        sections = []

        # Detect explicit sections using regex
        for pattern, title in self.SECTION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                sections.append({
                    'title': title,
                    'start_offset': match.start(),
                    'end_offset': -1,  # Will be filled by next section
                    'is_explicit': True,
                    'confidence': 0.95
                })

        # Sort by start offset
        sections = sorted(sections, key=lambda s: s['start_offset'])

        # Fill end offsets
        for i in range(len(sections) - 1):
            sections[i]['end_offset'] = sections[i + 1]['start_offset']

        # Last section extends to end of document
        if sections:
            sections[-1]['end_offset'] = len(text)

        # If no sections found, create one section for whole document
        if not sections:
            sections = [{
                'title': 'Clinical Note',
                'start_offset': 0,
                'end_offset': len(text),
                'is_explicit': False,
                'confidence': 0.5
            }]

        return sections
