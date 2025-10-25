"""Citation Extraction Skill - Jaccard overlap validation."""

from typing import Set


class CitationExtractor:
    """Citation validation using Jaccard overlap."""

    def calculate_jaccard_overlap(self, claim_text: str, source_text: str) -> float:
        """
        Calculate Jaccard index (intersection over union).

        Args:
            claim_text: The claim being cited
            source_text: Source text from clinical note

        Returns:
            Jaccard overlap score (0.0-1.0)
        """
        # Tokenize (simple word-based)
        claim_words = set(claim_text.lower().split())
        source_words = set(source_text.lower().split())

        # Calculate intersection and union
        intersection = claim_words.intersection(source_words)
        union = claim_words.union(source_words)

        # Jaccard index
        if len(union) == 0:
            return 0.0

        jaccard = len(intersection) / len(union)
        return round(jaccard, 2)

    def extract_citation(
        self,
        claim_text: str,
        source_text: str,
        start_offset: int,
        end_offset: int
    ) -> dict:
        """
        Create citation dict with Jaccard validation.

        Args:
            claim_text: The claim
            source_text: Source text
            start_offset: Source start position
            end_offset: Source end position

        Returns:
            Citation dictionary
        """
        jaccard = self.calculate_jaccard_overlap(claim_text, source_text)

        return {
            'claim_text': claim_text,
            'source_text': source_text,
            'start_offset': start_offset,
            'end_offset': end_offset,
            'confidence': min(jaccard, 1.0),
            'jaccard_overlap': jaccard
        }
