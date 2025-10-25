"""
Contextual Chunking Skill - Anthropic Contextual Retrieval Implementation.

This module implements context-enriched chunking for RAG systems, improving
retrieval accuracy by 49% through LLM-generated contextual summaries.
"""

import tiktoken
from typing import List, Dict, Any, Optional


class ContextualChunker:
    """
    Chunks documents with LLM-generated context prepended to each chunk.

    Attributes:
        ollama_client: Client for Phi-4 LLM inference
        chunk_size: Target tokens per chunk
        chunk_overlap: Overlap between chunks in tokens
        context_size: Target context tokens (50-100)
        encoding: Tiktoken encoding for token counting
    """

    def __init__(
        self,
        ollama_client,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        context_size: int = 75
    ):
        """
        Initialize contextual chunker.

        Args:
            ollama_client: OllamaClient instance for context generation
            chunk_size: Target tokens per chunk (default: 1000)
            chunk_overlap: Overlap tokens (default: 200, ~20%)
            context_size: Target context tokens (default: 75, range: 50-100)
        """
        self.ollama_client = ollama_client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.context_size = context_size

        # Use GPT-4 encoding as approximation for Phi-4
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))

    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.

        Args:
            text: Full document text

        Returns:
            List of chunk dictionaries with original_text and offsets
        """
        tokens = self.encoding.encode(text)
        chunks = []
        chunk_id = 0

        start_idx = 0
        while start_idx < len(tokens):
            # Get chunk tokens
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)

            # Calculate character offsets (approximate)
            start_char = len(self.encoding.decode(tokens[:start_idx]))
            end_char = start_char + len(chunk_text)

            chunks.append({
                'chunk_id': chunk_id,
                'original_text': chunk_text,
                'start_offset': start_char,
                'end_offset': end_char,
                'token_count': len(chunk_tokens)
            })

            chunk_id += 1

            # Move start index forward (with overlap)
            start_idx += (self.chunk_size - self.chunk_overlap)

        return chunks

    def _generate_context(self, chunk_text: str, doc_context: str) -> str:
        """
        Generate contextual summary for a chunk using LLM.

        Args:
            chunk_text: The chunk to contextualize
            doc_context: First ~2000 chars of full document for context

        Returns:
            Generated context string (50-100 tokens)
        """
        prompt = f"""Given the whole document context, provide succinct context (50-100 tokens) to situate this chunk for search retrieval purposes.

Document title/type: Clinical Note
Document context: {doc_context[:2000]}

Chunk to contextualize:
{chunk_text[:500]}...

Provide ONLY the context (no explanations):"""

        try:
            result = self.ollama_client.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=self.context_size
            )
            context = result["response"].strip()
            return context

        except Exception as e:
            # Fall back to empty context on error
            print(f"Context generation failed: {e}")
            return ""

    def chunk_with_context(
        self,
        document_text: str,
        doc_id: str
    ) -> List[Dict[str, Any]]:
        """
        Chunk document and enrich each chunk with LLM-generated context.

        Args:
            document_text: Full clinical note text
            doc_id: Document identifier (for chunk IDs)

        Returns:
            List of enriched chunk dictionaries with:
                - id: Unique chunk identifier
                - original_text: Original chunk text
                - context: LLM-generated context
                - enriched_text: context + original_text
                - start_offset: Character start position
                - end_offset: Character end position
                - token_count: Token count of original text
        """
        # Step 1: Chunk the document
        chunks = self._chunk_text(document_text)

        # Step 2: Generate context for each chunk
        enriched_chunks = []
        doc_context = document_text[:2000]  # First 2000 chars for context

        for chunk in chunks:
            # Generate context
            context = self._generate_context(chunk['original_text'], doc_context)

            # Create enriched chunk
            enriched_text = f"{context}\n\n{chunk['original_text']}" if context else chunk['original_text']

            enriched_chunks.append({
                'id': f"{doc_id}_chunk_{chunk['chunk_id']}",
                'original_text': chunk['original_text'],
                'context': context,
                'enriched_text': enriched_text,
                'start_offset': chunk['start_offset'],
                'end_offset': chunk['end_offset'],
                'token_count': chunk['token_count']
            })

        return enriched_chunks
