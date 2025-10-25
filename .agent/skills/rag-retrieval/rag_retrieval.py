"""RAG Retrieval Skill - Hybrid search for clinical passages."""

from typing import List, Dict, Any


class RAGRetriever:
    """Hybrid search retriever using ChromaDB."""

    def __init__(self, chroma_client, collection_name: str):
        """
        Initialize retriever.

        Args:
            chroma_client: ChromaClient instance
            collection_name: Collection to query
        """
        self.chroma_client = chroma_client
        self.collection_name = collection_name

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        metadata_filter: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant passages using hybrid search.

        Args:
            query: Search query
            n_results: Number of results to return
            metadata_filter: Optional metadata filter

        Returns:
            List of result dictionaries with text, score, offsets
        """
        # Query ChromaDB (automatic hybrid search)
        results = self.chroma_client.query(
            collection_name=self.collection_name,
            query_text=query,
            n_results=n_results,
            where=metadata_filter
        )

        # Format results
        formatted_results = []
        for doc, metadata, distance in zip(
            results["documents"],
            results["metadatas"],
            results["distances"]
        ):
            formatted_results.append({
                'text': metadata.get('original_text', doc),
                'enriched_text': doc,
                'score': 1.0 - distance,  # Convert distance to similarity
                'start_offset': metadata.get('start_offset', 0),
                'end_offset': metadata.get('end_offset', len(doc)),
                'chunk_id': metadata.get('chunk_id', ''),
                'distance': distance
            })

        return formatted_results
