"""
ChromaDB Client Skill.

This module provides a wrapper for interacting with ChromaDB
for vector storage and hybrid search (embedding + BM25).
"""

import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import uuid


class ChromaClient:
    """
    Client for ChromaDB vector database with hybrid search support.

    Attributes:
        client: ChromaDB client instance
        host: ChromaDB server host
        port: ChromaDB server port
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None
    ):
        """
        Initialize ChromaDB client.

        Args:
            host: ChromaDB server host (defaults to CHROMA_HOST env var or localhost)
            port: ChromaDB server port (defaults to 8000)
        """
        chroma_host_env = os.getenv('CHROMA_HOST', 'http://localhost:8000')
        # Extract host and port from URL if provided
        if '://' in chroma_host_env:
            # Remove protocol
            chroma_host_env = chroma_host_env.split('://')[-1]
        if ':' in chroma_host_env:
            default_host, default_port = chroma_host_env.rsplit(':', 1)
            default_port = int(default_port)
        else:
            default_host = chroma_host_env
            default_port = 8000

        self.host = host or default_host
        self.port = port or default_port

        self.client = chromadb.HttpClient(
            host=self.host,
            port=self.port
        )

    def create_collection(
        self,
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Create or get a ChromaDB collection.

        Args:
            collection_name: Name of the collection
            metadata: Optional metadata for the collection

        Returns:
            Collection instance
        """
        try:
            # Try to get existing collection first
            collection = self.client.get_collection(name=collection_name)
            return collection
        except Exception:
            # Create new collection if it doesn't exist
            collection = self.client.create_collection(
                name=collection_name,
                metadata=metadata or {}
            )
            return collection

    def add_chunks(
        self,
        collection_name: str,
        chunks: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add text chunks to a collection with automatic embeddings.

        Args:
            collection_name: Name of the collection
            chunks: List of text chunks to add
            metadatas: Optional metadata for each chunk
            ids: Optional IDs for each chunk (auto-generated if None)
        """
        collection = self.client.get_collection(name=collection_name)

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in chunks]

        # Add chunks with automatic embedding
        collection.add(
            documents=chunks,
            metadatas=metadatas or [{} for _ in chunks],
            ids=ids
        )

    def query(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query collection using hybrid search (embedding + BM25).

        Args:
            collection_name: Name of the collection
            query_text: Query string
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            Dict containing:
                - documents: List of matching document texts
                - metadatas: List of metadata for each document
                - distances: List of distance scores
                - ids: List of document IDs
        """
        collection = self.client.get_collection(name=collection_name)

        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )

        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "ids": results["ids"][0] if results["ids"] else []
        }

    def clear_collection(self, collection_name: str) -> None:
        """
        Clear all documents from a collection (session cleanup).

        Args:
            collection_name: Name of the collection to clear
        """
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            # Collection might not exist, which is fine
            pass

    def check_health(self) -> bool:
        """
        Check if ChromaDB server is accessible.

        Returns:
            bool: True if server is healthy, False otherwise
        """
        try:
            self.client.heartbeat()
            return True
        except Exception:
            return False
