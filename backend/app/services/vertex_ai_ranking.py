import logging
import os
from typing import Any, Dict, List, Optional

from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine

logger = logging.getLogger(__name__)


class VertexAIRankingService:
    """Service for re-ranking search results using Vertex AI Ranking API"""

    def __init__(self):
        self.project_id = os.getenv("GCP_PROJECT_ID", "main-env-demo")
        self.location = "global"

        # Initialize ranking client
        client_options = ClientOptions(
            api_endpoint=f"{self.location}-discoveryengine.googleapis.com"
        )
        self.client = discoveryengine.RankServiceClient(client_options=client_options)

        # Default ranking config
        self.ranking_config = "default_ranking_config"
        self.model = "semantic-ranker-512@latest"

    async def rerank_results(
        self, query: str, documents: List[Dict[str, Any]], top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank search results using Vertex AI Ranking API

        Args:
            query: The search query
            documents: List of documents to re-rank
            top_n: Number of top results to return (default: all)

        Returns:
            Re-ranked list of documents
        """
        try:
            if not documents:
                return []

            logger.info(f"Re-ranking {len(documents)} documents for query: {query}")

            # Prepare ranking records
            records = []
            for i, doc in enumerate(documents):
                record = discoveryengine.RankingRecord(
                    id=str(i),
                    title=doc.get("title", doc.get("filename", "")),
                    content=doc.get("content", "")[:5000],  # Limit content length
                )
                records.append(record)

            # Create ranking request
            parent = f"projects/{self.project_id}/locations/{self.location}"
            request = discoveryengine.RankRequest(
                ranking_config=f"{parent}/rankingConfigs/{self.ranking_config}",
                model=self.model,
                query=query,
                records=records,
                top_n=top_n or len(records),
            )

            # Get ranking response
            response = self.client.rank(request)

            # Sort documents based on ranking scores
            ranked_indices = []
            for record in response.records:
                idx = int(record.id)
                score = record.score if hasattr(record, "score") else 0.0
                ranked_indices.append((idx, score))

            # Sort by score descending
            ranked_indices.sort(key=lambda x: x[1], reverse=True)

            # Return re-ranked documents
            reranked_docs = []
            for idx, score in ranked_indices:
                doc = documents[idx].copy()
                doc["ranking_score"] = score
                reranked_docs.append(doc)

            logger.info(
                f"Re-ranking complete. Top score: {ranked_indices[0][1] if ranked_indices else 0}"
            )

            return reranked_docs[:top_n] if top_n else reranked_docs

        except Exception as e:
            logger.error(f"Error re-ranking results: {e}")
            # Fallback to original order
            return documents[:top_n] if top_n else documents

    async def rerank_chunks(
        self,
        query: str,
        chunks: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        top_n: Optional[int] = 5,
    ) -> List[tuple[str, Dict[str, Any]]]:
        """
        Re-rank text chunks for RAG applications

        Args:
            query: The search query
            chunks: List of text chunks
            metadata: Optional metadata for each chunk
            top_n: Number of top chunks to return

        Returns:
            List of (chunk_text, metadata) tuples, re-ranked
        """
        try:
            if not chunks:
                return []

            # Convert chunks to documents format
            documents = []
            for i, chunk in enumerate(chunks):
                doc = {"content": chunk, "title": f"Chunk {i+1}"}
                if metadata and i < len(metadata):
                    doc.update(metadata[i])
                documents.append(doc)

            # Re-rank
            reranked_docs = await self.rerank_results(query, documents, top_n)

            # Convert back to chunks format
            reranked_chunks = []
            for doc in reranked_docs:
                chunk_text = doc["content"]
                chunk_metadata = {k: v for k, v in doc.items() if k != "content"}
                reranked_chunks.append((chunk_text, chunk_metadata))

            return reranked_chunks

        except Exception as e:
            logger.error(f"Error re-ranking chunks: {e}")
            # Fallback to original chunks
            result = []
            for i, chunk in enumerate(chunks[:top_n] if top_n else chunks):
                meta = metadata[i] if metadata and i < len(metadata) else {}
                result.append((chunk, meta))
            return result
