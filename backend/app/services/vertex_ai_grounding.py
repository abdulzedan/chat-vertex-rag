import logging
import os
from typing import Any, Dict, List, Tuple

from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1alpha as discoveryengine

logger = logging.getLogger(__name__)


class VertexAIGroundingService:
    """Service for checking grounding of generated responses using Vertex AI Check Grounding API"""

    def __init__(self):
        self.project_id = os.getenv("GCP_PROJECT_ID", "main-env-demo")
        self.location = "global"

        # Initialize grounding client
        client_options = ClientOptions(
            api_endpoint=f"{self.location}-discoveryengine.googleapis.com"
        )
        self.client = discoveryengine.GroundedGenerationServiceClient(
            client_options=client_options
        )

        # Default grounding config
        self.grounding_config = "default_grounding_config"

    async def check_grounding(
        self, answer_candidate: str, facts: List[str], citation_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Check how well grounded an answer is in the provided facts

        Args:
            answer_candidate: The generated answer to check
            facts: List of facts/documents to check against
            citation_threshold: Minimum score for including citations

        Returns:
            Dictionary with grounding results
        """
        try:
            logger.info(f"Checking grounding for answer: {answer_candidate[:100]}...")

            # Prepare grounding facts
            grounding_facts = []
            for i, fact in enumerate(facts):
                grounding_fact = discoveryengine.GroundingFact(
                    fact_text=fact[:10000],
                    attributes={
                        "id": str(i),
                        "source": f"Document {i + 1}",
                    },  # Limit fact length
                )
                grounding_facts.append(grounding_fact)

            # Create check grounding request
            parent = f"projects/{self.project_id}/locations/{self.location}"
            request = discoveryengine.CheckGroundingRequest(
                grounding_config=f"{parent}/groundingConfigs/{self.grounding_config}",
                answer_candidate=answer_candidate,
                facts=grounding_facts,
                grounding_spec=discoveryengine.CheckGroundingSpec(
                    citation_threshold=citation_threshold
                ),
            )

            # Get grounding response
            response = self.client.check_grounding(request)

            # Process response
            result = {
                "support_score": (
                    response.support_score
                    if hasattr(response, "support_score")
                    else 0.0
                ),
                "cited_chunks": [],
                "claims": [],
                "answer_with_citations": answer_candidate,
            }

            # Extract cited chunks
            if hasattr(response, "cited_chunks"):
                for chunk in response.cited_chunks:
                    cited_chunk = {
                        "chunk_text": (
                            chunk.chunk_text if hasattr(chunk, "chunk_text") else ""
                        ),
                        "source": (
                            chunk.source if hasattr(chunk, "source") else "Unknown"
                        ),
                    }
                    result["cited_chunks"].append(cited_chunk)

            # Extract claims with citations
            if hasattr(response, "claims"):
                for claim in response.claims:
                    claim_info = {
                        "claim_text": (
                            claim.claim_text if hasattr(claim, "claim_text") else ""
                        ),
                        "start_pos": (
                            claim.start_pos if hasattr(claim, "start_pos") else 0
                        ),
                        "end_pos": claim.end_pos if hasattr(claim, "end_pos") else 0,
                        "citation_indices": (
                            list(claim.citation_indices)
                            if hasattr(claim, "citation_indices")
                            else []
                        ),
                    }
                    result["claims"].append(claim_info)

            # Add citations to answer
            result["answer_with_citations"] = self._add_citations_to_answer(
                answer_candidate, result["claims"]
            )

            logger.info(
                f"Grounding check complete. Support score: {result['support_score']}"
            )

            return result

        except Exception as e:
            logger.error(f"Error checking grounding: {e}")
            # Return basic result on error
            return {
                "support_score": 0.0,
                "cited_chunks": [],
                "claims": [],
                "answer_with_citations": answer_candidate,
                "error": str(e),
            }

    def _add_citations_to_answer(
        self, answer: str, claims: List[Dict[str, Any]]
    ) -> str:
        """Add citation markers to the answer text"""
        # Sort claims by position (reverse order to maintain positions)
        sorted_claims = sorted(claims, key=lambda x: x["end_pos"], reverse=True)

        result = answer
        for claim in sorted_claims:
            if claim["citation_indices"]:
                # Create citation string like [0,1,2]
                citations = "[" + ",".join(map(str, claim["citation_indices"])) + "]"
                # Insert at end position
                end_pos = claim["end_pos"]
                if 0 <= end_pos <= len(result):
                    result = result[:end_pos] + citations + result[end_pos:]

        return result

    async def validate_response(
        self,
        query: str,
        response: str,
        source_documents: List[Dict[str, Any]],
        min_support_score: float = 0.7,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate if a generated response is sufficiently grounded

        Args:
            query: The original query
            response: The generated response
            source_documents: Documents used to generate the response
            min_support_score: Minimum support score to consider grounded

        Returns:
            Tuple of (is_grounded, grounding_details)
        """
        try:
            # Extract facts from source documents
            facts = []
            for doc in source_documents:
                # Use content or snippets
                if "snippets" in doc and doc["snippets"]:
                    facts.extend(doc["snippets"])
                elif "content" in doc:
                    # Take relevant portions of content
                    content = doc["content"]
                    # Split into paragraphs and take most relevant ones
                    paragraphs = content.split("\n\n")
                    for para in paragraphs[:5]:  # Limit to first 5 paragraphs
                        if len(para.strip()) > 50:
                            facts.append(para.strip())

            if not facts:
                logger.warning("No facts extracted from source documents")
                return False, {"error": "No facts available for grounding check"}

            # Check grounding
            grounding_result = await self.check_grounding(response, facts)

            # Determine if sufficiently grounded
            is_grounded = grounding_result["support_score"] >= min_support_score

            # Add validation details
            grounding_result["is_grounded"] = is_grounded
            grounding_result["min_support_score"] = min_support_score
            grounding_result["query"] = query

            return is_grounded, grounding_result

        except Exception as e:
            logger.error(f"Error validating response: {e}")
            return False, {"error": str(e)}

    def format_grounded_response(self, grounding_result: Dict[str, Any]) -> str:
        """Format a response with grounding information"""
        response = grounding_result.get("answer_with_citations", "")
        support_score = grounding_result.get("support_score", 0.0)

        # Add grounding score as metadata
        formatted = f"{response}\n\n"
        formatted += f"*Grounding Score: {support_score:.2%}*"

        # Add source references if available
        if grounding_result.get("cited_chunks"):
            formatted += "\n\n**Sources:**"
            for i, chunk in enumerate(grounding_result["cited_chunks"]):
                source = chunk.get("source", f"Document {i + 1}")
                formatted += f"\n[{i}] {source}"

        return formatted
