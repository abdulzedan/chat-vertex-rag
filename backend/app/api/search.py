import logging
from typing import List

from fastapi import APIRouter, Query

from app.services.vertex_search import VertexSearchService

router = APIRouter()
logger = logging.getLogger(__name__)

# Share the search service instance
search_service = VertexSearchService()


@router.get("/autocomplete")
async def autocomplete_endpoint(
    q: str = Query(..., min_length=1, description="Query prefix for autocomplete"),
    max_suggestions: int = Query(5, ge=1, le=10, description="Max suggestions to return"),
):
    """Get autocomplete suggestions from Discovery Engine."""
    try:
        suggestions = await search_service.autocomplete(
            query=q, max_suggestions=max_suggestions
        )
        return {"query": q, "suggestions": suggestions}
    except Exception as e:
        logger.error(f"Autocomplete error: {e}")
        return {"query": q, "suggestions": []}
