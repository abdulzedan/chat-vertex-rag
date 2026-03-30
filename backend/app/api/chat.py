import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.websocket import activity_broadcaster
from app.models.schemas import ChatRequest, QueryRequest
from app.services.gemini_client import GeminiClient
from app.services.vertex_search import VertexSearchService

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize search service
search_service = VertexSearchService()

# In-memory conversation storage for context
conversation_memory: Dict[str, List[Dict]] = {}
# Store selected documents per session
session_documents: Dict[str, List[str]] = {}


async def stream_response(
    generator: AsyncGenerator[str, None],
    citations: List[Dict] = None,
) -> AsyncGenerator[bytes, None]:
    """Convert string generator to SSE format with optional citations"""
    try:
        async for chunk in generator:
            # Format as Server-Sent Events
            data = json.dumps({"content": chunk})
            yield f"data: {data}\n\n".encode()
    except Exception as e:
        error_data = json.dumps({"error": str(e)})
        yield f"data: {error_data}\n\n".encode()
    finally:
        # Send done signal with citations if available
        done_data = {"done": True}
        if citations:
            done_data["citations"] = citations
        yield f"data: {json.dumps(done_data)}\n\n".encode()


def extract_citations(search_results: List[Dict]) -> List[Dict]:
    """Extract citation metadata from search results for display"""
    if not search_results:
        return []

    # Group by filename to avoid duplicates
    citations_by_file = {}

    for result in search_results:
        filename = result.get("filename", "Unknown")
        if filename not in citations_by_file:
            citations_by_file[filename] = {
                "filename": filename,
                "document_id": result.get("parent_document_id") or result.get("document_id"),
                "pages": set(),
                "sections": set(),
                "relevance_score": result.get("relevance_score", 0),
            }

        # Collect page numbers
        page_start = result.get("page_start")
        page_end = result.get("page_end")
        if page_start is not None:
            if page_end is not None and page_end != page_start:
                citations_by_file[filename]["pages"].add(f"{page_start}-{page_end}")
            else:
                citations_by_file[filename]["pages"].add(str(page_start))

        # Collect section hints
        section = result.get("section_hint")
        if section:
            citations_by_file[filename]["sections"].add(section)

        # Keep highest relevance score
        if result.get("relevance_score", 0) > citations_by_file[filename]["relevance_score"]:
            citations_by_file[filename]["relevance_score"] = result.get("relevance_score", 0)

    # Convert to list and format
    citations = []
    for filename, data in citations_by_file.items():
        citation = {
            "filename": filename,
            "document_id": data["document_id"],
            "relevance_score": round(data["relevance_score"], 3),
        }

        # Format pages
        if data["pages"]:
            pages = sorted(data["pages"], key=lambda x: int(x.split("-")[0]) if x.split("-")[0].isdigit() else 0)
            citation["pages"] = pages[:5]  # Limit to 5 page references

        # Format sections
        if data["sections"]:
            citation["sections"] = list(data["sections"])[:3]  # Limit to 3 sections

        citations.append(citation)

    # Sort by relevance score
    citations.sort(key=lambda x: x["relevance_score"], reverse=True)

    return citations


@router.post("/query")
async def query_endpoint(request: QueryRequest):
    """Query documents using Vertex AI Search + Gemini with enhanced context and streaming response"""
    try:
        logger.info(f"Received query: {request.question}")
        logger.info(f"Session ID: {request.session_id}")
        logger.info(f"Raw document_ids: {request.document_ids}")
        logger.info(
            f"Document count: {len(request.document_ids) if request.document_ids else 0}"
        )

        # Generate or use session ID for context
        session_id = request.session_id or str(uuid.uuid4())
        logger.info(f"Using session ID: {session_id}")

        # Track search results for citations
        search_results = []
        citations = []

        # Handle document selection persistence per session
        if request.document_ids:
            # Store selected documents for this session
            session_documents[session_id] = request.document_ids
            document_ids = request.document_ids
            logger.info(
                f"Stored {len(document_ids)} documents for session {session_id}"
            )
        else:
            # Use previously selected documents for this session if available
            document_ids = session_documents.get(session_id, None)
            if document_ids:
                logger.info(
                    f"Using {len(document_ids)} previously selected documents for session {session_id}"
                )
            else:
                logger.info(
                    f"No document selection for session {session_id} - searching all documents"
                )

        # Search documents using Vertex AI Search with enhanced query
        try:
            logger.info("Searching documents with Vertex AI Search...")
            await activity_broadcaster.clear()

            # --- Activity: Query sent ---
            query_detail_parts = [f'Query: "{request.question}"']
            if document_ids:
                query_detail_parts.append(f"Scope: {len(document_ids)} selected documents")
            datastore_mode = "V2 (native chunking)" if search_service.use_v2_datastore else "V1 (custom chunks)"
            query_detail_parts.append(f"Data store: {datastore_mode}")
            await activity_broadcaster.emit_start(
                "search",
                "Querying Discovery Engine...",
            )

            # Use optimized search with dynamic scaling based on document count
            # vertex_search.py now handles optimal result calculation automatically
            search_results = await _enhanced_document_search(
                query=request.question,
                max_results=request.similarity_top_k,  # Base limit, will be scaled automatically
                document_ids=document_ids,
                session_id=session_id,
            )

            if search_results:
                # Check if this is a generic query that needs special handling
                if len(search_results) == 1 and search_results[0].get(
                    "is_generic_query"
                ):
                    logger.info("Handling generic query with document overview")
                    response_text = await _handle_generic_query(
                        query=request.question,
                        document_ids=document_ids,
                        session_id=session_id,
                    )
                else:
                    # --- Activity: Search results ---
                    unique_docs = len({r["filename"] for r in search_results})
                    doc_names = list({r["filename"] for r in search_results})
                    top_score = max(
                        (r.get("relevance_score", 0) for r in search_results), default=0
                    )

                    search_meta = getattr(search_service, "last_search_meta", {})
                    detail_parts = []
                    # Total indexed chunks for context
                    total_indexed = sum(
                        d.get("chunk_count", 0)
                        for d in search_service.documents.values()
                    )
                    total_size = search_meta.get("total_size", 0)
                    if total_indexed and total_size:
                        detail_parts.append(
                            f"Searched {total_indexed} chunks → {total_size} matched → {len(search_results)} returned"
                        )
                    elif total_size:
                        detail_parts.append(f"{total_size} matches in index")
                    if top_score > 0:
                        detail_parts.append(f"Top relevance: {top_score:.2f}")
                    if len(doc_names) <= 3:
                        detail_parts.append(f"Sources: {', '.join(doc_names)}")
                    else:
                        detail_parts.append(
                            f"Sources: {', '.join(doc_names[:2])} +{len(doc_names)-2} more"
                        )

                    await activity_broadcaster.emit_success(
                        "search",
                        f"Found {len(search_results)} results from {unique_docs} documents",
                        detail=" · ".join(detail_parts) if detail_parts else None,
                    )
                    await asyncio.sleep(0.3)

                    # --- Activity: Query rewriting by Discovery Engine ---
                    corrected = search_meta.get("corrected_query", "")
                    expanded = search_meta.get("expanded_query", "")
                    pinned = search_meta.get("pinned_result_count", 0)

                    rewrite_lines = []
                    if corrected and corrected != request.question:
                        rewrite_lines.append(f"Spell correction: \"{request.question}\" → \"{corrected}\"")
                    if expanded and expanded != request.question and expanded != corrected:
                        line = f"Query expansion: \"{expanded}\""
                        if pinned:
                            line += f" ({pinned} original results pinned)"
                        rewrite_lines.append(line)

                    if rewrite_lines:
                        await activity_broadcaster.emit_success(
                            "rewriting",
                            "Query rewritten by Discovery Engine",
                            detail="\n".join(rewrite_lines),
                        )
                    else:
                        await activity_broadcaster.emit_success(
                            "rewriting",
                            "Query accepted as-is (no rewriting needed)",
                            detail="Spell correction: AUTO · Query expansion: AUTO (pin_unexpanded=true)",
                        )
                    await asyncio.sleep(0.3)

                    # --- Activity: Extractive answer (direct answer from DE) ---
                    for r in search_results[:3]:
                        snippets = r.get("snippets", [])
                        if snippets:
                            best_snippet = max(snippets, key=len) if snippets else ""
                            if best_snippet and len(best_snippet) > 20:
                                clean = best_snippet.replace("&#39;", "'").replace("&amp;", "&")
                                if len(clean) > 150:
                                    clean = clean[:147] + "..."
                                await activity_broadcaster.emit_success(
                                    "extractive",
                                    "Extractive answer from Discovery Engine",
                                    detail=f'"{clean}"',
                                )
                                await asyncio.sleep(0.3)
                                break

                    # --- Activity: DE Summary (if available and useful) ---
                    de_summary = search_meta.get("summary_text", "")
                    # Filter out error/empty summaries from DE
                    is_useful_summary = (
                        de_summary
                        and len(de_summary) > 80
                        and "no results" not in de_summary.lower()
                        and "try rephrasing" not in de_summary.lower()
                        and "could not" not in de_summary.lower()
                    )
                    if is_useful_summary:
                        summary_preview = de_summary[:200]
                        if len(de_summary) > 200:
                            summary_preview += "..."
                        await activity_broadcaster.emit_success(
                            "de_summary",
                            f"Discovery Engine grounded summary ({len(de_summary)} chars)",
                            detail=summary_preview,
                        )
                        await asyncio.sleep(0.3)

                    # --- Activity: Ranked results breakdown ---
                    ranking_lines = []
                    sorted_results = sorted(
                        search_results,
                        key=lambda r: r.get("relevance_score", 0),
                        reverse=True,
                    )
                    for i, r in enumerate(sorted_results[:5]):
                        score = r.get("relevance_score", 0)
                        fname = r.get("filename", "?")
                        if len(fname) > 30:
                            fname = fname[:27] + "..."
                        section = r.get("section_hint") or r.get("title", "")
                        if section and len(section) > 40:
                            section = section[:37] + "..."
                        line = f"#{i+1} [{score:.2f}] {fname}"
                        if section:
                            line += f" — {section}"
                        ranking_lines.append(line)

                    await activity_broadcaster.emit_success(
                        "ranking",
                        f"Top {min(5, len(sorted_results))} results by relevance",
                        detail="\n".join(ranking_lines),
                    )
                    await asyncio.sleep(0.3)

                    logger.info(f"Found {len(search_results)} relevant documents")

                    # --- Activity: Context assembly ---
                    total_context_chars = sum(len(r.get("content", "")) for r in search_results)
                    doc_char_counts = {}
                    for r in search_results:
                        fn = r.get("filename", "?")
                        doc_char_counts[fn] = doc_char_counts.get(fn, 0) + len(r.get("content", ""))
                    context_breakdown = " · ".join(
                        f"{fn[:25]}: {chars:,}ch" for fn, chars in list(doc_char_counts.items())[:4]
                    )
                    await activity_broadcaster.emit_success(
                        "context",
                        f"Assembled {total_context_chars:,} chars of context",
                        detail=context_breakdown,
                    )
                    await asyncio.sleep(0.3)

                    # Extract citations from search results
                    citations = extract_citations(search_results)
                    logger.info(f"Extracted {len(citations)} citations")

                    # --- Activity: Generation start ---
                    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
                    await activity_broadcaster.emit_start(
                        "generate",
                        f"Generating response with {gemini_model}...",
                    )
                    # Generate response with conversation context
                    # Use streaming response generation
                    response_gen = _generate_contextual_response_stream(
                        question=request.question,
                        search_results=search_results,
                        session_id=session_id,
                    )
            elif document_ids:
                logger.info("No documents found in selected documents")
                await activity_broadcaster.emit_warning(
                    "search", "No results found in selected documents"
                )
                # When specific documents are selected but no results found
                response_text = f"I couldn't find information about '{request.question}' in the selected documents. The documents you've selected might not contain relevant information for this question. Try selecting different documents or search all documents."

                async def response_generator():
                    words = response_text.split(" ")
                    for i in range(0, len(words), 3):
                        chunk = " ".join(words[i : i + 3]) + " "
                        yield chunk

                response_gen = response_generator()
            else:
                logger.info("No documents found, using general Gemini response")
                # Fall back to direct Gemini only when no documents are selected
                gemini_client = GeminiClient()
                response_gen = gemini_client.generate_stream(
                    prompt=f"You are a helpful AI assistant. The user is asking: {request.question}. Please provide a helpful response.",
                    system_instruction="You are a helpful assistant.",
                )

        except Exception as search_error:
            logger.error(f"Search failed: {search_error}")
            await activity_broadcaster.emit_error(
                "search",
                "Search failed — falling back to Gemini",
                detail=str(search_error)[:200],
            )
            # Fall back to direct Gemini
            gemini_client = GeminiClient()
            response_gen = gemini_client.generate_stream(
                prompt=f"You are a helpful AI assistant. The user is asking: {request.question}. Please provide a helpful response.",
                system_instruction="You are a helpful assistant.",
            )

        # Return streaming response with citations
        return StreamingResponse(
            stream_response(response_gen, citations=citations),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        logger.error(f"Error in query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Enhanced chat endpoint with conversation memory and better document understanding"""
    try:
        # Generate or use session ID for context
        session_id = request.session_id or str(uuid.uuid4())
        logger.info(f"Chat request for session: {session_id}")

        # Handle document selection persistence per session
        if hasattr(request, "document_ids") and request.document_ids:
            # Store selected documents for this session
            session_documents[session_id] = request.document_ids
            document_ids = request.document_ids
            logger.info(
                f"Stored {len(document_ids)} documents for session {session_id}"
            )
        else:
            # Use previously selected documents for this session if available
            document_ids = session_documents.get(session_id, None)
            if document_ids:
                logger.info(
                    f"Using {len(document_ids)} previously selected documents for session {session_id}"
                )
            else:
                logger.info(
                    f"No document selection for session {session_id} - searching all documents"
                )

        if request.use_rag:
            # Use enhanced search + Gemini for document-based queries
            # Dynamic scaling handled automatically by vertex_search.py
            search_results = await _enhanced_document_search(
                query=request.question,
                max_results=10,  # Base limit, will be scaled automatically
                document_ids=document_ids,
                session_id=session_id,
            )

            if search_results:
                # Check if this is a generic query that needs special handling
                if len(search_results) == 1 and search_results[0].get(
                    "is_generic_query"
                ):
                    logger.info("Handling generic query with document overview")
                    response_text = await _handle_generic_query(
                        query=request.question,
                        document_ids=document_ids,
                        session_id=session_id,
                    )
                else:
                    response_text = await _generate_contextual_response(
                        question=request.question,
                        search_results=search_results,
                        session_id=session_id,
                    )

                # Store conversation turn
                _store_conversation_turn(session_id, request.question, response_text)

                # Convert to streaming generator
                async def response_generator():
                    words = response_text.split(" ")
                    for i in range(0, len(words), 5):
                        chunk = " ".join(words[i : i + 5]) + " "
                        yield chunk

                response_gen = response_generator()
            else:
                # No documents found
                response_text = f"I don't have any relevant documents to answer '{request.question}'. Please upload relevant documents first, or try rephrasing your question."

                async def response_generator():
                    words = response_text.split(" ")
                    for i in range(0, len(words), 3):
                        chunk = " ".join(words[i : i + 3]) + " "
                        yield chunk

                response_gen = response_generator()
        else:
            # Use direct Gemini for general queries
            gemini_client = GeminiClient()
            response_gen = gemini_client.generate_stream(
                prompt=request.question,
                system_instruction="You are a helpful assistant answering questions about documents.",
            )

        return StreamingResponse(
            stream_response(response_gen),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions for enhanced chat functionality


def _is_generic_query(query: str) -> bool:
    """Detect if a query is generic and needs special handling"""
    query_lower = query.lower()

    generic_patterns = [
        # Document overview queries
        "tell me about this document",
        "what is this document about",
        "summarize this document",
        "what does this document contain",
        "what is in this document",
        "describe this document",
        "overview of this document",
        # Generic information queries
        "tell me about",
        "what can you tell me",
        "what information",
        "give me information",
        "summary",
        "overview",
    ]

    # Check for exact matches or partial matches
    for pattern in generic_patterns:
        if pattern in query_lower:
            return True

    # Check if query is very short and generic
    MAX_GENERIC_QUERY_WORDS = 3
    words = query_lower.split()
    if len(words) <= MAX_GENERIC_QUERY_WORDS and any(
        word in ["document", "file", "about", "summary", "overview"] for word in words
    ):
        return True

    return False


async def _handle_generic_query(
    _query: str, document_ids: List[str] = None, _session_id: str = None
) -> str:
    """Handle generic queries by providing document overview"""

    # Get sample content from the selected documents to understand what's available
    try:
        # Search for common content types to see what topics are available
        topic_searches = [
            "rate pricing cost fee",
            "discount offer savings",
            "terms conditions requirements",
            "contact information phone email",
            "table data numbers",
            "dates deadlines timeline",
        ]

        available_topics = []
        sample_content = []

        for topic_query in topic_searches:
            results = await search_service.search_documents(
                query=topic_query, max_results=2, document_ids=document_ids
            )

            if results:
                for result in results:
                    filename = result["filename"]
                    chunk_type = result.get("chunk_type", "content")

                    # Extract what kind of content is available
                    if "pricing" in chunk_type or any(
                        term in result["content"].lower()
                        for term in ["rate", "price", "cost", "%", "bps"]
                    ):
                        available_topics.append("Pricing and rate information")
                        sample_content.append(f"Found pricing details in {filename}")

                    if (
                        "discount" in result["content"].lower()
                        or "off" in result["content"].lower()
                    ):
                        available_topics.append("Discounts and savings offers")
                        sample_content.append(
                            f"Found discount information in {filename}"
                        )

                    if any(
                        term in result["content"].lower()
                        for term in ["contact", "phone", "email", "call"]
                    ):
                        available_topics.append("Contact information")
                        sample_content.append(f"Found contact details in {filename}")

                    break  # Just check first result per topic

        # Remove duplicates
        available_topics = list(set(available_topics))
        sample_content = list(set(sample_content))

        if available_topics:
            # Get document name for response
            doc_name = "the selected document"
            if document_ids and document_ids[0] in search_service.documents:
                doc_name = search_service.documents[document_ids[0]]["filename"]

            response = f"Based on {doc_name}, I can help you with:\\n\\n"

            for i, topic in enumerate(available_topics, 1):
                response += f"{i}. {topic}\\n"

            response += "\\nTry asking specific questions like:\\n"
            response += "• 'What are the rates?' or 'What is the base offer?'\\n"
            response += "• 'What discounts are available?'\\n"
            response += "• 'How can I contact them?'\\n"

            if sample_content:
                response += f"\\n{sample_content[0]}"

            return response
        else:
            return "I can see you've selected a document, but I need more specific questions to find relevant information. Try asking about specific topics like rates, pricing, discounts, or contact information."

    except Exception as e:
        logger.warning(f"Error handling generic query: {e}")
        return "I can help you with information from this document. Try asking specific questions about rates, pricing, discounts, or other topics you're interested in."


async def _enhanced_document_search(
    query: str, max_results: int, document_ids: List[str] = None, session_id: str = None
) -> List[Dict]:
    """Enhanced document search with conversation context and fallback strategies"""

    # Check if this is a generic query that needs special handling
    # COMMENTED OUT: Generic query detection was interfering with specific queries
    # if _is_generic_query(query) and document_ids:
    #     logger.info(f"Detected generic query: {query}")
    #     # Return a special result that signals to handle this as a generic query
    #     return [{'is_generic_query': True, 'original_query': query}]

    # Context-aware query reformulation for conversational RAG
    if session_id and conversation_memory.get(session_id):
        logger.info("Reformulating query with conversation context...")
        conversation_history = conversation_memory.get(session_id, [])
        if conversation_history:
            query = await _reformulate_query_with_context(query, conversation_history)
            logger.info(f"Reformulated query: {query}")

    # Use regular search - context handled by feeding previous responses to model
    search_results = await search_service.search_documents(
        query=query, max_results=max_results, document_ids=document_ids
    )

    return search_results


async def _generate_contextual_response_stream(
    question: str, search_results: List[Dict], session_id: str
) -> AsyncGenerator[str, None]:
    """Generate streaming response with conversation context"""

    # Get conversation history for context but don't store yet
    conversation_history = conversation_memory.get(session_id, [])

    # Build conversation context if available
    conversation_context = ""
    if conversation_history:
        recent_turn = conversation_history[-1]
        conversation_context = "\n\nPrevious conversation context:\n"
        conversation_context += f"Previous User Question: {recent_turn['question']}\n"
        conversation_context += (
            f"Previous Assistant Response: {recent_turn['response']}\n\n"
        )

    # Prepare the search results with conversation context
    full_response = ""
    async for chunk in search_service.generate_response_stream(
        question, search_results
    ):
        full_response += chunk
        yield chunk

    # Store conversation turn after streaming is complete
    _store_conversation_turn(session_id, question, full_response)

    # Signal that generation is done
    await activity_broadcaster.emit_success(
        "generate",
        "Response generated",
        detail=f"{len(full_response)} characters",
    )


async def _generate_contextual_response(
    question: str, search_results: List[Dict], session_id: str
) -> str:
    """Generate response with conversation context"""

    # Get conversation history
    conversation_history = conversation_memory.get(session_id, [])
    logger.info(
        f"Session {session_id} has {len(conversation_history)} conversation turns"
    )

    # Prepare enhanced context
    context = ""
    for i, result in enumerate(search_results[:5], 1):
        context += f"\n--- Document {i}: {result['filename']} ---\n"

        # Use snippets if available, otherwise use full content
        if result.get("snippets"):
            context += "\n".join(result["snippets"])
        else:
            # Get more content for better context
            content = result["content"][:3000]  # Increased from 2000
            context += content
        context += "\n"

    # Build conversation context - include full previous response for better context
    conversation_context = ""
    if conversation_history:
        conversation_context = "\n\nPrevious conversation context:\n"
        # For follow-up questions, include the most recent exchange with full response
        recent_turn = conversation_history[-1]  # Most recent turn
        logger.info("Including most recent conversation turn for context")
        conversation_context += f"Previous User Question: {recent_turn['question']}\n"
        conversation_context += (
            f"Previous Assistant Response: {recent_turn['response']}\n\n"
        )
        logger.debug(
            f"Previous Q: '{recent_turn['question'][:50]}...', A: '{recent_turn['response'][:100]}...'"
        )
    else:
        logger.info("No conversation history available for context")

    # Enhanced prompt with conversation context

    # Generate response using search results with conversation context included in prompt
    return await search_service.generate_response(question, search_results)


async def _reformulate_query_with_context(
    query: str, conversation_history: List[Dict]
) -> str:
    """Reformulate query to be context-aware using conversation history"""
    try:
        # Build context from recent user questions only (avoid negative feedback from assistant responses)
        previous_questions = []
        for turn in conversation_history[-2:]:  # Last 2 user questions for context
            previous_questions.append(turn["question"])

        # Use simple keyword-based reformulation for better results
        query_lower = query.lower()

        # If query has contextual references, expand with previous topics
        if (
            any(
                word in query_lower
                for word in [
                    "that",
                    "those",
                    "other",
                    "compare",
                    "them",
                    "this",
                    "these",
                ]
            )
            and previous_questions
        ):
            context_source = previous_questions[-1]
            context_words = context_source.split()
            context_snippet = " ".join(context_words[:8])
            reformulated = f"{query.strip()} (context: {context_snippet})"
            logger.debug(f"Reformulated query with context snippet: {reformulated}")
            return reformulated

        # If no context needed, return original query
        return query

    except Exception as e:
        logger.warning(f"Query reformulation failed: {e}, using original query")
        return query


def _store_conversation_turn(session_id: str, question: str, response: str):
    """Store conversation turn for context"""
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []

    conversation_memory[session_id].append(
        {
            "question": question,
            "response": response,
            "timestamp": datetime.now().isoformat(),
        }
    )

    logger.info(
        f"Stored conversation turn for session {session_id}. Total turns: {len(conversation_memory[session_id])}"
    )
    logger.debug(f"Stored Q: '{question[:50]}...', A: '{response[:50]}...'")

    # Keep only last 10 turns per session
    MAX_CONVERSATION_TURNS = 10
    if len(conversation_memory[session_id]) > MAX_CONVERSATION_TURNS:
        conversation_memory[session_id] = conversation_memory[session_id][
            -MAX_CONVERSATION_TURNS:
        ]
        logger.info(
            f"Trimmed conversation history for session {session_id} to {MAX_CONVERSATION_TURNS} turns"
        )


@router.post("/stream-answer")
async def stream_answer_endpoint(request: QueryRequest):
    """Use Discovery Engine's AnswerQuery API for one-call retrieval+generation.

    This endpoint combines search and answer generation in a single API call,
    leveraging Discovery Engine's built-in grounding. Only active when
    USE_STREAM_ANSWER=true and USE_V2_DATASTORE=true.
    """
    try:
        if not search_service.use_stream_answer or not search_service.use_v2_datastore:
            raise HTTPException(
                status_code=400,
                detail="Stream answer requires USE_STREAM_ANSWER=true and USE_V2_DATASTORE=true",
            )

        logger.info(f"Stream answer request: {request.question}")

        document_ids = request.document_ids
        if not document_ids:
            session_id = request.session_id or ""
            document_ids = session_documents.get(session_id)

        response_gen = search_service.stream_answer(
            query=request.question,
            document_ids=document_ids,
        )

        return StreamingResponse(
            stream_response(response_gen),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in stream-answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    return {
        "session_id": session_id,
        "conversation": conversation_memory.get(session_id, []),
    }


@router.delete("/conversations/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation history for a session"""
    # Clear both local memory and Vertex AI conversation context
    local_cleared = False
    vertex_cleared = False

    if session_id in conversation_memory:
        del conversation_memory[session_id]
        local_cleared = True

    # Also clear stored documents for this session
    if session_id in session_documents:
        del session_documents[session_id]
        local_cleared = True

    # Clear Vertex AI conversation context
    vertex_cleared = search_service.clear_conversation(session_id)

    if local_cleared or vertex_cleared:
        return {"message": f"Conversation {session_id} cleared"}
    return {"message": "Conversation not found"}
