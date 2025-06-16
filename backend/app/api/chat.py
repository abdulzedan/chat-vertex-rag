from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, Dict, List
import json
import logging
import uuid
from datetime import datetime

from app.models.schemas import ChatRequest, QueryRequest
from app.services.vertex_search import VertexSearchService
from app.services.gemini_client import GeminiClient

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize search service
search_service = VertexSearchService()

# In-memory conversation storage for context
conversation_memory: Dict[str, List[Dict]] = {}

async def stream_response(generator: AsyncGenerator[str, None]) -> AsyncGenerator[bytes, None]:
    """Convert string generator to SSE format"""
    try:
        async for chunk in generator:
            # Format as Server-Sent Events
            data = json.dumps({"content": chunk})
            yield f"data: {data}\n\n".encode()
    except Exception as e:
        error_data = json.dumps({"error": str(e)})
        yield f"data: {error_data}\n\n".encode()
    finally:
        # Send done signal
        yield f"data: {json.dumps({'done': True})}\n\n".encode()

@router.post("/query")
async def query_endpoint(request: QueryRequest):
    """Query documents using Vertex AI Search + Gemini with enhanced context and streaming response"""
    try:
        logger.info(f"Received query: {request.question}")
        
        # Generate or use session ID for context
        session_id = getattr(request, 'session_id', None) or str(uuid.uuid4())
        
        # Search documents using Vertex AI Search with enhanced query
        try:
            logger.info("Searching documents with Vertex AI Search...")
            
            # Enhanced search - try multiple search strategies
            search_results = await _enhanced_document_search(
                query=request.question,
                max_results=request.similarity_top_k,
                document_ids=request.document_ids,
                session_id=session_id
            )
            
            if search_results:
                # Check if this is a generic query that needs special handling
                if len(search_results) == 1 and search_results[0].get('is_generic_query'):
                    logger.info("Handling generic query with document overview")
                    response_text = await _handle_generic_query(
                        query=request.question,
                        document_ids=request.document_ids,
                        session_id=session_id
                    )
                else:
                    logger.info(f"Found {len(search_results)} relevant documents")
                    # Generate response with conversation context
                    response_text = await _generate_contextual_response(
                        question=request.question,
                        search_results=search_results,
                        session_id=session_id
                    )
                
                # Store conversation turn
                _store_conversation_turn(session_id, request.question, response_text)
                
                # Convert to async generator for streaming
                async def response_generator():
                    # Split response into chunks for streaming effect
                    words = response_text.split(' ')
                    for i in range(0, len(words), 5):  # Send 5 words at a time
                        chunk = ' '.join(words[i:i+5]) + ' '
                        yield chunk
                        
                response_gen = response_generator()
            else:
                if request.document_ids:
                    logger.info("No documents found in selected documents")
                    # When specific documents are selected but no results found
                    response_text = f"I couldn't find information about '{request.question}' in the selected documents. The documents you've selected might not contain relevant information for this question. Try selecting different documents or search all documents."
                    
                    async def response_generator():
                        words = response_text.split(' ')
                        for i in range(0, len(words), 3):
                            chunk = ' '.join(words[i:i+3]) + ' '
                            yield chunk
                    
                    response_gen = response_generator()
                else:
                    logger.info("No documents found, using general Gemini response")
                    # Fall back to direct Gemini only when no documents are selected
                    gemini_client = GeminiClient()
                    response_gen = gemini_client.generate_stream(
                        prompt=f"You are a helpful AI assistant. The user is asking: {request.question}. Please provide a helpful response.",
                        system_instruction="You are a helpful assistant."
                    )
                
        except Exception as search_error:
            logger.error(f"Search failed: {search_error}")
            # Fall back to direct Gemini
            gemini_client = GeminiClient()
            response_gen = gemini_client.generate_stream(
                prompt=f"You are a helpful AI assistant. The user is asking: {request.question}. Please provide a helpful response.",
                system_instruction="You are a helpful assistant."
            )
        
        # Return streaming response
        return StreamingResponse(
            stream_response(response_gen),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Enhanced chat endpoint with conversation memory and better document understanding"""
    try:
        # Generate or use session ID for context
        session_id = getattr(request, 'session_id', None) or str(uuid.uuid4())
        logger.info(f"Chat request for session: {session_id}")
        
        if request.use_rag:
            # Use enhanced search + Gemini for document-based queries
            search_results = await _enhanced_document_search(
                query=request.question,
                max_results=10,
                document_ids=request.document_ids,
                session_id=session_id
            )
            
            if search_results:
                # Check if this is a generic query that needs special handling
                if len(search_results) == 1 and search_results[0].get('is_generic_query'):
                    logger.info("Handling generic query with document overview")
                    response_text = await _handle_generic_query(
                        query=request.question,
                        document_ids=request.document_ids,
                        session_id=session_id
                    )
                else:
                    response_text = await _generate_contextual_response(
                        question=request.question,
                        search_results=search_results,
                        session_id=session_id
                    )
                
                # Store conversation turn
                _store_conversation_turn(session_id, request.question, response_text)
                
                # Convert to streaming generator
                async def response_generator():
                    words = response_text.split(' ')
                    for i in range(0, len(words), 5):
                        chunk = ' '.join(words[i:i+5]) + ' '
                        yield chunk
                response_gen = response_generator()
            else:
                # No documents found
                response_text = f"I don't have any relevant documents to answer '{request.question}'. Please upload relevant documents first, or try rephrasing your question."
                
                async def response_generator():
                    words = response_text.split(' ')
                    for i in range(0, len(words), 3):
                        chunk = ' '.join(words[i:i+3]) + ' '
                        yield chunk
                response_gen = response_generator()
        else:
            # Use direct Gemini for general queries
            gemini_client = GeminiClient()
            response_gen = gemini_client.generate_stream(
                prompt=request.question,
                system_instruction="You are a helpful assistant answering questions about documents."
            )
        
        return StreamingResponse(
            stream_response(response_gen),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
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
        'tell me about this document',
        'what is this document about',
        'summarize this document',
        'what does this document contain',
        'what is in this document',
        'describe this document',
        'overview of this document',
        
        # Generic information queries
        'tell me about',
        'what can you tell me',
        'what information',
        'give me information',
        'summary',
        'overview'
    ]
    
    # Check for exact matches or partial matches
    for pattern in generic_patterns:
        if pattern in query_lower:
            return True
    
    # Check if query is very short and generic
    words = query_lower.split()
    if len(words) <= 3 and any(word in ['document', 'file', 'about', 'summary', 'overview'] for word in words):
        return True
    
    return False


async def _handle_generic_query(
    query: str,
    document_ids: List[str] = None,
    session_id: str = None
) -> str:
    """Handle generic queries by providing document overview"""
    
    # Get sample content from the selected documents to understand what's available
    try:
        # Search for common content types to see what topics are available
        topic_searches = [
            'rate pricing cost fee',
            'discount offer savings',
            'terms conditions requirements',
            'contact information phone email',
            'table data numbers',
            'dates deadlines timeline'
        ]
        
        available_topics = []
        sample_content = []
        
        for topic_query in topic_searches:
            results = await search_service.search_documents(
                query=topic_query,
                max_results=2,
                document_ids=document_ids
            )
            
            if results:
                for result in results:
                    filename = result['filename']
                    chunk_type = result.get('chunk_type', 'content')
                    
                    # Extract what kind of content is available
                    if 'pricing' in chunk_type or any(term in result['content'].lower() for term in ['rate', 'price', 'cost', '%', 'bps']):
                        available_topics.append('Pricing and rate information')
                        sample_content.append(f"Found pricing details in {filename}")
                    
                    if 'discount' in result['content'].lower() or 'off' in result['content'].lower():
                        available_topics.append('Discounts and savings offers')
                        sample_content.append(f"Found discount information in {filename}")
                    
                    if any(term in result['content'].lower() for term in ['contact', 'phone', 'email', 'call']):
                        available_topics.append('Contact information')
                        sample_content.append(f"Found contact details in {filename}")
                    
                    break  # Just check first result per topic
        
        # Remove duplicates
        available_topics = list(set(available_topics))
        sample_content = list(set(sample_content))
        
        if available_topics:
            # Get document name for response
            doc_name = "the selected document"
            if document_ids and document_ids[0] in search_service.documents:
                doc_name = search_service.documents[document_ids[0]]['filename']
            
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
    query: str,
    max_results: int,
    document_ids: List[str] = None,
    session_id: str = None
) -> List[Dict]:
    """Enhanced document search with multiple strategies and generic query handling"""
    
    # Check if this is a generic query that needs special handling
    # COMMENTED OUT: Generic query detection was interfering with specific queries
    # if _is_generic_query(query) and document_ids:
    #     logger.info(f"Detected generic query: {query}")
    #     # Return a special result that signals to handle this as a generic query
    #     return [{'is_generic_query': True, 'original_query': query}]
    
    # Strategy 1: Direct search
    search_results = await search_service.search_documents(
        query=query,
        max_results=max_results,
        document_ids=document_ids
    )
    
    # If no results, try broader search terms
    if not search_results and len(query.split()) > 2:
        logger.info("No direct results, trying broader search...")
        # Extract key terms and search again
        key_terms = [word for word in query.split() if len(word) > 3]
        if key_terms:
            broader_query = ' '.join(key_terms[:3])  # Use first 3 key terms
            search_results = await search_service.search_documents(
                query=broader_query,
                max_results=max_results,
                document_ids=document_ids
            )
    
    # If still no results, try individual key terms
    if not search_results:
        logger.info("No broader results, trying individual terms...")
        for term in query.split():
            if len(term) > 4:  # Only search meaningful terms
                term_results = await search_service.search_documents(
                    query=term,
                    max_results=max_results // 2,
                    document_ids=document_ids
                )
                search_results.extend(term_results)
                if search_results:
                    break
    
    return search_results


async def _generate_contextual_response(
    question: str,
    search_results: List[Dict],
    session_id: str
) -> str:
    """Generate response with conversation context"""
    
    # Get conversation history
    conversation_history = conversation_memory.get(session_id, [])
    
    # Prepare enhanced context
    context = ""
    for i, result in enumerate(search_results[:5], 1):
        context += f"\n--- Document {i}: {result['filename']} ---\n"
        
        # Use snippets if available, otherwise use full content
        if result.get('snippets'):
            context += "\n".join(result['snippets'])
        else:
            # Get more content for better context
            content = result['content'][:3000]  # Increased from 2000
            context += content
        context += "\n"
    
    # Build conversation context
    conversation_context = ""
    if conversation_history:
        conversation_context = "\n\nPrevious conversation context:\n"
        for turn in conversation_history[-3:]:  # Last 3 turns
            conversation_context += f"User: {turn['question']}\n"
            conversation_context += f"Assistant: {turn['response'][:200]}...\n\n"
    
    # Enhanced prompt with conversation context
    prompt = f"""You are a helpful AI assistant answering questions about uploaded documents. Use the conversation context to provide more relevant and connected responses.

Documents:
{context}
{conversation_context}
Current Question: {question}

Instructions:
- Answer based on the information provided in the documents
- Reference specific details from the documents when possible
- If this question relates to previous questions in the conversation, acknowledge that connection
- If the documents don't contain enough information, say so clearly
- Be specific about numbers, percentages, rates, and other quantitative information
- When discussing pricing or rates, include the specific values mentioned in the documents
- Cite which document(s) you're referencing

Answer:"""
    
    # Generate response
    return await search_service.generate_response(question, search_results)


def _store_conversation_turn(session_id: str, question: str, response: str):
    """Store conversation turn for context"""
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []
    
    conversation_memory[session_id].append({
        'question': question,
        'response': response,
        'timestamp': datetime.now().isoformat()
    })
    
    # Keep only last 10 turns per session
    if len(conversation_memory[session_id]) > 10:
        conversation_memory[session_id] = conversation_memory[session_id][-10:]


@router.get("/conversations/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    return {
        "session_id": session_id,
        "conversation": conversation_memory.get(session_id, [])
    }


@router.delete("/conversations/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation history for a session"""
    if session_id in conversation_memory:
        del conversation_memory[session_id]
        return {"message": f"Conversation {session_id} cleared"}
    return {"message": "Conversation not found"}