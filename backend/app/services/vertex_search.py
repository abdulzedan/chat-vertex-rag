import logging
import os
import random
import time
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, Dict, List, Optional

from google.api_core.exceptions import GoogleAPIError, ServiceUnavailable
from google.cloud import discoveryengine_v1 as discoveryengine
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from vertexai.generative_models import GenerativeModel

# Optional Vertex AI Builder APIs
try:
    from app.services.vertex_ai_grounding import VertexAIGroundingService
    from app.services.vertex_ai_ranking import VertexAIRankingService

    VERTEX_AI_BUILDER_AVAILABLE = True
except ImportError:
    VERTEX_AI_BUILDER_AVAILABLE = False
    VertexAIRankingService = None
    VertexAIGroundingService = None

logger = logging.getLogger(__name__)

# Constants for search and context processing
MAX_CONTEXT_CHARS = 900000
CONTEXT_PREVIEW_CHARS = 1000
SHORT_QUERY_WORDS = 3


class VertexSearchService:
    """Fast document indexing and retrieval using Vertex AI Search with enhanced local fallback"""

    def __init__(self):
        self.project_id = os.getenv("GCP_PROJECT_ID", "your-project-id")
        self.location = "global"  # Vertex AI Search uses global location
        self.data_store_id = os.getenv(
            "VERTEX_SEARCH_DATASTORE_ID", "rag-demo-datastore"
        )
        self.app_id = os.getenv(
            "VERTEX_SEARCH_APP_ID", "rag-demo-app"
        )  # Enterprise app ID
        self.serving_config_id = "default_config"

        # In-memory storage for document metadata only (not for search)
        self.documents = {}  # document_id -> document_data

        # Conversation storage for context management
        self.conversations = {}  # session_id -> conversation_id

        # Optional Vertex AI Builder APIs
        self.use_ranking = (
            VERTEX_AI_BUILDER_AVAILABLE
            and os.getenv("USE_VERTEX_RANKING", "false").lower() == "true"
        )
        self.use_grounding = (
            VERTEX_AI_BUILDER_AVAILABLE
            and os.getenv("USE_VERTEX_GROUNDING", "false").lower() == "true"
        )

        if self.use_ranking:
            self.ranking_service = VertexAIRankingService()
            logger.info("Vertex AI Ranking enabled")
        else:
            self.ranking_service = None

        if self.use_grounding:
            self.grounding_service = VertexAIGroundingService()
            logger.info("Vertex AI Grounding enabled")
        else:
            self.grounding_service = None

        # Initialize clients
        self.client = discoveryengine.DocumentServiceClient()
        self.search_client = discoveryengine.SearchServiceClient()
        self.conversation_client = discoveryengine.ConversationalSearchServiceClient()
        self.data_store_client = discoveryengine.DataStoreServiceClient()

        # Initialize data store
        self._ensure_data_store_exists()

    def _ensure_data_store_exists(self):
        """Create data store if it doesn't exist"""
        try:
            # Check if data store exists
            data_store_path = self.data_store_client.data_store_path(
                project=self.project_id,
                location=self.location,
                data_store=self.data_store_id,
            )

            try:
                self.data_store_client.get_data_store(name=data_store_path)
                logger.info(f"Using existing data store: {self.data_store_id}")
                return
            except Exception:
                logger.info(
                    f"Data store {self.data_store_id} doesn't exist, will create..."
                )

            # Create data store
            collection_path = self.data_store_client.collection_path(
                project=self.project_id,
                location=self.location,
                collection="default_collection",
            )

            data_store = discoveryengine.DataStore(
                display_name="RAG Demo Data Store",
                industry_vertical=discoveryengine.IndustryVertical.GENERIC,
                solution_types=[discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH],
                content_config=discoveryengine.DataStore.ContentConfig.CONTENT_REQUIRED,
            )

            try:
                self.data_store_client.create_data_store(
                    parent=collection_path,
                    data_store=data_store,
                    data_store_id=self.data_store_id,
                )
                logger.info(f"Creating data store: {self.data_store_id}")
                # Note: Data store creation is async, but we can start using it immediately
            except GoogleAPIError as e:
                if "already exists" in str(e).lower():
                    logger.info(
                        f"Data store {self.data_store_id} already exists, continuing..."
                    )
                else:
                    raise

        except Exception as e:
            logger.error(f"Error in data store initialization: {e}")
            # Don't raise here - continue with existing setup
            logger.info("Continuing with existing data store configuration...")

    async def generate_response_stream(
        self, query: str, search_results: List[Dict[str, Any]]
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response using Gemini with search results"""
        try:
            # Group search results by source document and prepare context
            grouped_results = {}

            # Dynamic context window based on document diversity
            unique_docs = len({r["filename"] for r in search_results})
            max_results_to_use = self._calculate_optimal_context_size(
                unique_docs, len(search_results)
            )
            logger.info(
                f"Using {max_results_to_use} search results from {unique_docs} unique documents"
            )

            # Ensure document diversity in selected results
            selected_results = self._ensure_document_diversity(
                search_results[:max_results_to_use], unique_docs
            )

            # Group selected results by document
            for result in selected_results:
                filename = result["filename"]
                if filename not in grouped_results:
                    grouped_results[filename] = []
                grouped_results[filename].append(result)

            context = ""
            doc_counter = 1
            for filename, doc_results in grouped_results.items():
                context += f"\n--- Document {doc_counter}: {filename} ---\n"
                for chunk_result in doc_results:
                    content = chunk_result["content"]
                    context += content + "\n\n"
                doc_counter += 1
                context += "\n"

            if not context.strip():
                yield "I couldn't find any relevant information in the uploaded documents to answer your question."
                return

            # Check context size and truncate if needed
            if len(context) > MAX_CONTEXT_CHARS:
                context = self._intelligently_truncate_context(
                    grouped_results, MAX_CONTEXT_CHARS
                )

            prompt = f"""Based on the following documents, answer the user's question comprehensively.

Documents:
{context}

Question: {query}

Please provide a complete answer based on the information in these documents. Include all relevant details, numbers, and context that would be helpful to fully address the question."""

            # Use streaming generation
            model = GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001"))

            generation_config = {
                "max_output_tokens": 8192,
                "temperature": 0.2,
                "top_p": 0.8,
            }

            response_stream = model.generate_content(
                prompt, generation_config=generation_config, stream=True
            )

            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            yield f"Error generating response: {str(e)}"

    async def generate_response_direct(self, prompt: str) -> str:
        """Generate a direct response using Gemini without RAG context"""
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location="us-central1")

            # Use the same model as the main generation
            model = GenerativeModel("gemini-2.0-flash-001")

            response = model.generate_content(prompt)
            return response.text

        except Exception as e:
            logger.error(f"Direct response generation failed: {e}")
            return prompt  # Fallback to original if reformulation fails

    async def index_document(
        self,
        document_id: str,
        filename: str,
        text_content: str,
        chunks: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> str:
        """Index a document with chunk-based semantic indexing"""
        try:
            logger.info(f"Indexing document with semantic chunks: {filename}")

            # Store in memory immediately for fallback
            doc_metadata = metadata or {}
            document_data = {
                "id": document_id,
                "filename": filename,
                "content": text_content,
                "chunks": chunks,
                "metadata": doc_metadata,
                "upload_time": datetime.now().isoformat(),
            }
            self.documents[document_id] = document_data
            logger.info(f"Document stored in memory: {document_id}")

            # Index in Vertex AI Search with chunk-based strategy
            try:
                branch_path = self.client.branch_path(
                    project=self.project_id,
                    location=self.location,
                    data_store=self.data_store_id,
                    branch="default_branch",
                )

                # Index individual chunks as separate documents (no main document)
                chunk_ids = await self._index_document_chunks(
                    document_id, filename, chunks, doc_metadata, branch_path
                )

                logger.info(
                    f"Document indexed as {len(chunk_ids)} semantic chunks in Vertex AI Search: {document_id}"
                )

            except Exception as search_error:
                logger.warning(f"Vertex AI Search indexing failed: {search_error}")
                raise

            logger.info(f"Document indexed successfully: {document_id}")
            return document_id

        except Exception as e:
            logger.error(f"Error indexing document {filename}: {e}")
            raise

    async def search_documents_with_conversation(
        self,
        query: str,
        session_id: str,
        max_results: int = 10,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search using Conversation API for context-aware multi-turn search"""
        try:
            logger.info(f"Conversational search for: {query} (session: {session_id})")

            # Get or create conversation for this session
            conversation_id = self.conversations.get(session_id)

            # Build serving config path using dataStore format (required for Conversation API)
            serving_config_path = f"projects/{self.project_id}/locations/{self.location}/collections/default_collection/dataStores/{self.data_store_id}/servingConfigs/{self.serving_config_id}"

            # Use existing conversation ID if available, otherwise create new one
            if conversation_id:
                conversation_path = f"projects/{self.project_id}/locations/{self.location}/collections/default_collection/dataStores/{self.data_store_id}/conversations/{conversation_id}"
                logger.info(f"Using existing conversation ID: {conversation_id}")
            else:
                conversation_path = f"projects/{self.project_id}/locations/{self.location}/collections/default_collection/dataStores/{self.data_store_id}/conversations/-"
                logger.info("Creating new conversation (auto-session mode)")

            # Create search specification with document filtering if needed
            discoveryengine.SearchRequest.SearchAsYouTypeSpec()

            # Build filter for specific documents if provided
            filter_expression = None
            if document_ids:
                # Create filter using ANY function for parent_document_id field
                doc_id_list = ", ".join([f'"{doc_id}"' for doc_id in document_ids])
                filter_expression = f"parent_document_id: ANY({doc_id_list})"
                logger.info(f"Applying document filter: {filter_expression}")

                # When filtering by documents, make the query broader to ensure coverage
                # The semantic search might be too restrictive when combined with document filters
                if (
                    len(query.split()) <= SHORT_QUERY_WORDS
                ):  # Short queries might be too specific
                    logger.info(
                        "Query is short - will search broadly within filtered documents"
                    )

            # Create conversation request
            # Request more results from conversation API to ensure better document coverage
            # We'll filter and limit after getting results
            expanded_results = max_results * 3 if document_ids else max_results
            logger.info(
                f"Requesting {expanded_results} results from conversation API (max_results={max_results}, has_document_filter={bool(document_ids)})"
            )

            request = discoveryengine.ConverseConversationRequest(
                name=conversation_path,
                query=discoveryengine.TextInput(input=query),
                serving_config=serving_config_path,
                safe_search=True,
                summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
                    summary_result_count=expanded_results,  # Request more results for better coverage
                    include_citations=True,
                    ignore_adversarial_query=True,
                ),
            )

            # Add boost spec for better ranking if available
            if hasattr(discoveryengine.SearchRequest, "BoostSpec"):
                logger.info("Adding boost specification for better ranking")
                request.boost_spec = discoveryengine.SearchRequest.BoostSpec()

            # Add document filter if we have one
            if filter_expression:
                request.filter = filter_expression
                logger.info(
                    f"Added pre-search filter to conversation request: {filter_expression}"
                )

            logger.info("Making conversational search request...")

            # Perform conversational search
            response = self.conversation_client.converse_conversation(request)

            # Store conversation ID for future use
            if hasattr(response, "conversation") and response.conversation.name:
                conversation_id = response.conversation.name.split("/")[-1]
                self.conversations[session_id] = conversation_id
                logger.info(
                    f"Stored conversation ID for session {session_id}: {conversation_id}"
                )

            # Process conversation response
            search_results = []

            # Extract search results from conversation response
            if hasattr(response, "search_results"):
                all_results = []
                filtered_results = []

                for result in response.search_results:
                    doc_data = {}
                    if result.document.struct_data:
                        # struct_data is a MapComposite that already converts values on access
                        doc_data = dict(result.document.struct_data)

                    result_data = {
                        "document_id": result.document.id,
                        "parent_document_id": doc_data.get("parent_document_id"),
                        "filename": doc_data.get("filename", "Unknown"),
                        "content": doc_data.get("content", ""),
                        "relevance_score": getattr(result, "relevance_score", 0.0),
                        "snippets": [],  # Conversation API handles this differently
                        "is_chunk": doc_data.get("document_type") == "chunk",
                    }
                    all_results.append(result_data)

                    # Apply document filtering if specified
                    if document_ids:
                        parent_doc_id = doc_data.get("parent_document_id")
                        if parent_doc_id in document_ids:
                            filtered_results.append(result_data)
                        else:
                            logger.debug(
                                f"Filtered out document: {doc_data.get('filename')} (parent_id: {parent_doc_id})"
                            )
                    else:
                        filtered_results.append(result_data)

                search_results = filtered_results

                # Log document distribution
                if document_ids:
                    doc_counts = {}
                    for result in search_results:
                        filename = result["filename"]
                        doc_counts[filename] = doc_counts.get(filename, 0) + 1
                    logger.info(
                        f"Document distribution in {len(search_results)} results: {doc_counts}"
                    )

                    # Check which requested documents have no results
                    found_doc_ids = {
                        result["parent_document_id"] for result in search_results
                    }
                    missing_doc_ids = set(document_ids) - found_doc_ids
                    if missing_doc_ids:
                        logger.warning(
                            f"No results found for document IDs: {missing_doc_ids}"
                        )

            logger.info(f"Conversational search found {len(search_results)} results")

            # When filtering by specific documents, return ALL results to ensure complete coverage
            # Otherwise limit to max_results to avoid overwhelming responses
            if document_ids:
                logger.info(
                    f"Document filtering enabled - returning all {len(search_results)} results for comprehensive coverage"
                )
                return search_results
            else:
                logger.info(
                    f"No document filtering - limiting to {max_results} results"
                )
                return search_results[:max_results]

        except Exception as e:
            logger.error(f"Error in conversational search: {e}")
            # Fallback to regular search
            logger.info("Falling back to regular search...")
            return await self.search_documents(query, max_results, document_ids)

    async def search_documents(
        self,
        query: str,
        max_results: int = 10,
        document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        try:
            logger.info(f"Searching for: {query}")

            # Dynamic search limits based on document count
            search_max_results = self._calculate_optimal_search_results(
                document_ids, max_results
            )

            if document_ids:
                logger.info(f"Pre-filtering to {len(document_ids)} specific documents")
                logger.info(
                    f"Requesting {search_max_results} results from Vertex AI Search with pre-filter"
                )
            else:
                logger.info(
                    f"Searching all documents with {search_max_results} results"
                )

            # Use Enterprise app serving config instead of data store
            serving_config_path = f"projects/{self.project_id}/locations/{self.location}/collections/default_collection/engines/{self.app_id}/servingConfigs/{self.serving_config_id}"

            logger.info(f"Using serving config path: {serving_config_path}")

            request = await self._create_enhanced_search_request(
                serving_config_path=serving_config_path,
                query=query,
                max_results=search_max_results,
                document_ids=document_ids,
            )

            logger.info("Making search request...")

            # Perform search with retry logic for 503 errors
            max_retries = 3
            retry_delay = 1

            for attempt in range(max_retries):
                try:
                    response = self.search_client.search(request)
                    break  # Success, exit retry loop
                except ServiceUnavailable:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2**attempt) + random.uniform(0, 1)
                        logger.warning(
                            f"ServiceUnavailable error, retrying in {wait_time:.2f} seconds... (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"ServiceUnavailable error after {max_retries} attempts"
                        )
                        raise

            logger.info("Search response received, processing results...")

            # Process results with enhanced semantic understanding
            all_results = await self._process_search_results(response.results, query)

            # Pre-search filtering already applied - use all results
            results = all_results

            # Enhanced logging for document distribution and coverage
            if document_ids:
                doc_counts = {}
                for result in results:
                    doc_name = result.get("filename", "Unknown")
                    doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1

                logger.info(f"Pre-filtered results per document: {doc_counts}")
                logger.info(
                    f"Documents with results: {len(doc_counts)} out of {len(document_ids)} selected"
                )

                # Check which documents have no results
                documents_with_results = set(doc_counts.keys())
                selected_doc_names = set()
                for doc_id in document_ids:
                    if doc_id in self.documents:
                        selected_doc_names.add(self.documents[doc_id]["filename"])

                missing_docs = selected_doc_names - documents_with_results
                if missing_docs:
                    logger.warning(
                        f"No results found for {len(missing_docs)} documents: {list(missing_docs)[:5]}{'...' if len(missing_docs) > 5 else ''}"
                    )
                else:
                    logger.info("All selected documents have results in the search")
            else:
                # No document filtering - limit to max_results to avoid overwhelming responses
                results = results[:max_results]

            # Apply re-ranking if enabled - but keep more results for multi-document queries
            if self.use_ranking and self.ranking_service and results:
                try:
                    logger.info("Applying Vertex AI re-ranking to search results")
                    # For multi-document queries, keep more results after re-ranking
                    rerank_limit = (
                        max_results * 3
                        if document_ids and len(document_ids) > 5
                        else max_results * 2
                    )
                    rerank_limit = min(
                        rerank_limit, 50
                    )  # Cap at 50 to avoid overwhelming responses

                    # For large document sets, skip re-ranking to preserve coverage of all sources
                    if document_ids and len(document_ids) > 20:
                        logger.info(
                            f"Skipping re-ranking for large document set ({len(document_ids)} docs) to preserve all {len(results)} results across sources"
                        )
                    else:
                        logger.info(
                            f"Re-ranking with limit: {rerank_limit} (original max_results: {max_results})"
                        )

                        results = await self.ranking_service.rerank_results(
                            query=query, documents=results, top_n=rerank_limit
                        )
                        logger.info("Re-ranking complete")
                except Exception as e:
                    logger.warning(f"Re-ranking failed, using original results: {e}")

            logger.info(f"Found {len(results)} results for query: {query}")

            # Log what documents were retrieved (without content to keep logs clean)
            document_names = list({result["filename"] for result in results})
            logger.info(
                f"Retrieved chunks from {len(document_names)} documents out of {len(document_ids) if document_ids else 'all'} selected"
            )
            logger.info(f"Documents with results: {document_names}")

            # If we're missing many documents, try a broader search for comprehensive coverage
            if (
                document_ids and len(document_names) < len(document_ids) * 0.5
            ):  # Less than 50% coverage
                missing_docs = len(document_ids) - len(document_names)
                logger.info(
                    f"Low document coverage ({len(document_names)}/{len(document_ids)}). Attempting broader search for comprehensive comparison."
                )

                # Try a broader search with generic terms to get content from more documents
                broader_results = await self._perform_broader_search(
                    serving_config_path=serving_config_path,
                    document_ids=document_ids,
                    original_results=results,
                    max_additional_results=missing_docs
                    * 3,  # Try to get some content from missing docs
                )

                if broader_results:
                    results.extend(broader_results)
                    logger.info(
                        f"Broader search added {len(broader_results)} results from additional documents"
                    )

                    # Update document count after broader search
                    document_names = list({result["filename"] for result in results})
                    logger.info(
                        f"After broader search: {len(document_names)} documents represented"
                    )
            elif document_ids:
                logger.info(
                    f"Good document coverage: {len(document_names)}/{len(document_ids)} documents have relevant content"
                )

            if len(results) == 0:
                logger.warning(
                    f"No search results found in Vertex AI Search for query: {query}"
                )
                if document_ids:
                    logger.info(
                        "Attempting in-memory fallback using cached document chunks"
                    )
                    fallback_results = self._build_in_memory_results(
                        document_ids=document_ids, max_results=max_results
                    )
                    if fallback_results:
                        logger.info(
                            f"Returning {len(fallback_results)} fallback results from cached chunks"
                        )
                        return fallback_results
                return []

            return results

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            # No fallback - only use Vertex AI Search
            raise

    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all indexed documents from Vertex AI Search only"""
        try:
            branch_path = self.client.branch_path(
                project=self.project_id,
                location=self.location,
                data_store=self.data_store_id,
                branch="default_branch",
            )

            # List documents
            request = discoveryengine.ListDocumentsRequest(parent=branch_path)
            response = self.client.list_documents(request)

            # Group chunks by parent document to show one entry per original document
            document_groups = {}

            for doc in response:
                doc_data = {}
                if doc.struct_data:
                    doc_data = dict(doc.struct_data)

                document_type = doc_data.get("document_type", "chunk")

                if document_type == "chunk":
                    parent_id = doc_data.get("parent_document_id")
                    if parent_id:
                        if parent_id not in document_groups:
                            # Create entry for this document
                            document_groups[parent_id] = {
                                "id": parent_id,
                                "title": doc_data.get("filename", "Unknown"),
                                "filename": doc_data.get("filename", "Unknown"),
                                "file_type": doc_data.get("file_type", "unknown"),
                                "upload_time": None,  # Will get from first chunk
                                "chunk_count": 0,
                                "character_count": 0,
                            }

                        # Aggregate data from chunks
                        document_groups[parent_id]["chunk_count"] += 1
                        document_groups[parent_id]["character_count"] += doc_data.get(
                            "character_count", 0
                        )

                        # Get upload time from first chunk if not set
                        if not document_groups[parent_id]["upload_time"]:
                            # Extract upload time from memory storage if available
                            if parent_id in self.documents:
                                document_groups[parent_id]["upload_time"] = (
                                    self.documents[parent_id]["upload_time"]
                                )

            documents = list(document_groups.values())

            logger.info(f"Listed {len(documents)} documents from Vertex AI Search")
            return documents

        except Exception as e:
            logger.error(f"Error listing documents from Vertex AI Search: {e}")
            return []

    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document from the index"""
        try:
            logger.info(f"Attempting to delete document: {document_id}")

            # First check if document exists in memory
            if document_id not in self.documents:
                logger.warning(f"Document {document_id} not found in memory storage")
                # Try to find by searching through all documents
                for doc_id, doc_data in self.documents.items():
                    logger.debug(
                        f"Available document: {doc_id} - {doc_data.get('filename', 'unknown')}"
                    )

            # Delete all associated chunks (no main document to delete)
            deleted_chunks = 0
            chunk_index = 0

            while True:
                chunk_id = f"{document_id}_chunk_{chunk_index}"
                try:
                    chunk_path = self.client.document_path(
                        project=self.project_id,
                        location=self.location,
                        data_store=self.data_store_id,
                        branch="default_branch",
                        document=chunk_id,
                    )
                    self.client.delete_document(name=chunk_path)
                    deleted_chunks += 1
                    chunk_index += 1
                    logger.debug(f"Deleted chunk: {chunk_id}")
                except Exception as e:
                    # No more chunks to delete
                    logger.debug(
                        f"No more chunks to delete after {chunk_index} chunks: {str(e)}"
                    )
                    break

            if deleted_chunks > 0:
                logger.info(
                    f"Deleted {deleted_chunks} chunks for document: {document_id}"
                )
            else:
                logger.warning(f"No chunks found to delete for document: {document_id}")

            # Also remove from in-memory storage
            if document_id in self.documents:
                del self.documents[document_id]

            return deleted_chunks > 0

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False

    async def clear_all_documents(self) -> bool:
        """Delete all documents from the index"""
        try:
            logger.info("Starting to clear all documents...")

            # First, list all documents
            documents = await self.list_documents()

            if not documents:
                logger.info("No documents to clear")
                return True

            logger.info(f"Found {len(documents)} documents to delete")

            # Delete each document
            deleted_count = 0
            failed_deletions = []

            for doc in documents:
                try:
                    success = await self.delete_document(doc["id"])
                    if success:
                        deleted_count += 1
                    else:
                        failed_deletions.append(doc["id"])
                except Exception as e:
                    logger.error(f"Failed to delete document {doc['id']}: {e}")
                    failed_deletions.append(doc["id"])

            # Clear in-memory storage
            self.documents.clear()

            # Documents only stored in Vertex AI Search

            if failed_deletions:
                logger.warning(
                    f"Successfully deleted {deleted_count} documents, failed to delete {len(failed_deletions)}: {failed_deletions}"
                )
                return False
            else:
                logger.info(
                    f"Successfully cleared all {deleted_count} documents from Vertex AI Search"
                )
                return True

        except Exception as e:
            logger.error(f"Error clearing all documents: {e}")
            return False

    async def generate_response(
        self, query: str, search_results: List[Dict[str, Any]]
    ) -> str:
        """Generate response using Gemini with search results and optional grounding check"""
        try:
            # Group search results by source document and prepare context
            grouped_results = {}

            # Dynamic context window based on document diversity
            unique_docs = len({r["filename"] for r in search_results})
            max_results_to_use = self._calculate_optimal_context_size(
                unique_docs, len(search_results)
            )
            logger.info(
                f"Using {max_results_to_use} search results from {unique_docs} unique documents (total available: {len(search_results)})"
            )

            for result in search_results[:max_results_to_use]:
                filename = result["filename"]
                if filename not in grouped_results:
                    grouped_results[filename] = []
                grouped_results[filename].append(result)

            # Ensure document diversity in selected results
            selected_results = self._ensure_document_diversity(
                search_results[:max_results_to_use], unique_docs
            )

            # Group selected results by document
            grouped_results = {}
            for result in selected_results:
                filename = result["filename"]
                if filename not in grouped_results:
                    grouped_results[filename] = []
                grouped_results[filename].append(result)

            context = ""
            doc_counter = 1
            doc_names = list(grouped_results.keys())
            logger.info(
                f"Response generation using {len(grouped_results)} unique documents"
            )
            logger.info(f"Documents in context: {doc_names}")
            for filename, doc_results in grouped_results.items():
                context += f"\n--- Document {doc_counter}: {filename} ---\n"

                # Combine content from all chunks of this document - use full content, not snippets
                for chunk_result in doc_results:
                    # Always use full content instead of potentially corrupted snippets
                    content = chunk_result["content"]
                    context += content + "\n\n"

                doc_counter += 1
                context += "\n"

            if not context.strip():
                return "I couldn't find any relevant information in the uploaded documents to answer your question."

            # Debug: Log the context being sent to Gemini and check size limits
            context_length = len(context)
            logger.info(
                f"Context being sent to Gemini (length: {context_length} chars)"
            )

            # Check if context is too large for Gemini
            if (
                context_length > MAX_CONTEXT_CHARS
            ):  # Conservative limit for Gemini context window
                logger.warning(
                    f"Context too large ({context_length} chars), intelligent truncation needed..."
                )
                # Intelligent truncation: keep content from all documents
                context = self._intelligently_truncate_context(
                    grouped_results, MAX_CONTEXT_CHARS
                )
                logger.info(
                    f"Intelligently truncated context to {len(context)} chars while preserving all documents"
                )

            logger.info(f"Context preview: {context[:500]}...")
            if len(context) > CONTEXT_PREVIEW_CHARS:
                logger.info(f"Context end: ...{context[-300:]}")

            # Create natural prompt that lets Gemini understand user intent
            prompt = f"""Based on the following documents, answer the user's question comprehensively.

Documents:
{context}

Question: {query}

Please provide a complete answer based on the information in these documents. Include all relevant details, numbers, and context that would be helpful to fully address the question."""

            # Generate response using Gemini with retry logic and model fallback
            primary_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            fallback_models = os.getenv(
                "FALLBACK_MODELS", "gemini-1.5-pro,gemini-1.0-pro"
            ).split(",")
            all_models = [primary_model] + fallback_models

            max_retries = 3
            retry_delay = 1

            for model_name in all_models:
                logger.info(f"Trying model: {model_name}")
                model = GenerativeModel(model_name.strip())

                for attempt in range(max_retries):
                    try:
                        # Configure generation for comprehensive responses
                        generation_config = {
                            "max_output_tokens": 8192,
                            "temperature": 0.2,  # Lower temperature for factual responses
                            "top_p": 0.8,
                            # Removed top_k constraint to allow full context consideration
                        }

                        response = model.generate_content(
                            prompt, generation_config=generation_config
                        )
                        logger.info(
                            f"Successfully generated response with {model_name}"
                        )

                        response_text = (
                            response.text
                            if response.text
                            else "I apologize, but I couldn't generate a response at this time."
                        )

                        # Apply grounding check if enabled
                        if (
                            self.use_grounding
                            and self.grounding_service
                            and response.text
                        ):
                            try:
                                logger.info("Checking response grounding")
                                (
                                    is_grounded,
                                    grounding_result,
                                ) = await self.grounding_service.validate_response(
                                    query=query,
                                    response=response_text,
                                    source_documents=search_results,
                                    min_support_score=0.7,
                                )

                                if is_grounded:
                                    logger.info(
                                        f"Response is well grounded (score: {grounding_result.get('support_score', 0):.2%})"
                                    )
                                    return (
                                        self.grounding_service.format_grounded_response(
                                            grounding_result
                                        )
                                    )
                                else:
                                    logger.warning(
                                        f"Response has low grounding score: {grounding_result.get('support_score', 0):.2%}"
                                    )
                                    # Still return the response but with grounding information
                                    return f"{response_text}\n\n*Note: This response has a low grounding score ({grounding_result.get('support_score', 0):.2%}). Please verify information independently.*"
                            except Exception as e:
                                logger.warning(f"Grounding check failed: {e}")

                        return response_text
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "unavailable" in error_msg or "503" in error_msg:
                            if attempt < max_retries - 1:
                                wait_time = retry_delay * (2**attempt) + random.uniform(
                                    0, 1
                                )
                                logger.warning(
                                    f"{model_name} unavailable, retrying in {wait_time:.2f} seconds... (attempt {attempt + 1}/{max_retries})"
                                )
                                time.sleep(wait_time)
                            else:
                                logger.error(
                                    f"{model_name} unavailable after {max_retries} attempts"
                                )
                                break  # Try next model
                        elif "not found" in error_msg or "invalid" in error_msg:
                            logger.error(
                                f"Model {model_name} not found or invalid, trying next model..."
                            )
                            break  # Try next model immediately
                        else:
                            logger.error(f"Unexpected error with {model_name}: {e}")
                            break  # Try next model

            logger.error(f"All models failed: {all_models}")
            return "I apologize, but the AI service is temporarily unavailable. Please try again in a few moments."

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error while processing your question: {str(e)}"

    def _build_in_memory_results(
        self, document_ids: List[str], max_results: int
    ) -> List[Dict[str, Any]]:
        """Create synthetic search results from cached chunks when Vertex AI Search yields nothing."""

        doc_order = {doc_id: index for index, doc_id in enumerate(document_ids)}
        doc_chunk_counter: Dict[str, int] = {}
        fallback_results: List[Dict[str, Any]] = []

        for doc_id in document_ids:
            document = self.documents.get(doc_id)
            if not document:
                logger.info(
                    f"Fallback cache miss for document {doc_id}; fetching chunks from Vertex AI Search"
                )
                remote_chunks = self._fetch_chunks_from_vertex(
                    document_id=doc_id, max_results=max_results
                )
                fallback_results.extend(remote_chunks)
                continue

            filename = document.get("filename", "Unknown document")
            chunk_records = document.get("chunks", [])

            for chunk in chunk_records:
                if isinstance(chunk, dict):
                    chunk_text = chunk.get("text", "").strip()
                    chunk_metadata = chunk.get("metadata", {})
                else:
                    # Backward compatibility with legacy string-based chunks
                    chunk_text = str(chunk).strip()
                    chunk_metadata = {}

                if not chunk_text:
                    continue

                if not chunk_metadata:
                    logger.debug(
                        f"Fallback chunk from {doc_id} has no metadata; treating as legacy entry"
                    )

                raw_index = chunk_metadata.get("chunk_index")
                if isinstance(raw_index, int):
                    chunk_index = raw_index
                else:
                    chunk_index = doc_chunk_counter.get(doc_id, 0)

                doc_chunk_counter[doc_id] = chunk_index + 1

                fallback_results.append(
                    {
                        "document_id": f"{doc_id}_memory_{chunk_index if chunk_index is not None else len(fallback_results)}",
                        "parent_document_id": doc_id,
                        "is_chunk": True,
                        "content_type": chunk_metadata.get("content_type", "text"),
                        "title": chunk_metadata.get(
                            "headline",
                            f"{filename} - Chunk {chunk_index + 1 if isinstance(chunk_index, int) else 1}",
                        ),
                        "filename": filename,
                        "content": chunk_text,
                        "snippets": [chunk_text[:500]],
                        "relevance_score": 0.0,
                        "section_hint": chunk_metadata.get("section_hint"),
                        "page_start": chunk_metadata.get("page_start"),
                        "page_end": chunk_metadata.get("page_end"),
                        "keyword_terms": chunk_metadata.get("keyword_terms", []),
                        "entities": chunk_metadata.get("entities", {}),
                        "contains_table": chunk_metadata.get("contains_table", False),
                        "chunk_index": chunk_index,
                    }
                )

        fallback_results.sort(
            key=lambda item: (
                doc_order.get(item.get("parent_document_id"), len(doc_order)),
                item.get("chunk_index", 0)
                if isinstance(item.get("chunk_index"), int)
                else 0,
            )
        )

        truncated_results = fallback_results[:max_results]

        if truncated_results:
            logger.info(
                f"Constructed {len(truncated_results)} fallback results from in-memory chunks"
            )

        for result in truncated_results:
            result.pop("chunk_index", None)

        return truncated_results

    def _fetch_chunks_from_vertex(
        self, document_id: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """Fetch chunk documents directly from Vertex AI Search when they are not cached in memory."""

        fetched_results: List[Dict[str, Any]] = []
        chunk_index = 0

        while len(fetched_results) < max_results:
            chunk_id = f"{document_id}_chunk_{chunk_index}"
            chunk_path = self.client.document_path(
                project=self.project_id,
                location=self.location,
                data_store=self.data_store_id,
                branch="default_branch",
                document=chunk_id,
            )

            try:
                document = self.client.get_document(name=chunk_path)
            except Exception as e:
                logger.debug(
                    f"Stopping remote chunk fetch for {document_id} at index {chunk_index}: {e}"
                )
                break

            chunk_text = ""
            if document.content and document.content.raw_bytes:
                try:
                    chunk_text = document.content.raw_bytes.decode(
                        "utf-8", errors="ignore"
                    )
                except Exception:
                    chunk_text = document.content.raw_bytes.decode(
                        "latin-1", errors="ignore"
                    )

            struct_data: Dict[str, Any] = {}
            if document.struct_data:
                # struct_data is a MapComposite that already converts values on access
                struct_data = dict(document.struct_data)

            filename = struct_data.get("filename", "Unknown document")
            content_type = struct_data.get("content_type", "text")

            fetched_results.append(
                {
                    "document_id": chunk_id,
                    "parent_document_id": struct_data.get(
                        "parent_document_id", document_id
                    ),
                    "is_chunk": True,
                    "content_type": content_type,
                    "title": struct_data.get(
                        "title", f"{filename} - Chunk {chunk_index + 1}"
                    ),
                    "filename": filename,
                    "content": chunk_text,
                    "snippets": [chunk_text[:500]],
                    "relevance_score": 0.0,
                    "section_hint": struct_data.get("section_hint"),
                    "page_start": struct_data.get("page_start"),
                    "page_end": struct_data.get("page_end"),
                    "keyword_terms": struct_data.get("keyword_terms", []),
                    "entities": struct_data.get("entities", {}),
                    "contains_table": struct_data.get("contains_table", False),
                    "chunk_index": chunk_index,
                }
            )

            chunk_index += 1

        if fetched_results:
            logger.info(
                f"Fetched {len(fetched_results)} chunks directly from Vertex AI Search for document {document_id}"
            )

        return fetched_results

    def _convert_struct_to_dict(self, struct_obj: Struct) -> Dict[str, Any]:
        """Convert a protobuf Struct into native Python types recursively."""

        def _convert(value: Any) -> Any:
            if isinstance(value, Struct):
                return {k: _convert(v) for k, v in value.fields.items()}

            if isinstance(value, dict):
                if "stringValue" in value:
                    result = value["stringValue"]
                elif "numberValue" in value:
                    result = value["numberValue"]
                elif "boolValue" in value:
                    result = value["boolValue"]
                elif "listValue" in value:
                    result = [
                        _convert(item) for item in value["listValue"].get("values", [])
                    ]
                elif "structValue" in value:
                    result = {
                        k: _convert(v)
                        for k, v in value["structValue"].get("fields", {}).items()
                    }
                else:
                    result = {k: _convert(v) for k, v in value.items()}

                return result

            listvalue = getattr(value, "list_value", None)
            if listvalue is not None:
                return [_convert(item) for item in listvalue]

            return value

        if isinstance(struct_obj, Struct):
            return {key: _convert(val) for key, val in struct_obj.fields.items()}

        try:
            return MessageToDict(struct_obj)
        except Exception as e:
            logger.debug(f"Struct conversion fallback used: {e}")
            return {}

    # Enhanced indexing and search methods

    async def _index_main_document(
        self,
        document_id: str,
        filename: str,
        text_content: str,
        metadata: Dict[str, Any],
        branch_path: str,
    ) -> None:
        """Index the main document with enhanced metadata"""

        # Extract sections and entities for searchable fields
        sections = metadata.get("sections", [])
        entities = metadata.get("entities", {})

        # Create searchable content fields
        document_content = {
            "document_type": "main_document",
            "title": filename,
            "content": text_content,
            "filename": filename,
            "file_type": metadata.get("file_type", "unknown"),
            "upload_time": datetime.now().isoformat(),
            "chunk_count": metadata.get("chunk_count", 0),
            "character_count": len(text_content),
            "word_count": metadata.get("word_count", 0),
            "has_tables": metadata.get("has_tables", False),
            # Semantic fields for better search
            "sections": " | ".join(sections[:10]),  # Top 10 sections as searchable text
            "section_count": len(sections),
            # Entity fields for specific searches
            "percentages": " ".join(entities.get("percentages", [])),
            "currency": " ".join(entities.get("currency", [])),
            "dates": " ".join(entities.get("dates", [])),
            "abbreviations": " ".join(entities.get("abbreviations", [])),
            "phone_numbers": " ".join(entities.get("phone_numbers", [])),
            # Content categories for boosting
            "has_pricing": bool(
                entities.get("percentages") or entities.get("currency")
            ),
            "has_rates": bool(
                any(
                    term in text_content.lower()
                    for term in ["rate", "percentage", "%", "bps", "basis points"]
                )
            ),
            "has_offers": bool(
                any(
                    term in text_content.lower()
                    for term in ["offer", "pricing", "discount", "off"]
                )
            ),
        }

        # Create document with enhanced structure
        document = discoveryengine.Document(
            id=document_id,
            struct_data=document_content,
            content=discoveryengine.Document.Content(
                mime_type="text/plain", raw_bytes=text_content.encode("utf-8")
            ),
        )

        # Index main document
        self.client.create_document(
            parent=branch_path, document=document, document_id=document_id
        )

        logger.info(f"Main document indexed with enhanced metadata: {document_id}")

    async def _index_document_chunks(
        self,
        document_id: str,
        filename: str,
        chunks: List[Dict[str, Any]],
        doc_metadata: Dict[str, Any],
        branch_path: str,
    ) -> List[str]:
        """Index individual chunks as separate searchable documents"""

        chunk_ids: List[str] = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            chunk_text = chunk.get("text", "")
            chunk_metadata = chunk.get("metadata", {})

            struct_data = self._build_chunk_struct_data(
                document_id=document_id,
                filename=filename,
                chunk_index=i,
                chunk_text=chunk_text,
                chunk_metadata=chunk_metadata,
                doc_metadata=doc_metadata,
            )

            chunk_document = discoveryengine.Document(
                id=chunk_id,
                struct_data=struct_data,
                content=discoveryengine.Document.Content(
                    mime_type="text/plain", raw_bytes=chunk_text.encode("utf-8")
                ),
            )

            try:
                self.client.create_document(
                    parent=branch_path, document=chunk_document, document_id=chunk_id
                )
                chunk_ids.append(chunk_id)
                logger.debug(f"Indexed chunk {i + 1}/{len(chunks)}: {chunk_id}")
            except Exception as e:
                logger.warning(f"Failed to index chunk {chunk_id}: {e}")

        logger.info(f"Indexed {len(chunk_ids)} chunks for document {document_id}")
        return chunk_ids

    def _build_chunk_struct_data(
        self,
        *,
        document_id: str,
        filename: str,
        chunk_index: int,
        chunk_text: str,
        chunk_metadata: Dict[str, Any],
        doc_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare structured metadata for a chunk before indexing."""

        struct_data: Dict[str, Any] = {
            "document_type": "chunk",
            "parent_document_id": document_id,
            "chunk_index": chunk_index,
            "content_type": chunk_metadata.get("content_type", "text"),
            "title": chunk_metadata.get(
                "headline", f"{filename} - Chunk {chunk_index + 1}"
            ),
            "filename": filename,
            "file_type": doc_metadata.get("file_type", "unknown"),
            "content": chunk_text,
            "section_hint": chunk_metadata.get("section_hint"),
            "page_start": chunk_metadata.get("page_start"),
            "page_end": chunk_metadata.get("page_end"),
            "keyword_terms": chunk_metadata.get("keyword_terms", []),
            "entities": chunk_metadata.get("entities", {}),
            "word_count": chunk_metadata.get("word_count"),
            "char_count": chunk_metadata.get("char_count"),
            "character_count": chunk_metadata.get("char_count"),
            "token_estimate": chunk_metadata.get("token_estimate"),
            "overlap_with_previous": chunk_metadata.get("overlap_with_previous", False),
            "contains_table": chunk_metadata.get("contains_table", False),
            "source_start": chunk_metadata.get("source_start"),
            "source_end": chunk_metadata.get("source_end"),
            "split_reason": chunk_metadata.get("split_reason"),
            "sentence_count": chunk_metadata.get("sentence_count"),
        }

        # Remove fields that are None to keep the struct clean
        cleaned_struct = {
            key: value for key, value in struct_data.items() if value is not None
        }

        page_start = chunk_metadata.get("page_start")
        page_end = chunk_metadata.get("page_end")
        if page_start is not None:
            if page_end is not None and page_end != page_start:
                cleaned_struct["page_range_label"] = f"{page_start}-{page_end}"
            else:
                cleaned_struct["page_range_label"] = str(page_start)

        return cleaned_struct

    async def _create_enhanced_search_request(
        self,
        serving_config_path: str,
        query: str,
        max_results: int,
        document_ids: List[str] = None,
    ):
        """Create enhanced search request with semantic configuration and pre-filtering"""

        # Enhanced query processing
        enhanced_query = await self._enhance_query(query, document_ids)
        logger.info(f"Enhanced query: '{query}' -> '{enhanced_query}'")

        # Build filter expression for pre-search document filtering
        filter_expression = None
        if document_ids:
            doc_id_list = ", ".join([f'"{doc_id}"' for doc_id in document_ids])
            filter_expression = f"parent_document_id: ANY({doc_id_list})"
            logger.info(f"Applying pre-search filter to {len(document_ids)} documents")
            logger.info(
                f"Filter expression: {filter_expression[:200]}{'...' if len(filter_expression) > 200 else ''}"
            )
            logger.info(
                f"Sample document IDs being filtered: {document_ids[:3]}{'...' if len(document_ids) > 3 else ''}"
            )

        # Create search request with semantic boost configuration
        request = discoveryengine.SearchRequest(
            serving_config=serving_config_path,
            query=enhanced_query,
            page_size=max_results,
            filter=filter_expression,  # Apply document filter at search time!
            # Enhanced content search specification
            content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
                snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                    return_snippet=True, max_snippet_count=5, reference_only=False
                ),
                summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
                    summary_result_count=5,
                    include_citations=True,
                    ignore_adversarial_query=True,
                    ignore_non_summary_seeking_query=False,
                ),
                extractive_content_spec=discoveryengine.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                    max_extractive_answer_count=3, max_extractive_segment_count=5
                ),
            ),
            # Query expansion configuration
            query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
                condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
                pin_unexpanded_results=True,
            ),
            # Spell correction
            spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
                mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
            ),
        )

        return request

    async def _enhance_query(self, query: str, document_ids: List[str] = None) -> str:
        """Enhance query with natural semantic understanding - trust Vertex AI's capabilities"""
        query_lower = query.lower()

        # Only add context hints for large document sets
        if document_ids and len(document_ids) > 10:
            # For very large document sets, add a subtle hint for comprehensiveness
            if any(
                word in query_lower for word in ["compare", "across", "all", "each"]
            ):
                return f"{query} comprehensive comparison"

        # Trust Vertex AI's semantic understanding for most cases
        # It already understands:
        # - "rate" relates to "pricing", "percentage", "%", etc.
        # - "compare" means retrieve similar info from multiple sources
        return query

    async def _process_search_results(
        self, search_results, query: str
    ) -> List[Dict[str, Any]]:
        """Process search results with enhanced semantic understanding"""

        all_results = []
        result_count = 0
        query.lower()

        for result in search_results:
            result_count += 1
            logger.info(f"Processing result {result_count}: {result.document.id}")

            doc_data = {}
            if result.document.struct_data:
                # struct_data is a MapComposite that already converts values on access
                doc_data = dict(result.document.struct_data)
                logger.info(f"Document data keys: {list(doc_data.keys())}")
                # Debug: Log content field info
                content_val = doc_data.get("content", "")
                logger.info(f"Content type: {type(content_val)}, length: {len(str(content_val)) if content_val else 0}")
                if content_val:
                    logger.info(f"Content preview: {str(content_val)[:200]}...")

            # Extract snippets
            snippets = []
            if result.document.derived_struct_data:
                derived_data = dict(result.document.derived_struct_data)
                if "snippets" in derived_data:
                    snippets = [
                        snippet.get("snippet", "")
                        for snippet in derived_data["snippets"]
                    ]

            # Use Vertex AI Search's natural relevance score without artificial boosting
            relevance_score = getattr(result, "relevance_score", 0.0)

            # Determine if this is a chunk or main document
            document_type = doc_data.get("document_type", "unknown")
            is_chunk = document_type == "chunk"

            # For chunks, get the parent document ID for grouping
            parent_doc_id = (
                doc_data.get("parent_document_id") if is_chunk else result.document.id
            )

            result_data = {
                "document_id": result.document.id,
                "parent_document_id": parent_doc_id,
                "is_chunk": is_chunk,
                "content_type": (
                    doc_data.get("content_type", "text") if is_chunk else "document"
                ),
                "title": doc_data.get("title", "Unknown"),
                "filename": doc_data.get("filename", "Unknown"),
                "content": doc_data.get("content", ""),
                "snippets": snippets,
                "relevance_score": relevance_score,
                "section_hint": doc_data.get("section_hint"),
                "page_start": doc_data.get("page_start"),
                "page_end": doc_data.get("page_end"),
                "keyword_terms": doc_data.get("keyword_terms", []),
                "entities": doc_data.get("entities", {}),
                "contains_table": doc_data.get("contains_table", False),
            }

            all_results.append(result_data)

            # Log relevance score for debugging
            logger.debug(
                f"Document {result.document.id} relevance score: {relevance_score:.3f}"
            )

        # Sort by natural relevance score
        all_results.sort(key=lambda x: x["relevance_score"], reverse=True)

        return all_results

    def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation context for a session"""
        if session_id in self.conversations:
            del self.conversations[session_id]
            logger.info(f"Cleared conversation context for session: {session_id}")
            return True
        return False

    def get_conversation_id(self, session_id: str) -> Optional[str]:
        """Get conversation ID for a session"""
        return self.conversations.get(session_id)

    def _calculate_optimal_search_results(
        self, document_ids: Optional[List[str]], base_max_results: int
    ) -> int:
        """Calculate optimal search results based on document count"""
        if not document_ids:
            return base_max_results

        num_docs = len(document_ids)

        if num_docs <= 3:
            # Few documents: get comprehensive coverage
            return num_docs * 50
        elif num_docs <= 10:
            # Medium set: balanced approach
            return num_docs * 30
        elif num_docs <= 25:
            # Large set: ensure good coverage
            return num_docs * 15
        else:
            # Very large set: ensure minimum viable coverage
            return max(300, num_docs * 8)  # At least 300 total, or 8 per doc

    def _calculate_optimal_context_size(
        self, unique_docs: int, total_results: int
    ) -> int:
        """Calculate optimal context window size based on document diversity"""
        if unique_docs == 1:
            # Single document: use more chunks for depth
            return min(50, total_results)
        elif unique_docs <= 5:
            # Few documents: balance depth and breadth
            return min(60, total_results)
        elif unique_docs <= 15:
            # Many documents: ensure representation from each
            return min(80, total_results)
        else:
            # Very many documents: focus on coverage
            return min(100, total_results)

    def _ensure_document_diversity(
        self, search_results: List[Dict[str, Any]], unique_docs: int
    ) -> List[Dict[str, Any]]:
        """Ensure diverse representation from all documents in the result set"""
        if unique_docs <= 5:
            # Small set: return all results
            return search_results

        # For larger sets, ensure minimum representation per document
        min_per_doc = max(2, min(8, len(search_results) // unique_docs))

        doc_results = {}
        selected_results = []

        # First pass: ensure minimum per document
        for result in search_results:
            filename = result["filename"]
            if filename not in doc_results:
                doc_results[filename] = []

            if len(doc_results[filename]) < min_per_doc:
                doc_results[filename].append(result)
                selected_results.append(result)

        # Second pass: fill remaining slots with best results
        remaining_slots = len(search_results) - len(selected_results)
        for result in search_results:
            if result not in selected_results and remaining_slots > 0:
                selected_results.append(result)
                remaining_slots -= 1

        logger.info(
            f"Ensured diversity: {len(selected_results)} results from {len(doc_results)} documents"
        )
        return selected_results

    def _intelligently_truncate_context(
        self, grouped_results: Dict[str, List[Dict]], max_chars: int
    ) -> str:
        """Intelligently truncate context while preserving content from all documents"""
        total_docs = len(grouped_results)
        chars_per_doc = max_chars // total_docs

        context = ""
        doc_counter = 1

        for filename, doc_results in grouped_results.items():
            doc_context = f"\n--- Document {doc_counter}: {filename} ---\n"
            doc_content = ""

            # Combine content from this document
            for chunk_result in doc_results:
                content = chunk_result["content"]
                if len(doc_content) + len(content) <= chars_per_doc:
                    doc_content += content + "\n\n"
                else:
                    # Truncate this chunk to fit
                    remaining_chars = chars_per_doc - len(doc_content)
                    if remaining_chars > 100:  # Only add if meaningful amount
                        doc_content += content[:remaining_chars] + "...\n\n"
                    break

            context += doc_context + doc_content + "\n"
            doc_counter += 1

        return context

    async def _perform_broader_search(
        self,
        serving_config_path: str,
        document_ids: List[str],
        original_results: List[Dict[str, Any]],
        max_additional_results: int,
    ) -> List[Dict[str, Any]]:
        """Perform a broader search to get content from documents not represented in original results"""
        try:
            # Get documents that had no results in original search
            original_doc_names = {result["filename"] for result in original_results}

            # Find missing document IDs
            missing_doc_ids = []
            for doc_id in document_ids:
                if doc_id in self.documents:
                    doc_filename = self.documents[doc_id]["filename"]
                    if doc_filename not in original_doc_names:
                        missing_doc_ids.append(doc_id)

            if not missing_doc_ids:
                return []

            logger.info(
                f"Attempting broader search for {len(missing_doc_ids)} missing documents"
            )

            # Use neutral prompts to surface broadly representative content
            broader_queries = [
                "overview",
                "key points",
                "important sections",
                "supporting details",
            ]

            additional_results = []

            for query_term in broader_queries:
                if len(additional_results) >= max_additional_results:
                    break

                # Build filter for missing documents only
                missing_doc_list = ", ".join(
                    [f'"{doc_id}"' for doc_id in missing_doc_ids]
                )
                missing_filter = f"parent_document_id: ANY({missing_doc_list})"

                # Create broader search request
                request = discoveryengine.SearchRequest(
                    serving_config=serving_config_path,
                    query=query_term,
                    page_size=min(50, max_additional_results - len(additional_results)),
                    filter=missing_filter,
                    # Simplified search spec for broader results
                    content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
                        snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                            return_snippet=True,
                            max_snippet_count=3,
                            reference_only=False,
                        )
                    ),
                )

                # Perform search
                response = self.search_client.search(request)

                # Process results
                query_results = await self._process_search_results(
                    response.results, query_term
                )

                # Add results from documents we haven't seen yet
                for result in query_results:
                    if result["filename"] not in original_doc_names:
                        additional_results.append(result)
                        if len(additional_results) >= max_additional_results:
                            break

            logger.info(
                f"Broader search found {len(additional_results)} additional results from {len({r['filename'] for r in additional_results})} new documents"
            )
            return additional_results

        except Exception as e:
            logger.warning(f"Broader search failed: {e}")
            return []
