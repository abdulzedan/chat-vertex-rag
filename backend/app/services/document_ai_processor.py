import logging
import os
from typing import Any, Dict, List

from google.api_core.client_options import ClientOptions
from google.cloud import documentai, storage

logger = logging.getLogger(__name__)


class DocumentAIProcessor:
    """Enhanced document processor using Google Document AI Form Parser for table extraction"""

    def __init__(self):
        self.project_id = os.getenv("GCP_PROJECT_ID", "main-env-demo")
        self.location = os.getenv("DOCAI_LOCATION", "us")
        self.processor_id = os.getenv("DOCAI_PROCESSOR_ID")
        self.processor_version = os.getenv(
            "DOCAI_PROCESSOR_VERSION", "pretrained-form-parser-v2.0-2023-08-22"
        )

        # Initialize Document AI client
        client_options = ClientOptions(
            api_endpoint=f"{self.location}-documentai.googleapis.com"
        )
        self.client = documentai.DocumentProcessorServiceClient(
            client_options=client_options
        )

        # Initialize storage client for batch processing
        self.storage_client = storage.Client()

        # Chunking configuration
        self.chunk_size = 500
        self.include_ancestor_headings = True

    def _ensure_processor(self) -> str:
        """Ensure layout parser processor exists"""
        try:
            if self.processor_id:
                processor_name = self.client.processor_path(
                    self.project_id, self.location, self.processor_id
                )
                return processor_name

            # Create processor if not exists
            parent = self.client.common_location_path(self.project_id, self.location)

            # List existing processors
            processors = self.client.list_processors(parent=parent)

            # First try to find a form parser processor (best for tables)
            for processor in processors:
                if processor.type_ == "FORM_PARSER_PROCESSOR":
                    logger.info(
                        f"Using existing Form Parser processor: {processor.name}"
                    )
                    return processor.name

            # Create new form parser processor for better table extraction
            logger.info("Creating new Document AI Form Parser processor")
            processor = documentai.Processor(
                display_name="Form Parser for RAG", type_="FORM_PARSER_PROCESSOR"
            )

            created_processor = self.client.create_processor(
                parent=parent, processor=processor
            )

            logger.info(f"Created processor: {created_processor.name}")
            return created_processor.name

        except Exception as e:
            logger.error(f"Error ensuring processor: {e}")
            raise

    async def process_document_online(
        self, file_path: str, file_type: str, filename: str
    ) -> Dict[str, Any]:
        """Process document using online Document AI with Layout Parser"""
        try:
            logger.info(f"Processing document with Document AI: {filename}")

            processor_name = self._ensure_processor()

            # Read document
            with open(file_path, "rb") as file:
                document_content = file.read()

            # Create process options - OCR processor doesn't need layout config
            process_options = documentai.ProcessOptions()

            # Create raw document
            raw_document = documentai.RawDocument(
                content=document_content, mime_type=file_type
            )

            # Process document
            request = documentai.ProcessRequest(
                name=processor_name,
                raw_document=raw_document,
                process_options=process_options,
                skip_human_review=True,
            )

            result = self.client.process_document(request=request)
            document = result.document

            # Debug logging
            logger.info(
                f"Document AI response - text length: {len(document.text) if document.text else 0}"
            )
            logger.info(f"Document AI response - pages: {len(document.pages)}")
            logger.info(
                f"Document AI response - has chunked_document: {hasattr(document, 'chunked_document') and document.chunked_document is not None}"
            )
            if hasattr(document, "chunked_document") and document.chunked_document:
                logger.info(
                    f"Document AI response - chunks: {len(document.chunked_document.chunks) if document.chunked_document.chunks else 0}"
                )

            # Extract structured information
            extracted_data = self._extract_structured_data(document, filename)

            logger.info(f"Document AI processing complete for {filename}")
            return extracted_data

        except Exception as e:
            logger.error(f"Error processing document with Document AI: {e}")
            raise

    def _extract_structured_data(
        self, document: documentai.Document, filename: str
    ) -> Dict[str, Any]:
        """Extract structured data from Document AI response"""
        try:
            # Extract text
            full_text = document.text

            # Extract chunks from chunked document
            chunks = []
            chunk_details = []

            if document.chunked_document and document.chunked_document.chunks:
                for _i, chunk in enumerate(document.chunked_document.chunks):
                    chunk_text = chunk.content
                    chunks.append(chunk_text)

                    # Extract chunk metadata
                    chunk_meta = {
                        "chunk_id": chunk.chunk_id,
                        "type": "layout_parser",
                        "page_span": self._extract_page_span(chunk),
                        "source_elements": self._extract_source_elements(chunk),
                    }
                    chunk_details.append(chunk_meta)
            else:
                # Fallback: create chunks from pages
                for i, page in enumerate(document.pages):
                    page_text = self._extract_page_text(page, full_text)
                    if page_text.strip():
                        chunks.append(page_text)
                        chunk_details.append(
                            {
                                "chunk_id": f"page_{i+1}",
                                "type": "page",
                                "page_number": page.page_number,
                            }
                        )

            # Extract entities and structure
            entities = self._extract_entities(document)
            tables = self._extract_tables(document, full_text)

            # Extract metadata
            metadata = {
                "page_count": len(document.pages),
                "word_count": len(full_text.split()) if full_text else 0,
                "char_count": len(full_text) if full_text else 0,
                "has_tables": len(tables) > 0,
                "entities": entities,
                "language": self._detect_language(full_text),
                "confidence_scores": self._extract_confidence_scores(document),
            }

            return {
                "filename": filename,
                "file_type": "application/pdf",  # Document AI primarily for PDFs
                "full_text": full_text,
                "chunks": chunks,
                "chunk_details": chunk_details,
                "tables": tables,
                "metadata": metadata,
                "chunk_count": len(chunks),
                "character_count": len(full_text) if full_text else 0,
            }

        except Exception as e:
            logger.error(f"Error extracting structured data: {e}")
            raise

    def _extract_page_span(self, chunk) -> Dict[str, Any]:
        """Extract page span information from chunk"""
        try:
            if hasattr(chunk, "page_span"):
                return {
                    "page_start": chunk.page_span.page_start,
                    "page_end": chunk.page_span.page_end,
                }
            return {}
        except Exception:
            return {}

    def _extract_source_elements(self, chunk) -> List[str]:
        """Extract source elements from chunk"""
        try:
            elements = []
            if hasattr(chunk, "source_block_ids"):
                elements.extend(chunk.source_block_ids)
            return elements
        except Exception:
            return []

    def _extract_page_text(self, page, full_text: str) -> str:
        """Extract text from a page"""
        try:
            if page.layout and page.layout.text_anchor:
                return self._get_text_from_anchor(page.layout.text_anchor, full_text)
            return ""
        except Exception:
            return ""

    def _get_text_from_anchor(self, text_anchor, full_text: str) -> str:
        """Get text from text anchor"""
        try:
            text_segments = []
            for segment in text_anchor.text_segments:
                start_index = int(segment.start_index) if segment.start_index else 0
                end_index = (
                    int(segment.end_index) if segment.end_index else len(full_text)
                )
                text_segments.append(full_text[start_index:end_index])
            return "".join(text_segments)
        except Exception:
            return ""

    def _extract_entities(self, document: documentai.Document) -> Dict[str, List[str]]:
        """Extract entities from document"""
        entities = {"dates": [], "numbers": [], "addresses": [], "organizations": []}

        try:
            for entity in document.entities:
                entity_type = entity.type_.lower() if entity.type_ else "unknown"
                entity_text = (
                    entity.text_anchor
                    and self._get_text_from_anchor(entity.text_anchor, document.text)
                    or entity.mention_text
                )

                if "date" in entity_type:
                    entities["dates"].append(entity_text)
                elif "number" in entity_type or "amount" in entity_type:
                    entities["numbers"].append(entity_text)
                elif "address" in entity_type:
                    entities["addresses"].append(entity_text)
                elif "organization" in entity_type:
                    entities["organizations"].append(entity_text)
        except Exception as e:
            logger.warning(f"Error extracting entities: {e}")

        return entities

    def _extract_tables(
        self, document: documentai.Document, full_text: str
    ) -> List[Dict[str, Any]]:
        """Extract tables from document"""
        tables = []

        try:
            logger.info(f"Extracting tables - Document has {len(document.pages)} pages")
            for page_idx, page in enumerate(document.pages):
                logger.info(f"Page {page_idx} has {len(page.tables)} tables")
                for table_idx, table in enumerate(page.tables):
                    logger.info(f"Processing table {table_idx} on page {page_idx}")
                    table_data = {
                        "id": f"table_{page.page_number}_{table_idx}",
                        "page": page.page_number,
                        "rows": [],
                        "headers": [],
                    }

                    # Extract header rows
                    logger.info(f"Table has {len(table.header_rows)} header rows")
                    for row in table.header_rows:
                        header_cells = []
                        for cell in row.cells:
                            cell_text = self._get_text_from_anchor(
                                cell.layout.text_anchor, full_text
                            )
                            header_cells.append(cell_text.strip())
                        table_data["headers"].append(header_cells)

                    # Extract body rows
                    logger.info(f"Table has {len(table.body_rows)} body rows")
                    for row in table.body_rows:
                        row_cells = []
                        for cell in row.cells:
                            cell_text = self._get_text_from_anchor(
                                cell.layout.text_anchor, full_text
                            )
                            row_cells.append(cell_text.strip())
                        table_data["rows"].append(row_cells)

                    logger.info(
                        f"Extracted table with {len(table_data['headers'])} headers and {len(table_data['rows'])} rows"
                    )
                    tables.append(table_data)

        except Exception as e:
            logger.warning(f"Error extracting tables: {e}")

        return tables

    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        # This is a simplified version - could use actual language detection
        if not text:
            return "unknown"

        # Count English-like patterns
        english_indicators = [
            "the",
            "and",
            "of",
            "to",
            "in",
            "a",
            "is",
            "it",
            "you",
            "that",
        ]
        words = text.lower().split()[:100]  # Check first 100 words

        english_count = sum(1 for word in words if word in english_indicators)
        if english_count > len(words) * 0.1:  # If 10% are common English words
            return "en"

        return "unknown"

    def _extract_confidence_scores(
        self, _document: documentai.Document
    ) -> Dict[str, float]:
        """Extract confidence scores from document"""
        scores = {
            "overall_confidence": 0.0,
            "text_confidence": 0.0,
            "table_confidence": 0.0,
        }

        try:
            # This would need to be implemented based on actual Document AI response structure
            # Document AI doesn't always provide explicit confidence scores
            pass
        except Exception as e:
            logger.warning(f"Error extracting confidence scores: {e}")

        return scores
