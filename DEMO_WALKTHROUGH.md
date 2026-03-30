# RAG Engine Demo - Technical Walkthrough & Blueprint

> **Purpose:** Speaker notes and architectural blueprint for walking stakeholders through how we built RAG with Vertex AI Search, how it compares to Gemini Enterprise, and the trade-offs involved.
> **Duration:** ~30-40 minutes
> **Audience:** Technical developers, architects, product managers

---

## Table of Contents

1. [Two Ways to Build RAG on Google Cloud](#1-two-ways-to-build-rag-on-google-cloud)
2. [Why We Chose the API Approach](#2-why-we-chose-the-api-approach)
3. [Document Ingestion: Our Approach vs Gemini Enterprise](#3-document-ingestion-our-approach-vs-gemini-enterprise)
4. [How Vertex AI Search Actually Works](#4-how-vertex-ai-search-actually-works)
5. [Query & Retrieval Flow](#5-query--retrieval-flow)
6. [Response Generation](#6-response-generation)
7. [Things to Be Aware Of](#7-things-to-be-aware-of)
8. [V2 Data Store (Native Layout Parsing + Chunking)](#8-v2-data-store-native-layout-parsing--chunking)
9. [Out-of-the-Box Discovery Engine Features](#9-out-of-the-box-discovery-engine-features)
10. [Metadata Extracted Per Document](#10-metadata-extracted-per-document)
11. [Activity Log — What It Shows and Why](#11-activity-log--what-it-shows-and-why)
12. [Live Demo Script](#12-live-demo-script)
13. [Architecture Diagrams](#13-architecture-diagrams)

---

## 1. Two Ways to Build RAG on Google Cloud

There are two distinct approaches to building RAG with Vertex AI Search. Understanding the difference is critical before looking at any code.

### Gemini Enterprise (SaaS)

Gemini Enterprise is the **fully managed SaaS product**. You interact through the Google Cloud Console or a simple API. It abstracts away everything:

- **Upload a PDF** → Gemini Enterprise automatically parses it (Layout parser by default)
- **Chunking happens automatically** → layout-aware, content-aware, configurable at data store creation
- **Search just works** → embeddings, indexing, retrieval, ranking are all handled
- **Answers with citations** → built-in grounded generation, follow-ups, summaries

From the docs:
> "The default parser for Gemini Enterprise is the Layout parser. It detects and understands document hierarchy, which leads to better chunking and ultimately better answer generation and retrieval."

**You write zero application code.** You configure a data store, upload files, and query.

### Vertex AI Search API (What This Demo Uses)

Vertex AI Search API is the **underlying engine** that Gemini Enterprise is built on. Same Discovery Engine, same infrastructure, but you write the code. You get:

- Full control over document processing
- Custom parsing pipelines
- Your own chunking strategy with custom metadata
- Custom search logic, filters, fallbacks
- Your own UI and user experience
- Programmatic document management

**You write application code.** You build the pipeline, the UI, the logic.

### Side-by-Side Comparison

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        GEMINI ENTERPRISE (SaaS)                          │
│                                                                          │
│   Upload PDF ──▶ [Layout Parser] ──▶ [Auto-Chunk] ──▶ [Index] ──▶ Ask  │
│                                                                          │
│   Everything managed. Console UI. Zero code. Out-of-the-box.            │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                     VERTEX AI SEARCH API (This Demo)                     │
│                                                                          │
│   Upload ──▶ [Our Parser] ──▶ [Our Chunker] ──▶ [API Index] ──▶ [Our   │
│               Doc AI /        NLTK-based        Each chunk as   Search   │
│               Gemini Vision   + metadata        a "document"   + Gen]   │
│                                                                          │
│   Custom pipeline. Custom UI. Full code. Full control.                  │
└──────────────────────────────────────────────────────────────────────────┘
```

| Aspect | Gemini Enterprise | This Demo (API) |
|--------|------------------|-----------------|
| **Setup effort** | Minutes (Console UI) | Days (full application) |
| **Parsing** | Layout parser (managed) | Document AI + Gemini Vision + fallbacks |
| **Chunking** | Automatic, layout-aware (100-500 tokens) | Custom NLTK-based (300-1500 chars) |
| **Chunk metadata** | Page spans, headings | Page spans, headings + keywords, entities, section hints, table flags |
| **Indexing** | Upload file → auto-indexed | We index each chunk as a separate document via API |
| **Search** | Managed semantic + keyword | Same engine, but we control filters, fallbacks |
| **Answer generation** | Built-in with citations | We assemble context + call Gemini ourselves |
| **UI** | Google Cloud Console | Custom React frontend |
| **Adjacent chunks** | `numPreviousChunks` / `numNextChunks` | Not used (we handle context diversity ourselves) |
| **Cost** | Enterprise license + per-query | API calls only |
| **Customizability** | Configuration-based | Code-level control |

---

## 2. Why We Chose the API Approach

### Talk Track

> "We built this demo using the Vertex AI Search API rather than Gemini Enterprise for a specific reason: we wanted to show what's possible when you need full control over the RAG pipeline.
>
> Gemini Enterprise is the right choice when you want RAG out of the box - upload documents, get answers, done. But many production use cases require custom document processing, custom metadata enrichment, custom UI, and custom search logic. That's what this demo demonstrates.
>
> The underlying search infrastructure is the same - both use Discovery Engine. The difference is who controls the pipeline."

### When to Use Which

| Use Case | Recommendation |
|----------|---------------|
| Internal knowledge base for employees | **Gemini Enterprise** - fast setup, managed |
| Prototype / proof of concept | **Gemini Enterprise** - minutes to deploy |
| Customer-facing application with custom UI | **Vertex AI Search API** - full control |
| Custom document processing needs | **Vertex AI Search API** - bring your own parsing |
| Need custom metadata for filtering | **Vertex AI Search API** - enrich chunks |
| Compliance / audit requirements | **Vertex AI Search API** - control data flow |
| Multi-tenant application | **Vertex AI Search API** - programmatic management |
| Simple Q&A over company docs | **Gemini Enterprise** - don't over-engineer |

---

## 3. Document Ingestion: Our Approach vs Gemini Enterprise

### How Gemini Enterprise Handles It

When you upload a document to Gemini Enterprise, the following happens automatically:

```
PDF uploaded
    │
    ▼
┌────────────────────────────────────────────────────────┐
│  LAYOUT PARSER (default, managed)                      │
│  • Detects paragraphs, tables, images, lists           │
│  • Identifies titles, headings, structural elements    │
│  • OCR for scanned content (optional add-on)           │
│  • Image annotation (Preview)                          │
│  • Table annotation (Preview)                          │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│  LAYOUT-AWARE CHUNKING (automatic)                     │
│  • Chunk size: 100-500 tokens (default 500)            │
│  • All text in a chunk from same layout entity         │
│  • includeAncestorHeadings: appends headings to chunks │
│  • Preserves semantic coherence                        │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│  INDEXING + EMBEDDING (fully managed)                  │
│  • Keyword extraction                                  │
│  • Semantic embeddings (hidden vectors)                │
│  • Metadata processing                                 │
│  • Searchable within seconds to minutes                │
└────────────────────────────────────────────────────────┘
```

**Configuration at data store creation:**
```json
{
  "documentProcessingConfig": {
    "chunkingConfig": {
      "layoutBasedChunkingConfig": {
        "chunkSize": 500,
        "includeAncestorHeadings": true
      }
    },
    "defaultParsingConfig": {
      "layoutParsingConfig": {}
    }
  }
}
```

**Key constraint:** Chunking cannot be turned on or off after data store creation.

### How This Demo Handles It

We built a custom pipeline because we wanted richer metadata per chunk:

```
PDF uploaded to FastAPI
    │
    ▼
┌────────────────────────────────────────────────────────┐
│  CUSTOM PARSING CHAIN                                  │
│  1. Document AI Form Parser (structured table output)  │
│  2. Gemini 2.5 Flash Vision (semantic understanding)   │
│  3. PyPDF2 / python-docx (fallback)                    │
│                                                        │
│  WHY: Document AI gives us machine-readable table      │
│  structures (row/column), not prose descriptions.      │
│  Gemini gives us semantic understanding of complex     │
│  layouts. Fallbacks ensure reliability.                │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│  NORMALIZATION                                         │
│  • Unicode cleanup                                     │
│  • Section/heading detection                           │
│  • Entity extraction (dates, URLs, emails, currencies) │
│                                                        │
│  WHY: Clean text produces better chunks and better     │
│  search results.                                       │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│  CUSTOM CHUNKING (NLTK sentence tokenization)          │
│  • 5 sentences per chunk, 2 sentence overlap           │
│  • 300-1500 character windows                          │
│  • Table-aware: tables stay in their own chunks        │
│  • Metadata per chunk:                                 │
│    - section_hint (detected heading)                   │
│    - page_start / page_end                             │
│    - keywords (top 8 non-stopwords)                    │
│    - has_table (boolean)                               │
│    - entities (dates, URLs, etc.)                      │
│    - token_estimate                                    │
│                                                        │
│  WHY: Richer metadata = better filtering and context.  │
│  Gemini Enterprise gives you page spans and headings.  │
│  We additionally get keywords, entities, table flags.  │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│  INDEX VIA API (chunks as individual documents)        │
│  • Each chunk → discoveryengine.Document               │
│  • struct_data with all metadata fields                │
│  • parent_document_id links back to source             │
│  • Also stored in-memory for fallback                  │
│                                                        │
│  WHY: Indexing chunks as documents gives us fine-      │
│  grained filtering via parent_document_id.             │
│  Trade-off: more documents in the data store.          │
└────────────────────────────────────────────────────────┘
```

### The Trade-Off

| Our Approach | Benefit | Cost |
|-------------|---------|------|
| Custom parsing chain | Structured table extraction, reliability | More code, more GCP API calls |
| Custom chunking | Richer metadata per chunk | Must maintain NLTK pipeline |
| Chunks as documents | Fine-grained filtering | More documents in data store |
| In-memory cache | Fallback when search returns nothing | Memory usage, cache staleness |
| Custom UI | Full brand control | Must build and maintain React app |

**Gemini Enterprise handles all of this with a config object at data store creation.**

---

## 4. How Vertex AI Search Actually Works

### The Engine Behind Both Approaches

Whether you use Gemini Enterprise or the API, the same Discovery Engine handles search. Here is what it does internally:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    DISCOVERY ENGINE INTERNALS                          │
│                    (Same for Gemini Enterprise and API)               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  DATA PROCESSING                                                     │
│  ├── Keyword Extraction: identifies important terms for retrieval    │
│  ├── Semantic Embeddings: creates vector representations (HIDDEN)    │
│  ├── Metadata Processing: processes struct_data for filtering        │
│  └── Cross-Attention: models query-document relationships            │
│                                                                      │
│  RETRIEVAL                                                           │
│  ├── Keyword Matching: conventional term-based lookup                │
│  ├── Semantic Search: embedding-based conceptual similarity          │
│  ├── Filter Application: metadata filters (parent_document_id, etc) │
│  └── Knowledge Graph: disambiguation and query expansion             │
│                                                                      │
│  RANKING                                                             │
│  ├── Relevance Score: combination of keyword + semantic (0.0-1.0)   │
│  ├── Boosting/Burying: custom rules to promote/demote results       │
│  ├── Freshness: age of documents                                     │
│  └── Personalization: user event signals (if configured)            │
│                                                                      │
│  Key: You NEVER see the embeddings. Relevance scores are bucketed   │
│  into 11 levels (0, 0.1, 0.2, ... 1.0). Higher = more relevant.   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Our Data Store Configuration

**File:** `backend/app/services/vertex_search.py`

```python
data_store = discoveryengine.DataStore(
    display_name="RAG Demo Data Store",
    industry_vertical=discoveryengine.IndustryVertical.GENERIC,
    solution_types=[discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH],
    content_config=discoveryengine.DataStore.ContentConfig.CONTENT_REQUIRED,
)
```

| Setting | Value | Why |
|---------|-------|-----|
| `industry_vertical` | `GENERIC` | Not retail/media/healthcare specific |
| `solution_types` | `SOLUTION_TYPE_SEARCH` | Enables semantic search capability |
| `content_config` | `CONTENT_REQUIRED` | Documents contain extractable text |

### Search Engine Configuration

```json
{
  "searchTier": "SEARCH_TIER_ENTERPRISE",
  "searchAddOns": ["SEARCH_ADD_ON_LLM"]
}
```

**Critical:** `SEARCH_TIER_ENTERPRISE` is what enables semantic/vector search. Without it, you only get keyword matching.

### What We Did NOT Configure (But Gemini Enterprise Does)

We did **not** configure layout-based chunking at the data store level:

```json
// We did NOT do this:
{
  "documentProcessingConfig": {
    "chunkingConfig": {
      "layoutBasedChunkingConfig": {
        "chunkSize": 500,
        "includeAncestorHeadings": true
      }
    },
    "defaultParsingConfig": {
      "layoutParsingConfig": {}
    }
  }
}
```

**Why not?** Because we handle parsing and chunking ourselves before indexing. We index pre-chunked content as individual "documents" with `struct_data`. This means:

- Discovery Engine treats each chunk as a standalone document
- Discovery Engine still generates embeddings for each chunk (hidden)
- We lose Discovery Engine's layout-aware chunking but gain custom metadata
- We lose the `numPreviousChunks`/`numNextChunks` adjacency feature since our chunks are separate documents, not chunks within a document

### Gemini Enterprise's Chunk Response (What We Don't Get)

In Gemini Enterprise with chunking enabled, search returns chunks with adjacency:

```json
{
  "chunk": {
    "id": "c17",
    "content": "ESS10: Stakeholder Engagement...",
    "pageSpan": { "pageStart": 14, "pageEnd": 15 },
    "chunkMetadata": {
      "previousChunks": [{ "id": "c16", "content": "..." }],
      "nextChunks": [{ "id": "c18", "content": "..." }]
    }
  }
}
```

We handle this differently - we ensure document diversity and context assembly in our own code rather than relying on adjacent chunk retrieval.

---

## 5. Query & Retrieval Flow

### What Happens When You Search

**Files:** `backend/app/api/chat.py` + `backend/app/services/vertex_search.py`

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  1. QUERY UNDERSTANDING             │
│     - Conversation context          │
│     - Reference resolution          │
│       ("that document" → expanded)  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  2. RETRIEVAL (Discovery Engine)    │
│     - Document filter applied       │
│       parent_document_id: ANY(...)  │
│     - Keyword + Semantic matching   │
│     - Top K chunks retrieved        │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  3. RANKING                         │
│     - Relevance scoring (0.0-1.0)  │
│     - Optional: Vertex AI Ranking  │
│       (semantic-ranker-512)        │
│     - Document diversity ensured    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  4. FALLBACK (if search empty)      │
│     - In-memory chunk cache         │
│     - Direct fetch by document ID   │
│     - Always provide context        │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  5. RESPONSE GENERATION             │
│     - Context assembled from chunks │
│     - Gemini 2.5 Flash streaming    │
│     - Citations extracted           │
└─────────────────────────────────────┘
```

### Document Filtering

```python
# When users select specific documents:
filter_expression = f"parent_document_id: ANY({doc_id_list})"
```

This uses Discovery Engine's filter syntax. Works because we indexed `parent_document_id` as a field in each chunk's `struct_data`.

### Conversational Search

We use the **ConverseConversation** API for multi-turn context:

```python
request = discoveryengine.ConverseConversationRequest(
    name=conversation_path,
    query=discoveryengine.TextInput(input=query),
    serving_config=serving_config_path,
    summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
        include_citations=True,
    ),
)
```

### Fallback Strategy

When semantic search returns nothing (e.g., instruction-style queries like "Extract key fields"):

1. Check in-memory cache for the selected document's chunks
2. If cache miss, fetch chunks directly from Discovery Engine by document ID
3. Always provide context to Gemini for generation

---

## 6. Response Generation

**File:** `backend/app/services/vertex_search.py`

### Context Assembly

```python
# Group search results by source document
for filename, doc_results in grouped_results.items():
    context += f"--- Document: {filename} ---\n"
    for chunk in doc_results:
        context += chunk["content"] + "\n"
```

- Max context: 900,000 characters (~225K tokens)
- Dynamic scaling based on document diversity
- Intelligent truncation preserves document structure

### Generation

```python
model = GenerativeModel("gemini-2.5-flash")  # GA model

generation_config = {
    "max_output_tokens": 8192,   # Long-form responses
    "temperature": 0.2,          # Factual, deterministic
    "top_p": 0.8,                # Focused sampling
}

# Stream response via SSE to frontend
response_stream = model.generate_content(prompt, stream=True)
for chunk in response_stream:
    if chunk.text:
        yield chunk.text
```

**Why temperature 0.2?** RAG demands factual grounding over creativity.

### How Gemini Enterprise Does This Differently

Gemini Enterprise has built-in answer generation:
- You call the `answer` or `streamAnswer` method
- It retrieves, ranks, and generates in a single API call
- Citations are included automatically
- Follow-up queries are handled natively

We do this manually: retrieve chunks → assemble context → call Gemini → stream SSE → extract citations.

---

## 7. Things to Be Aware Of

### Gotchas and Lessons Learned

**1. Chunks as Documents vs Native Chunking**

We index chunks as separate documents in Discovery Engine. This means:
- Each chunk gets its own embeddings (good for fine-grained search)
- We lose native adjacent chunk retrieval (`numPreviousChunks`/`numNextChunks`)
- Data store document count is `N_documents × avg_chunks_per_document`
- Must use `parent_document_id` filter to scope searches

**If starting over:** Consider enabling `layoutBasedChunkingConfig` at data store creation and uploading full documents. You get automatic chunking, adjacency, and page spans for free.

**2. Enterprise Tier is Required for Semantic Search**

`SEARCH_TIER_STANDARD` = keyword matching only. You **must** use `SEARCH_TIER_ENTERPRISE` for semantic/vector search. This is a common misconfiguration.

**3. Chunking Cannot Be Changed After Data Store Creation**

From the docs:
> "Document chunking can't be turned on or off after data store creation."

Plan your chunking strategy before creating the data store. If you change your mind, you need a new data store.

**4. Discovery Engine Location is Always `global`**

Don't confuse this with `GCP_LOCATION` (which is for Vertex AI / Gemini). Discovery Engine uses `global` internally. Setting it to `us-central1` will fail.

**5. Instruction-Style Queries Don't Match Semantically**

Queries like "Extract key fields" or "Summarize this document" have no semantic overlap with document content. They hit our fallback path. Gemini Enterprise handles these better because it has built-in query understanding.

**6. Indexing Lag**

After indexing a document via API, there can be a delay before it appears in search results. This is why we maintain an in-memory cache as a fallback.

**7. Custom Embeddings Are Possible But Not Used Here**

From the docs, you can bring your own embeddings (768 dimensions max) and use custom ranking expressions like:
```
0.5 * relevance_score + 0.3 * dotProduct(my_embedding_field)
```

We don't use this - we rely on Discovery Engine's built-in embeddings.

### What This Demo Proves

- You **can** build a fully custom RAG application on Vertex AI Search
- The underlying search quality is the same as Gemini Enterprise
- Custom parsing gives you structured table extraction
- Custom metadata gives you richer filtering
- Custom UI gives you brand control
- But it requires significantly more engineering effort

---

## 8. V2 Data Store (Native Layout Parsing + Chunking)

The demo now supports a **second data store** that uses Discovery Engine's native document processing instead of our custom pipeline. This lets you compare both approaches side by side.

### V1 vs V2 Data Store

| Aspect | V1 (Custom Pipeline) | V2 (Native) |
|--------|---------------------|-------------|
| **Parsing** | Document AI + Gemini Vision + fallbacks | Discovery Engine Layout Parser |
| **Chunking** | Custom NLTK (5 sentences, 2 overlap) | Layout-based (500 tokens, ancestor headings) |
| **Indexing** | Each chunk as a separate document | Full document uploaded; DE chunks it |
| **Search mode** | Standard document search | Chunk search with adjacent chunks |
| **Adjacent chunks** | N/A (chunks are separate docs) | `numPreviousChunks: 2` / `numNextChunks: 2` |
| **Spell correction** | Both — `SpellCorrectionSpec.Mode.AUTO` | Same |
| **Query expansion** | Both — `QueryExpansionSpec.Condition.AUTO` | Same |
| **Relevance scoring** | `model_scores["relevance_score"]` (0.0–1.0) | Same |

### V2 Data Store Configuration

The V2 data store is created with native document processing enabled:

```python
DocumentProcessingConfig(
    chunking_config=ChunkingConfig(
        layout_based_chunking_config=LayoutBasedChunkingConfig(
            chunk_size=500,                # ~500 tokens per chunk
            include_ancestor_headings=True  # Prepend parent headings to chunks
        )
    ),
    default_parsing_config=ParsingConfig(
        layout_parsing_config=LayoutParsingConfig()  # Enable layout-aware parsing
    ),
)
```

**What this gives us vs V1:**
- **Layout-aware parsing**: Discovery Engine detects paragraphs, tables, headings, lists — same as Gemini Enterprise
- **Automatic chunking**: Chunks respect layout boundaries (a table stays in one chunk, headings attach to their content)
- **Adjacent chunk retrieval**: `numPreviousChunks: 2` / `numNextChunks: 2` lets us pull surrounding context for each search result — critical for understanding chunks in context
- **No pre-processing needed**: Upload the raw document, DE handles everything

### V2 Search Mode: Chunk Search

V2 uses `searchResultMode: CHUNKS` instead of searching across separate documents:

```python
content_search_spec = ContentSearchSpec(
    search_result_mode=SearchResultMode.CHUNKS,
    chunk_spec=ChunkSpec(
        num_previous_chunks=2,  # 2 chunks of leading context
        num_next_chunks=2,      # 2 chunks of trailing context
    ),
)
```

### Configuration

```bash
# In backend/.env
USE_V2_DATASTORE=true       # Route search to V2 (falls back to V1 if empty)
USE_STREAM_ANSWER=false     # Use DE's AnswerQuery API for one-call retrieval+generation
```

### Setup

Run `scripts/setup-gcp.sh` to create both data stores:
- `rag-demo-datastore` (V1) — unstructured, chunks indexed as documents
- `rag-demo-datastore-v2` (V2) — layout parsing + chunking configured at creation

### New API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/search/autocomplete?q=...` | Autocomplete suggestions from Discovery Engine |
| `POST /api/chat/stream-answer` | One-call retrieval+generation via DE's AnswerQuery API |

---

## 9. Out-of-the-Box Discovery Engine Features

These features are enabled by the API — no custom code needed. They apply to both V1 and V2 data stores. **Every feature below is actively used AND surfaced in the Activity Log** so stakeholders can see it working in real time.

### Spell Correction ✅ Configured → Read → Displayed

```python
spell_correction_spec=SpellCorrectionSpec(
    mode=SpellCorrectionSpec.Mode.AUTO
)
```

Discovery Engine auto-corrects typos. The corrected query is read from `response.corrected_query` and shown in the Activity Log as a dedicated `spelling` event.

**Example:** `"revnue key figures"` → `"revenue key figures"` — visible in real time.

### Query Expansion ✅ Configured → Read → Displayed

```python
query_expansion_spec=QueryExpansionSpec(
    condition=QueryExpansionSpec.Condition.AUTO,
    pin_unexpanded_results=True
)
```

Discovery Engine expands queries with synonyms and related terms. Read from `response.query_expansion_info.expanded_query` and `pinned_result_count`. Shown as an `expansion` event in the Activity Log.

**What `pin_unexpanded_results=True` does:** Original query matches are pinned at the top of results; expanded matches are appended after. This prevents query drift.

### Extractive Answers ✅ Configured → Read → Displayed

```python
extractive_content_spec=ExtractiveContentSpec(
    max_extractive_answer_count=3,
    max_extractive_segment_count=5,
)
```

Discovery Engine extracts direct answers from document content — no LLM needed. Read from `result.document.derived_struct_data["extractive_answers"]`. The best extractive answer is shown as an `extractive` event in the Activity Log.

**Why this matters:** The extractive answer comes directly from the document text. It's a factual quote, not a generated summary. This is Discovery Engine's built-in answer extraction — same as what Gemini Enterprise uses.

### Summary Generation ✅ Configured → Read → Displayed

```python
summary_spec=SummarySpec(
    summary_result_count=5,
    include_citations=True,
    ignore_adversarial_query=True,
)
```

Discovery Engine generates its own summary from search results — separate from our Gemini call. Read from `response.summary.summary_text`. Shown as a `de_summary` event in the Activity Log with a preview of the text.

**Note:** We still use Gemini for the final user-facing response because we want streaming and custom prompt engineering. But the DE summary is captured and displayed to show that Discovery Engine can do end-to-end retrieval+generation on its own.

### Relevance Scoring ✅ Configured → Read → Displayed

Discovery Engine returns relevance scores via `model_scores["relevance_score"]` on each `SearchResult`. Extracted by our `_extract_relevance_score()` helper. Shown in the `ranking` event as scored results.

**How it works internally:**
- Keyword matching (BM25-style term frequency)
- Semantic similarity (hidden embeddings, vector search)
- Cross-attention scoring (query-document relationship modeling)
- Scores: 0.0–1.0 in 11 buckets

### Snippets ✅ Configured → Read → Used in Context

```python
snippet_spec=SnippetSpec(
    return_snippet=True,
    max_snippet_count=5,
    reference_only=False,
)
```

Discovery Engine generates highlighted snippets from `derived_struct_data["snippets"]`. Used as fallback context when full chunk content is unavailable. The best snippet is also used for the extractive answer display.

### Autocomplete ✅ Wired to Endpoint

```
GET /api/search/autocomplete?q=rev
→ [{"suggestion": "revenue"}, {"suggestion": "revenue growth"}]
```

Uses the `CompletionServiceClient` with `query_model="document-completable"` and `include_tail_suggestions=True`.

### Boost/Bury Rules (V2 Only)

V2 search includes a `BoostSpec` that can promote or demote results by metadata:

```python
boost_spec=BoostSpec(
    condition_boost_specs=[
        ConditionBoostSpec(
            condition="contains_table = true",
            boost=0.1,
        ),
    ]
)
```

### Query Reformulation (Custom)

Not a Discovery Engine feature, but our code reformulates follow-up queries using conversation context:

```python
# If query contains "that", "those", "this", etc. and there's conversation history:
reformulated = f"{query} (context: {previous_question_snippet})"
```

This is shown in backend logs but not yet in the Activity Log (it happens before the search request).

---

## 10. Metadata Extracted Per Document

### Document-Level Metadata (from Enhanced Processor)

| Field | Source | Example |
|-------|--------|---------|
| `file_type` | Upload MIME type | `application/pdf` |
| `chunk_count` | NLTK chunker | `108` |
| `character_count` | Full text length | `20,388` |
| `word_count` | Token count | `3,245` |
| `has_tables` | Table detection | `true` |
| `sections` | Heading detection | `["Revenue", "EBITDA", "Guidance"]` |
| `entities.percentages` | Regex extraction | `["15%", "3.2%"]` |
| `entities.currency` | Regex extraction | `["$485M", "$941M"]` |
| `entities.dates` | Regex extraction | `["Q4 2025", "2026"]` |
| `entities.phone_numbers` | Regex extraction | `["416-496-5856"]` |
| `entities.abbreviations` | Uppercase detection | `["EBITDA", "MECU"]` |

### Chunk-Level Metadata (per indexed chunk)

| Field | Purpose | Example |
|-------|---------|---------|
| `parent_document_id` | Links chunk to source document | `681a1c56-...` |
| `filename` | Source file name | `Chemtrade-Q4-2025.pdf` |
| `chunk_index` | Position in document | `42` |
| `section_hint` | Detected heading above this chunk | `"Adjusted EBITDA Reconciliation"` |
| `page_start` / `page_end` | Page span | `14` / `15` |
| `keyword_terms` | Top 8 non-stopwords | `["revenue", "adjusted", "ebitda"]` |
| `contains_table` | Table content flag | `true` |
| `content_type` | Text vs table | `"table"` |
| `word_count` | Chunk size | `87` |
| `token_estimate` | LLM token estimate | `~120` |
| `sentence_count` | Sentences in chunk | `5` |
| `overlap_with_previous` | Overlap flag | `true` |

---

## 11. Activity Log — What It Shows and Why

The Activity Log is a real-time WebSocket panel that shows every step of the backend pipeline during uploads and queries. It's designed to give stakeholders transparency into what the system is doing.

### Upload Pipeline Events

| Stage | Event | Detail Shown |
|-------|-------|-------------|
| `validate` | File accepted | File name, MIME type, size (KB/MB) |
| `process` | Processing with enhanced parser → Created N chunks | Characters extracted, tables detected, sections found, word count |
| `index` | Indexing chunks → Indexed N chunks | V2 note if dual-indexing enabled |

### Query Pipeline Events

| Stage | Event | Detail Shown | DE Feature |
|-------|-------|-------------|------------|
| `search` | Querying Discovery Engine → Found N results from M documents | Total index matches, top relevance score, source document names | `SearchRequest` + `model_scores` |
| `rewriting` | Query rewritten / accepted as-is | Spell correction + query expansion details, or "no rewriting needed" with active config | `SpellCorrectionSpec` + `QueryExpansionSpec` |
| `extractive` | Extractive answer from Discovery Engine | Direct answer quote from the document | `ExtractiveContentSpec` (answers + segments) |
| `de_summary` | Discovery Engine summary | Auto-generated summary with character count | `SummarySpec` (include_citations) |
| `ranking` | Top 5 results by relevance | Ranked list: `#1 [0.30] filename — section hint` | `model_scores["relevance_score"]` |
| `context` | Assembled N chars of context | Per-document character breakdown | Custom context assembly |
| `generate` | Generating response with gemini-2.5-flash → Response generated | Character count of response | Vertex AI Gemini |

### How Ranking Is Displayed

The Activity Log shows the top 5 search results sorted by Discovery Engine's relevance score:

```
#1 [0.30] Chemtrade-Q4-2025.pdf — Adjusted EBITDA Reconciliation
#2 [0.20] Chemtrade-Q4-2025.pdf — Financial Highlights
#3 [0.20] Guidance-2026.pdf — Revenue Outlook
#4 [0.10] Guidance-2026.pdf — Capital Expenditures
#5 [0.10] Chemtrade-Q4-2025.pdf — Net Debt Summary
```

Each line shows: rank, relevance score (0.0–1.0), source filename, and section hint (the detected heading above that chunk).

### Architecture

```
Backend (FastAPI)                           Frontend (React)
┌─────────────────────┐                    ┌──────────────────────┐
│  ActivityBroadcaster │ ── WebSocket ──▶  │  ActivityLog.tsx      │
│  (websocket.py)      │    /ws/activity   │  - Auto-reconnect    │
│                      │                    │  - Stage replacement │
│  emit_start()        │                    │  - Multi-line detail │
│  emit_success()      │                    │  - Spinner → check   │
│  emit_warning()      │                    └──────────────────────┘
│  emit_error()        │
│  clear()             │
└─────────────────────┘
    ▲           ▲
    │           │
documents.py  chat.py
(upload)      (query)
```

---

## 12. Live Demo Script

### Setup

**Terminal 1 - Backend:**
```bash
cd ~/Desktop/Developer/rag-engine-demo/backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd ~/Desktop/Developer/rag-engine-demo/frontend
npm run dev
```

Open http://localhost:3000

### UI Layout

The demo uses a three-panel layout — all panels are always visible:

```
┌──────────────┬──────────────────────┬──────────────────┐
│              │                      │                  │
│  Documents   │    Document Q&A      │  Activity Log    │
│  (upload,    │    (chat, always     │  (real-time      │
│   select,    │     visible)         │   pipeline       │
│   manage)    │                      │   events)        │
│              │                      │                  │
└──────────────┴──────────────────────┴──────────────────┘
```

The **Activity Log** shows real-time backend events during uploads and queries:
- File validation, document processing, chunk creation
- Search requests, result counts, relevance scores
- Gemini generation, citation extraction
- Errors and fallback paths

### Demo Flow

1. **Upload a document** — watch the Activity Log in real time:
   - `validate` → File accepted: Chemtrade-Q4-2025.pdf · Type: application/pdf · Size: 245 KB
   - `process` → Created 108 semantic chunks · 20,388 characters · 1 table detected · 3,245 words
   - `index` → Indexed 108 chunks in Discovery Engine
2. **Select the document** — checkbox enables document-scoped filtering
3. **Ask a specific question** — "What are the key financial figures?"
   - `search` → Found 25 results from 1 document · 29 total matches in index · Top relevance: 0.30
   - `ranking` → Top 5 ranked results with scores and section hints
   - `context` → Assembled 12,450 chars of context · 25 chunks
   - `generate` → Generating response with gemini-2.5-flash → Response generated · 1,765 chars
4. **Ask with a typo** — "What is the revnue?" → Activity Log shows spell correction: `"revnue"` → `"revenue"`
5. **Show ranking breakdown** — point out the ranked results list with relevance scores and section hints
6. **Show citations** — point out source references with page numbers and sections in the response
7. **Walk through the Activity Log** — explain each stage: what the system did, what Discovery Engine features were used, what metadata was extracted

### Key Talking Points

| Question | Answer |
|----------|--------|
| "Where are the embeddings?" | Hidden by design. Discovery Engine handles them. Same engine as Gemini Enterprise. |
| "How is relevance measured?" | Discovery Engine combines keyword matching (BM25), semantic similarity (hidden vectors), and cross-attention scoring. The final score is 0.0–1.0 in 11 buckets. You see the combined score, never the individual components. |
| "What does the Activity Log show?" | Every pipeline step: file validation, document processing (characters, tables, sections, words), search (spell correction, total matches, relevance scores), ranked results with section hints, context assembly with per-document breakdown, and Gemini generation with character count. |
| "What metadata is extracted per chunk?" | Section hint, page span, top 8 keywords, entity types (dates, currencies, percentages, phone numbers), table flag, content type, word count, token estimate, sentence count, and overlap flag. |
| "What out-of-the-box features are we using?" | **7 features, all visible in the Activity Log:** Spell correction, query expansion (with result pinning), extractive answers (direct document quotes), DE summary generation (with citations), relevance scoring (model_scores), snippets, and autocomplete. Every feature is configured in the request, the response is read, and the result is displayed. |
| "Why not just use Gemini Enterprise?" | This demo shows what you build when you need custom UI, custom parsing, or custom metadata. Gemini Enterprise is right for most internal use cases. |
| "What about the V2 data store?" | V2 uses Discovery Engine's native layout parsing — same as Gemini Enterprise under the hood. Full documents are uploaded, DE handles chunking. Search uses chunk mode with adjacent chunks (2 before, 2 after) for richer context. Falls back to V1 if V2 returns no results. |
| "How does this scale?" | Discovery Engine is fully managed. Same infrastructure whether 10 docs or 10 million. |
| "What about latency?" | Search: 200-500ms. Full E2E with Gemini streaming: 1-3s. All visible in Activity Log timestamps. |

---

## 13. Architecture Diagrams

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                              │
│                         React + Vite (localhost:3000)                    │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────┐ │
│   │   Upload    │  │   Select    │  │      Chat (SSE Streaming)       │ │
│   │  Documents  │  │  Documents  │  │                                 │ │
│   └──────┬──────┘  └──────┬──────┘  └────────────────┬────────────────┘ │
└──────────┼────────────────┼──────────────────────────┼──────────────────┘
           │                │                          │
           ▼                ▼                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         FASTAPI BACKEND (localhost:8000)                  │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                     DOCUMENT PROCESSING                             │  │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐   │  │
│  │  │ Document AI  │──▶│Gemini 2.5    │──▶│ Format Parsers       │   │  │
│  │  │ Form Parser  │   │Flash Vision  │   │ (PyPDF2, docx, etc.) │   │  │
│  │  │ (tables)     │   │ (semantics)  │   │ (fallback)           │   │  │
│  │  └──────────────┘   └──────────────┘   └──────────────────────┘   │  │
│  │            vs. Gemini Enterprise: Layout Parser (managed)           │  │
│  │                              │                                      │  │
│  │                              ▼                                      │  │
│  │  ┌────────────────────────────────────────────────────────────┐   │  │
│  │  │              CUSTOM CHUNKING (NLTK)                         │   │  │
│  │  │   • 5 sentences / chunk, 2 sentence overlap                │   │  │
│  │  │   • Metadata: section, pages, keywords, entities, tables   │   │  │
│  │  │                                                             │   │  │
│  │  │   vs. Gemini Enterprise: layoutBasedChunkingConfig          │   │  │
│  │  │   (100-500 tokens, layout-aware, includeAncestorHeadings)  │   │  │
│  │  └────────────────────────────────────────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                      VERTEX SEARCH SERVICE                          │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐   │  │
│  │  │ Index Chunks   │  │ Search API     │  │ Conversation API   │   │  │
│  │  │ (as documents) │  │ (filter+rank)  │  │ (multi-turn)       │   │  │
│  │  └────────────────┘  └────────────────┘  └────────────────────┘   │  │
│  │  ┌────────────────┐  ┌────────────────────────────────────────┐   │  │
│  │  │ In-Memory      │  │ Fallback: direct chunk fetch when     │   │  │
│  │  │ Cache          │  │ semantic search returns empty          │   │  │
│  │  └────────────────┘  └────────────────────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                     RESPONSE GENERATION                             │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐   │  │
│  │  │ Context        │  │ Gemini 2.5     │  │ SSE Streaming      │   │  │
│  │  │ Assembly       │──▶│ Flash          │──▶│ Response           │   │  │
│  │  │ (our logic)    │  │ (temp: 0.2)    │  │ + Citations        │   │  │
│  │  └────────────────┘  └────────────────┘  └────────────────────┘   │  │
│  │  vs. Gemini Enterprise: answer/streamAnswer method (one call)     │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
           │                          │                          │
           ▼                          ▼                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         GOOGLE CLOUD PLATFORM                             │
│                                                                           │
│  ┌─────────────────────┐  ┌─────────────────────────────────────────┐   │
│  │   VERTEX AI         │  │   DISCOVERY ENGINE (global)              │   │
│  │   (us-central1)     │  │   Same engine as Gemini Enterprise       │   │
│  │                     │  │                                          │   │
│  │  ┌───────────────┐  │  │  ┌─────────────┐  ┌─────────────────┐   │   │
│  │  │ Gemini 2.5    │  │  │  │  Data Store │  │  Search Engine  │   │   │
│  │  │ Flash (GA)    │  │  │  │  (our chunks│  │  (Enterprise    │   │   │
│  │  └───────────────┘  │  │  │  as docs)   │  │   Tier)         │   │   │
│  │  ┌───────────────┐  │  │  └─────────────┘  └─────────────────┘   │   │
│  │  │ Document AI   │  │  │                                          │   │
│  │  │ Form Parser   │  │  │  ┌─────────────────────────────────┐   │   │
│  │  └───────────────┘  │  │  │  INTERNALS (Hidden)              │   │   │
│  │                     │  │  │  • Embedding generation          │   │   │
│  │  ┌───────────────┐  │  │  │  • Vector index                 │   │   │
│  │  │ Ranking API   │  │  │  │  • Semantic similarity          │   │   │
│  │  │ (optional)    │  │  │  │  • Cross-attention scoring      │   │   │
│  │  └───────────────┘  │  │  └─────────────────────────────────┘   │   │
│  └─────────────────────┘  └─────────────────────────────────────────┘   │
│                                                                           │
│  ┌─────────────────────┐                                                 │
│  │   CLOUD STORAGE     │  Staging bucket for large file uploads          │
│  └─────────────────────┘                                                 │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `backend/app/main.py` | FastAPI entry, Vertex AI init |
| `backend/app/services/vertex_search.py` | Core search service (~800 lines) |
| `backend/app/services/enhanced_document_processor.py` | Custom document processing chain |
| `backend/app/services/document_ai_processor.py` | Document AI Form Parser integration |
| `backend/app/services/gemini_client.py` | Gemini 2.5 Flash wrapper |
| `backend/app/services/vertex_ai_ranking.py` | Optional re-ranking service |
| `backend/app/services/vertex_ai_grounding.py` | Optional grounding/validation |
| `backend/app/api/chat.py` | Chat/query endpoints with SSE, stream-answer |
| `backend/app/api/documents.py` | Upload, delete, Google Drive import |
| `backend/app/api/search.py` | Autocomplete endpoint |
| `backend/app/api/websocket.py` | WebSocket chat + ActivityBroadcaster |
| `frontend/src/components/ChatInterface.tsx` | Chat UI (inline, always visible) |
| `frontend/src/components/ActivityLog.tsx` | Real-time pipeline activity panel |
| `frontend/src/components/FileUploader.tsx` | Upload component |

---

## Environment Variables

```bash
# Required
GCP_PROJECT_ID=your-project-id
GCP_LOCATION=us-central1                    # Vertex AI region (NOT global)
VERTEX_SEARCH_DATASTORE_ID=rag-demo-datastore
VERTEX_SEARCH_APP_ID=rag-demo-app
GCS_STAGING_BUCKET=your-project-id-rag-temp
GEMINI_MODEL=gemini-2.5-flash               # GA model

# V2 Data Store (native layout parsing + chunking)
VERTEX_SEARCH_DATASTORE_V2_ID=rag-demo-datastore-v2
VERTEX_SEARCH_APP_V2_ID=rag-demo-app-v2
USE_V2_DATASTORE=true                       # Route search to V2 (falls back to V1)
USE_STREAM_ANSWER=false                     # DE AnswerQuery API

# Feature flags
USE_DOCUMENT_AI=true                        # Document AI Form Parser
USE_VERTEX_RANKING=true                     # Re-ranking service
USE_VERTEX_GROUNDING=false                  # Response validation
```

---

*End of Walkthrough*
