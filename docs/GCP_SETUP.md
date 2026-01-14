# Google Cloud Platform Setup Guide

This guide walks through setting up the required Google Cloud resources for the RAG Engine Demo. It covers three methods: automated script, REST API (recommended fallback), and manual Cloud Console setup.

## Prerequisites

- A Google Cloud project with billing enabled
- `gcloud` CLI installed and initialized
- Sufficient IAM permissions (see [Required IAM Roles](#required-iam-roles))

## Quick Reference

| Resource | ID | Purpose |
|----------|-----|---------|
| Data Store | `rag-demo-datastore` | Stores indexed document chunks |
| Search Engine | `rag-demo-app` | Provides search/retrieval capabilities |
| GCS Bucket | `{PROJECT_ID}-rag-temp` | Temporary staging for uploads |

---

## Method 1: Automated Script (Recommended)

The repository includes a setup script that automates resource creation. It automatically falls back to REST API when the gcloud CLI doesn't have Discovery Engine commands.

### Basic Usage

```bash
./scripts/setup-gcp.sh
```

### Custom Configuration

You can customize resource names via environment variables:

```bash
export GCP_PROJECT_ID="my-project"           # Google Cloud project
export DATASTORE_ID="my-custom-datastore"    # Data store name
export ENGINE_ID="my-custom-engine"          # Search engine name
export GCS_STAGING_BUCKET="my-staging-bucket"  # GCS bucket name
export ENGINE_LOCATION="global"              # Discovery Engine location

./scripts/setup-gcp.sh
```

### How the Script Works

1. **Detects CLI availability**: Checks for `gcloud discovery-engine` or `gcloud alpha discovery-engine`
2. **Falls back to REST API**: If CLI unavailable, uses `curl` with Discovery Engine REST API
3. **Idempotent**: Safe to run multiple times; skips existing resources

The script will:
- Enable required APIs
- Create/verify the data store
- Create/verify the search engine
- Create/verify the GCS staging bucket
- Print the `.env` configuration to copy

---

## Method 2: REST API Setup (Recommended Fallback)

When the gcloud CLI doesn't have Discovery Engine commands, use the REST API directly.

### Step 1: Authenticate and Set Variables

```bash
# Set your project ID
export PROJECT_ID="your-project-id"

# Authenticate and get access token
gcloud auth login
gcloud config set project $PROJECT_ID
export ACCESS_TOKEN=$(gcloud auth print-access-token)
```

### Step 2: Enable Required APIs

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  discoveryengine.googleapis.com \
  documentai.googleapis.com \
  storage.googleapis.com
```

**APIs Explained:**
| API | Purpose |
|-----|---------|
| `aiplatform.googleapis.com` | Vertex AI for Gemini models and embeddings |
| `discoveryengine.googleapis.com` | Discovery Engine (Vertex AI Search) |
| `documentai.googleapis.com` | Document AI for PDF parsing (optional) |
| `storage.googleapis.com` | Cloud Storage for file staging |

### Step 3: Create the Data Store

```bash
curl -X POST \
  "https://discoveryengine.googleapis.com/v1/projects/${PROJECT_ID}/locations/global/collections/default_collection/dataStores?dataStoreId=rag-demo-datastore" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "displayName": "RAG Demo Data Store",
    "industryVertical": "GENERIC",
    "solutionTypes": ["SOLUTION_TYPE_SEARCH"],
    "contentConfig": "CONTENT_REQUIRED"
  }'
```

**Configuration Explained:**
| Field | Value | Description |
|-------|-------|-------------|
| `dataStoreId` | `rag-demo-datastore` | Unique identifier for the data store |
| `displayName` | `RAG Demo Data Store` | Human-readable name shown in Console |
| `industryVertical` | `GENERIC` | Use GENERIC for general-purpose search (other options: RETAIL, MEDIA, HEALTHCARE_FHIR) |
| `solutionTypes` | `SOLUTION_TYPE_SEARCH` | Enables search functionality (other options: SOLUTION_TYPE_CHAT, SOLUTION_TYPE_RECOMMENDATION) |
| `contentConfig` | `CONTENT_REQUIRED` | Documents contain extractable text content (vs. metadata-only) |

**Expected Response:**
```json
{
  "name": "projects/{PROJECT_ID}/locations/global/collections/default_collection/dataStores/rag-demo-datastore/operations/{OPERATION_ID}",
  "metadata": {
    "@type": "type.googleapis.com/google.cloud.discoveryengine.v1.CreateDataStoreMetadata"
  }
}
```

Wait ~30 seconds for the operation to complete, then verify:

```bash
curl -X GET \
  "https://discoveryengine.googleapis.com/v1/projects/${PROJECT_ID}/locations/global/collections/default_collection/dataStores/rag-demo-datastore" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### Step 4: Create the Search Engine

```bash
curl -X POST \
  "https://discoveryengine.googleapis.com/v1/projects/${PROJECT_ID}/locations/global/collections/default_collection/engines?engineId=rag-demo-app" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "displayName": "RAG Demo Search App",
    "solutionType": "SOLUTION_TYPE_SEARCH",
    "dataStoreIds": ["rag-demo-datastore"],
    "searchEngineConfig": {
      "searchTier": "SEARCH_TIER_ENTERPRISE",
      "searchAddOns": ["SEARCH_ADD_ON_LLM"]
    }
  }'
```

**Configuration Explained:**
| Field | Value | Description |
|-------|-------|-------------|
| `engineId` | `rag-demo-app` | Unique identifier for the search engine |
| `displayName` | `RAG Demo Search App` | Human-readable name |
| `solutionType` | `SOLUTION_TYPE_SEARCH` | Search application type |
| `dataStoreIds` | `["rag-demo-datastore"]` | Links to the data store created above |
| `searchTier` | `SEARCH_TIER_ENTERPRISE` | Enables semantic/vector search (vs. SEARCH_TIER_STANDARD for keyword-only) |
| `searchAddOns` | `SEARCH_ADD_ON_LLM` | Enables LLM-powered features like summarization |

**Expected Response:**
```json
{
  "name": "projects/{PROJECT_ID}/locations/global/collections/default_collection/engines/rag-demo-app/operations/{OPERATION_ID}",
  "metadata": {
    "@type": "type.googleapis.com/google.cloud.discoveryengine.v1.CreateEngineMetadata"
  }
}
```

Wait ~1-2 minutes for engine creation, then verify:

```bash
curl -X GET \
  "https://discoveryengine.googleapis.com/v1/projects/${PROJECT_ID}/locations/global/collections/default_collection/engines/rag-demo-app" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### Step 5: Create the Staging Bucket

```bash
gsutil mb -p ${PROJECT_ID} -l us-central1 gs://${PROJECT_ID}-rag-temp
```

---

## Method 3: Manual Cloud Console Setup

If you prefer the UI or need to troubleshoot, follow these steps:

### Step 1: Enable APIs

1. Go to [APIs & Services](https://console.cloud.google.com/apis/library)
2. Search for and enable each API:
   - "Vertex AI API"
   - "Discovery Engine API"
   - "Document AI API"
   - "Cloud Storage API"

### Step 2: Create a Data Store

1. Go to [Agent Builder > Data Stores](https://console.cloud.google.com/gen-app-builder/data-stores)
2. Click **"Create data store"**
3. Select **"Cloud Storage"** as the source type
   - **Important:** You can select "No data" or create an empty bucket - the application imports documents via API, not bucket sync
4. Configure the data store:
   - **Data store name:** `rag-demo-datastore`
   - **Location:** `global`
   - **Document processing:** Leave default (or enable advanced parsing if desired)
5. Click **"Create"**

**About Sync Frequency (if prompted):**
- Select **"One-time"** or **"On-demand"** - the demo uses API-based imports, not scheduled sync
- Sync frequency only matters if you're importing directly from a GCS bucket on a schedule

### Step 3: Create a Search App

1. Go to [Agent Builder > Apps](https://console.cloud.google.com/gen-app-builder/apps)
2. Click **"Create app"**
3. Select **"Search"** as the app type
4. Configure:
   - **App name:** `rag-demo-app`
   - **Company name:** Your organization
   - **Location:** `global`
5. Select your data store: `rag-demo-datastore`
6. Click **"Create"**

### Step 4: Create GCS Bucket

1. Go to [Cloud Storage](https://console.cloud.google.com/storage/browser)
2. Click **"Create bucket"**
3. Configure:
   - **Name:** `{your-project-id}-rag-temp`
   - **Location type:** Region
   - **Location:** `us-central1`
   - **Storage class:** Standard
4. Click **"Create"**

---

## Required IAM Roles

The user or service account running the application needs these roles:

| Role | Purpose |
|------|---------|
| `roles/aiplatform.user` | Access Vertex AI models (Gemini) |
| `roles/discoveryengine.admin` | Manage Discovery Engine resources |
| `roles/storage.objectAdmin` | Read/write to GCS staging bucket |
| `roles/documentai.editor` | (Optional) Use Document AI processors |

Grant roles via:
```bash
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="user:your-email@example.com" \
  --role="roles/aiplatform.user"
```

---

## Verification

After setup, verify all resources exist:

```bash
# Check data store
curl -s "https://discoveryengine.googleapis.com/v1/projects/${PROJECT_ID}/locations/global/collections/default_collection/dataStores/rag-demo-datastore" \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" | jq .name

# Check search engine
curl -s "https://discoveryengine.googleapis.com/v1/projects/${PROJECT_ID}/locations/global/collections/default_collection/engines/rag-demo-app" \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" | jq .name

# Check bucket
gsutil ls gs://${PROJECT_ID}-rag-temp
```

---

## Environment Configuration

After creating resources, configure `backend/.env`:

```bash
# Required - Google Cloud
GCP_PROJECT_ID=your-project-id
GCP_LOCATION=us-central1              # Region for Vertex AI (Gemini models) - NOT "global"
GCS_STAGING_BUCKET=your-project-id-rag-temp

# Required - Vertex AI Search (Discovery Engine)
VERTEX_SEARCH_DATASTORE_ID=rag-demo-datastore
VERTEX_SEARCH_APP_ID=rag-demo-app
# Note: Discovery Engine location is always "global" and is hardcoded in the backend

# Required - Gemini model
GEMINI_MODEL=gemini-2.0-flash-001

# Optional feature flags
USE_DOCUMENT_AI=false                 # Enable Document AI for advanced PDF parsing
USE_VERTEX_RANKING=false              # Enable Vertex AI Search ranking
USE_VERTEX_GROUNDING=false            # Enable response grounding
```

**Important:** `GCP_LOCATION` must be a valid Vertex AI region (e.g., `us-central1`, `us-east1`, `europe-west1`) - **not** `global`. Discovery Engine uses `global` internally but that's handled by the backend code.

---

## Troubleshooting

### "Discovery Engine CLI not available"

The `gcloud discovery-engine` and `gcloud alpha discovery-engine` commands are not available in all gcloud versions. Use Method 2 (REST API) or Method 3 (Console) instead.

### "API not enabled" errors

Ensure all required APIs are enabled:
```bash
gcloud services list --enabled | grep -E "(aiplatform|discoveryengine|documentai|storage)"
```

### "Permission denied" errors

Check your IAM roles:
```bash
gcloud projects get-iam-policy ${PROJECT_ID} \
  --flatten="bindings[].members" \
  --filter="bindings.members:$(gcloud config get-value account)" \
  --format="table(bindings.role)"
```

### Data store creation fails

- Ensure the project has billing enabled
- Verify the Discovery Engine API is enabled
- Check quotas in [IAM & Admin > Quotas](https://console.cloud.google.com/iam-admin/quotas)

---

## Next Steps

1. Run `./scripts/setup-dev.sh` to install dependencies
2. Start the backend: `cd backend && source venv/bin/activate && uvicorn app.main:app --reload --port 8000`
3. Start the frontend: `cd frontend && npm run dev`
4. Open http://localhost:3000 and upload a document to test
