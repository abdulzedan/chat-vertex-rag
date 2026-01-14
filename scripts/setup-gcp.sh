#!/bin/bash

# =============================================================================
# RAG Engine Demo - Google Cloud Setup Script
# =============================================================================
#
# This script provisions the required Google Cloud resources for the RAG demo:
#   - Enables required APIs
#   - Creates a Discovery Engine data store
#   - Creates a Discovery Engine search engine
#   - Creates a GCS staging bucket
#
# USAGE:
#   ./scripts/setup-gcp.sh
#
# CONFIGURATION (via environment variables):
#   GCP_PROJECT_ID          - Google Cloud project ID (default: current gcloud project)
#   DATASTORE_ID            - Discovery Engine data store ID (default: rag-demo-datastore)
#   ENGINE_ID               - Discovery Engine search engine ID (default: rag-demo-app)
#   GCS_STAGING_BUCKET      - GCS bucket name for staging (default: {PROJECT_ID}-rag-temp)
#   ENGINE_LOCATION         - Discovery Engine location (default: global)
#
# EXAMPLE:
#   export GCP_PROJECT_ID="my-project"
#   export DATASTORE_ID="my-custom-datastore"
#   export ENGINE_ID="my-custom-engine"
#   ./scripts/setup-gcp.sh
#
# METHODS:
#   1. Tries gcloud discovery-engine CLI (if available)
#   2. Falls back to REST API via curl (works with any gcloud version)
#
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
ENGINE_LOCATION="${ENGINE_LOCATION:-global}"
DATASTORE_ID="${DATASTORE_ID:-rag-demo-datastore}"
APP_ID="${ENGINE_ID:-rag-demo-app}"
STAGING_BUCKET="${GCS_STAGING_BUCKET:-${PROJECT_ID}-rag-temp}"

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
if ! command -v gcloud >/dev/null 2>&1; then
    echo "❌ gcloud CLI is not installed. Install the Google Cloud SDK and re-run this script." >&2
    exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "❌ curl is not installed. Please install curl and re-run this script." >&2
    exit 1
fi

if [[ -z "${PROJECT_ID}" ]]; then
    echo "❌ No Google Cloud project configured." >&2
    echo "   Run 'gcloud config set project <project-id>' or export GCP_PROJECT_ID." >&2
    exit 1
fi

echo "============================================================"
echo "  RAG Engine Demo - Google Cloud Setup"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Project ID      : ${PROJECT_ID}"
echo "  Data Store ID   : ${DATASTORE_ID}"
echo "  Engine ID       : ${APP_ID}"
echo "  Location        : ${ENGINE_LOCATION}"
echo "  Staging Bucket  : ${STAGING_BUCKET}"
echo ""

gcloud config set project "${PROJECT_ID}" >/dev/null

# -----------------------------------------------------------------------------
# Authentication
# -----------------------------------------------------------------------------
echo "🔐 Verifying authentication..."
CURRENT_USER=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
if [[ -z "${CURRENT_USER}" ]]; then
    echo "❌ No active gcloud account. Run 'gcloud auth login' and try again." >&2
    exit 1
fi
echo "   Active account: ${CURRENT_USER}"

# Check Application Default Credentials
if ! gcloud auth application-default print-access-token >/dev/null 2>&1; then
    echo "❌ Application Default Credentials not found." >&2
    echo "   Run 'gcloud auth application-default login' and try again." >&2
    exit 1
fi
echo "   ADC configured: ✅"

# -----------------------------------------------------------------------------
# Enable APIs
# -----------------------------------------------------------------------------
echo ""
echo "📡 Enabling required APIs..."
REQUIRED_APIS=(
    "aiplatform.googleapis.com"
    "discoveryengine.googleapis.com"
    "documentai.googleapis.com"
    "storage.googleapis.com"
)
for api in "${REQUIRED_APIS[@]}"; do
    if gcloud services list --enabled --filter="name:${api}" --format="value(name)" 2>/dev/null | grep -q "${api}"; then
        echo "   ✅ ${api}"
    else
        echo "   🔄 Enabling ${api}..."
        gcloud services enable "${api}"
        echo "   ✅ ${api}"
    fi
done

# -----------------------------------------------------------------------------
# Check IAM Roles
# -----------------------------------------------------------------------------
echo ""
echo "🔍 Checking IAM roles for ${CURRENT_USER}..."
USER_ROLES=$(gcloud projects get-iam-policy "${PROJECT_ID}" \
    --flatten="bindings[].members" \
    --filter="bindings.members~${CURRENT_USER}" \
    --format="value(bindings.role)" 2>/dev/null || true)
if [[ -n "${USER_ROLES}" ]]; then
    echo "   Assigned roles:"
    echo "${USER_ROLES}" | sed 's/^/     /'
else
    echo "   ⚠️  No project-level roles detected. You may need additional permissions."
fi

# -----------------------------------------------------------------------------
# Helper: REST API calls for Discovery Engine
# -----------------------------------------------------------------------------
get_access_token() {
    gcloud auth application-default print-access-token 2>/dev/null
}

api_get() {
    local url="$1"
    local token
    token=$(get_access_token)
    curl -s -X GET "${url}" \
        -H "Authorization: Bearer ${token}" \
        -H "x-goog-user-project: ${PROJECT_ID}"
}

api_post() {
    local url="$1"
    local data="$2"
    local token
    token=$(get_access_token)
    curl -s -X POST "${url}" \
        -H "Authorization: Bearer ${token}" \
        -H "x-goog-user-project: ${PROJECT_ID}" \
        -H "Content-Type: application/json" \
        -d "${data}"
}

wait_for_operation() {
    local operation_name="$1"
    local max_attempts=30
    local attempt=0

    echo "   Waiting for operation to complete..."
    while [[ $attempt -lt $max_attempts ]]; do
        local status
        status=$(api_get "https://discoveryengine.googleapis.com/v1/${operation_name}")
        if echo "${status}" | grep -q '"done": true'; then
            if echo "${status}" | grep -q '"error"'; then
                echo "   ❌ Operation failed:"
                echo "${status}" | grep -o '"message": "[^"]*"' | head -1
                return 1
            fi
            return 0
        fi
        sleep 2
        ((attempt++))
    done
    echo "   ⚠️  Operation still in progress after ${max_attempts} attempts"
    return 0
}

# -----------------------------------------------------------------------------
# Detect Discovery Engine CLI availability
# -----------------------------------------------------------------------------
echo ""
echo "🔍 Detecting Discovery Engine CLI..."
DISCOVERY_CMD=()
if gcloud discovery-engine data-stores list \
        --project="${PROJECT_ID}" \
        --location="${ENGINE_LOCATION}" \
        --collection="default_collection" \
        --format="value(name)" >/dev/null 2>&1; then
    DISCOVERY_CMD=(gcloud discovery-engine)
    echo "   Using: gcloud discovery-engine"
elif gcloud alpha discovery-engine data-stores list \
        --project="${PROJECT_ID}" \
        --location="${ENGINE_LOCATION}" \
        --collection="default_collection" \
        --format="value(name)" >/dev/null 2>&1; then
    DISCOVERY_CMD=(gcloud alpha discovery-engine)
    echo "   Using: gcloud alpha discovery-engine"
else
    echo "   CLI not available, using REST API fallback"
fi

# -----------------------------------------------------------------------------
# Create/Verify Data Store
# -----------------------------------------------------------------------------
echo ""
echo "🏗️  Setting up Discovery Engine Data Store..."

DATASTORE_API_URL="https://discoveryengine.googleapis.com/v1/projects/${PROJECT_ID}/locations/${ENGINE_LOCATION}/collections/default_collection/dataStores"

if [[ ${#DISCOVERY_CMD[@]} -gt 0 ]]; then
    # Use gcloud CLI
    if "${DISCOVERY_CMD[@]}" data-stores describe "${DATASTORE_ID}" \
            --project="${PROJECT_ID}" \
            --location="${ENGINE_LOCATION}" \
            --collection="default_collection" >/dev/null 2>&1; then
        echo "   ✅ Data store '${DATASTORE_ID}' already exists"
    else
        echo "   📦 Creating data store '${DATASTORE_ID}'..."
        "${DISCOVERY_CMD[@]}" data-stores create "${DATASTORE_ID}" \
            --project="${PROJECT_ID}" \
            --location="${ENGINE_LOCATION}" \
            --collection="default_collection" \
            --display-name="RAG Demo Data Store" \
            --industry-vertical=GENERIC \
            --solution-types=SOLUTION_TYPE_SEARCH
        echo "   ✅ Data store created"
    fi
else
    # Use REST API
    EXISTING=$(api_get "${DATASTORE_API_URL}/${DATASTORE_ID}")
    if echo "${EXISTING}" | grep -q '"name"'; then
        echo "   ✅ Data store '${DATASTORE_ID}' already exists"
    else
        echo "   📦 Creating data store '${DATASTORE_ID}' via REST API..."
        RESPONSE=$(api_post "${DATASTORE_API_URL}?dataStoreId=${DATASTORE_ID}" '{
            "displayName": "RAG Demo Data Store",
            "industryVertical": "GENERIC",
            "solutionTypes": ["SOLUTION_TYPE_SEARCH"],
            "contentConfig": "CONTENT_REQUIRED"
        }')

        if echo "${RESPONSE}" | grep -q '"error"'; then
            if echo "${RESPONSE}" | grep -q "ALREADY_EXISTS"; then
                echo "   ✅ Data store '${DATASTORE_ID}' already exists"
            else
                echo "   ❌ Failed to create data store:"
                echo "${RESPONSE}" | grep -o '"message": "[^"]*"' | head -1
                exit 1
            fi
        else
            # Wait for operation to complete
            OPERATION_NAME=$(echo "${RESPONSE}" | grep -o '"name": "[^"]*"' | head -1 | sed 's/"name": "//;s/"$//')
            if [[ -n "${OPERATION_NAME}" ]]; then
                wait_for_operation "${OPERATION_NAME}"
            fi
            echo "   ✅ Data store created"
        fi
    fi
fi

# -----------------------------------------------------------------------------
# Create/Verify Search Engine
# -----------------------------------------------------------------------------
echo ""
echo "🏗️  Setting up Discovery Engine Search Engine..."

ENGINE_API_URL="https://discoveryengine.googleapis.com/v1/projects/${PROJECT_ID}/locations/${ENGINE_LOCATION}/collections/default_collection/engines"

if [[ ${#DISCOVERY_CMD[@]} -gt 0 ]]; then
    # Use gcloud CLI
    if "${DISCOVERY_CMD[@]}" engines describe "${APP_ID}" \
            --project="${PROJECT_ID}" \
            --location="${ENGINE_LOCATION}" \
            --collection="default_collection" >/dev/null 2>&1; then
        echo "   ✅ Search engine '${APP_ID}' already exists"
    else
        echo "   📦 Creating search engine '${APP_ID}'..."
        "${DISCOVERY_CMD[@]}" engines create "${APP_ID}" \
            --project="${PROJECT_ID}" \
            --location="${ENGINE_LOCATION}" \
            --collection="default_collection" \
            --display-name="RAG Demo Search App" \
            --industry-vertical=GENERIC \
            --solution-types=SOLUTION_TYPE_SEARCH \
            --data-store-ids="${DATASTORE_ID}"
        echo "   ✅ Search engine created"
    fi
else
    # Use REST API
    EXISTING=$(api_get "${ENGINE_API_URL}/${APP_ID}")
    if echo "${EXISTING}" | grep -q '"name"' && ! echo "${EXISTING}" | grep -q '"error"'; then
        echo "   ✅ Search engine '${APP_ID}' already exists"
    else
        echo "   📦 Creating search engine '${APP_ID}' via REST API..."
        RESPONSE=$(api_post "${ENGINE_API_URL}?engineId=${APP_ID}" "{
            \"displayName\": \"RAG Demo Search App\",
            \"solutionType\": \"SOLUTION_TYPE_SEARCH\",
            \"dataStoreIds\": [\"${DATASTORE_ID}\"],
            \"searchEngineConfig\": {
                \"searchTier\": \"SEARCH_TIER_ENTERPRISE\",
                \"searchAddOns\": [\"SEARCH_ADD_ON_LLM\"]
            }
        }")

        if echo "${RESPONSE}" | grep -q '"error"'; then
            if echo "${RESPONSE}" | grep -q "ALREADY_EXISTS"; then
                echo "   ✅ Search engine '${APP_ID}' already exists"
            else
                echo "   ❌ Failed to create search engine:"
                echo "${RESPONSE}" | grep -o '"message": "[^"]*"' | head -1
                exit 1
            fi
        else
            # Wait for operation to complete
            OPERATION_NAME=$(echo "${RESPONSE}" | grep -o '"name": "[^"]*"' | head -1 | sed 's/"name": "//;s/"$//')
            if [[ -n "${OPERATION_NAME}" ]]; then
                wait_for_operation "${OPERATION_NAME}"
            fi
            echo "   ✅ Search engine created"
        fi
    fi
fi

# -----------------------------------------------------------------------------
# Create/Verify GCS Bucket
# -----------------------------------------------------------------------------
echo ""
echo "🪣 Setting up GCS staging bucket..."
if gsutil ls -b "gs://${STAGING_BUCKET}" >/dev/null 2>&1; then
    echo "   ✅ Bucket gs://${STAGING_BUCKET} already exists"
else
    echo "   📦 Creating bucket gs://${STAGING_BUCKET}..."
    gsutil mb -p "${PROJECT_ID}" -l "us-central1" "gs://${STAGING_BUCKET}"
    echo "   ✅ Bucket created"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  ✅ Google Cloud Setup Complete"
echo "============================================================"
echo ""
echo "Resources created/verified:"
echo "  Data Store    : projects/${PROJECT_ID}/locations/${ENGINE_LOCATION}/collections/default_collection/dataStores/${DATASTORE_ID}"
echo "  Search Engine : projects/${PROJECT_ID}/locations/${ENGINE_LOCATION}/collections/default_collection/engines/${APP_ID}"
echo "  GCS Bucket    : gs://${STAGING_BUCKET}"
echo ""
echo "📝 Copy the following into backend/.env:"
echo ""
echo "  GCP_PROJECT_ID=${PROJECT_ID}"
echo "  GCP_LOCATION=us-central1"
echo "  VERTEX_SEARCH_DATASTORE_ID=${DATASTORE_ID}"
echo "  VERTEX_SEARCH_APP_ID=${APP_ID}"
echo "  GCS_STAGING_BUCKET=${STAGING_BUCKET}"
echo "  GEMINI_MODEL=gemini-2.0-flash-001"
echo ""
echo "  # Note: GCP_LOCATION is for Vertex AI (Gemini), not Discovery Engine."
echo "  # Discovery Engine uses 'global' internally (handled by the backend)."
echo ""
echo "Next steps:"
echo "  1. Create/update backend/.env with the values above"
echo "  2. Run ./scripts/setup-dev.sh to install dependencies"
echo "  3. Start backend: cd backend && source venv/bin/activate && uvicorn app.main:app --reload --port 8000"
echo "  4. Start frontend: cd frontend && npm run dev"
echo "  5. Open http://localhost:3000 and upload a document"
echo ""
