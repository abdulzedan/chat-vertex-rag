#!/bin/bash

# Setup Google Cloud resources required by the RAG Engine demo.
# Creates/validates Discovery Engine data store + search app and a staging bucket.

set -euo pipefail

if ! command -v gcloud >/dev/null 2>&1; then
    echo "❌ gcloud CLI is not installed. Install the Google Cloud SDK and re-run this script." >&2
    exit 1
fi

PROJECT_ID=${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}
ENGINE_LOCATION="global"
DATASTORE_ID="rag-demo-datastore"
APP_ID="rag-demo-app"
STAGING_BUCKET_SUFFIX="rag-temp"

if [[ -z "${PROJECT_ID}" ]]; then
    echo "❌ No Google Cloud project configured. Run 'gcloud config set project <project-id>' or export GCP_PROJECT_ID." >&2
    exit 1
fi

STAGING_BUCKET="${PROJECT_ID}-${STAGING_BUCKET_SUFFIX}"

echo "🔧 Configuring Google Cloud resources for project: ${PROJECT_ID}"

gcloud config set project "${PROJECT_ID}" >/dev/null

echo "🔐 Verifying authentication..."
CURRENT_USER=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
if [[ -z "${CURRENT_USER}" ]]; then
    echo "❌ No active gcloud account. Run 'gcloud auth login' and try again." >&2
    exit 1
fi

echo "   Active gcloud account: ${CURRENT_USER}"

# Ensure Application Default Credentials exist
if ! gcloud auth application-default print-access-token >/dev/null 2>&1; then
    echo "❌ Application Default Credentials not found. Run 'gcloud auth application-default login'." >&2
    exit 1
fi

echo "📡 Enabling required APIs (idempotent)..."
REQUIRED_APIS=(
    "aiplatform.googleapis.com"
    "discoveryengine.googleapis.com"
    "documentai.googleapis.com"
    "storage.googleapis.com"
)
for api in "${REQUIRED_APIS[@]}"; do
    if gcloud services list --enabled --filter="name=${api}" --format="value(name)" | grep -q "${api}"; then
        echo "   ✅ ${api} already enabled"
    else
        echo "   🔄 Enabling ${api}..."
        gcloud services enable "${api}"
    fi
done

echo "🔍 Checking IAM roles for ${CURRENT_USER}..."
USER_ROLES=$(gcloud projects get-iam-policy "${PROJECT_ID}" \
    --flatten="bindings[].members" \
    --filter="bindings.members~${CURRENT_USER}" \
    --format="value(bindings.role)")
if [[ -n "${USER_ROLES}" ]]; then
    echo "   Roles:\n${USER_ROLES}"
else
    echo "   ⚠️  No project-level roles detected for ${CURRENT_USER}." >&2
fi

declare -a DISCOVERY_CMD
if gcloud discovery-engine data-stores list \
        --project="${PROJECT_ID}" \
        --location="${ENGINE_LOCATION}" \
        --collection="default_collection" \
        --format="value(name)" >/dev/null 2>&1; then
    DISCOVERY_CMD=(gcloud discovery-engine)
elif gcloud alpha discovery-engine data-stores list \
        --project="${PROJECT_ID}" \
        --location="${ENGINE_LOCATION}" \
        --collection="default_collection" \
        --format="value(name)" >/dev/null 2>&1; then
    DISCOVERY_CMD=(gcloud alpha discovery-engine)
else
    DISCOVERY_CMD=()
fi

if [[ ${#DISCOVERY_CMD[@]} -eq 0 ]]; then
    echo "⚠️  Discovery Engine CLI not available in this gcloud install." >&2
    echo "   Please create the resources manually via the Cloud Console:" >&2
    echo "   • Data Store: name '${DATASTORE_ID}', type 'Unstructured documents', location 'Global'" >&2
    echo "   • Search App / Engine: name '${APP_ID}', industry 'Generic', solution 'Search'" >&2
else
    echo "🏗️  Using '${DISCOVERY_CMD[*]}' to verify Discovery Engine resources..."

    DATASTORE_RESOURCE="projects/${PROJECT_ID}/locations/${ENGINE_LOCATION}/collections/default_collection/dataStores/${DATASTORE_ID}"
    APP_RESOURCE="projects/${PROJECT_ID}/locations/${ENGINE_LOCATION}/collections/default_collection/engines/${APP_ID}"

    if "${DISCOVERY_CMD[@]}" data-stores describe "${DATASTORE_ID}" \
            --project="${PROJECT_ID}" \
            --location="${ENGINE_LOCATION}" \
            --collection="default_collection" >/dev/null 2>&1; then
        echo "   ✅ Data store '${DATASTORE_ID}' exists"
    else
        echo "   📦 Creating data store '${DATASTORE_ID}'"
        "${DISCOVERY_CMD[@]}" data-stores create "${DATASTORE_ID}" \
            --project="${PROJECT_ID}" \
            --location="${ENGINE_LOCATION}" \
            --collection="default_collection" \
            --display-name="RAG Demo Data Store" \
            --industry-vertical=GENERIC \
            --solution-types=SOLUTION_TYPE_SEARCH
        echo "   ✅ Data store created"
    fi

    if "${DISCOVERY_CMD[@]}" engines describe "${APP_ID}" \
            --project="${PROJECT_ID}" \
            --location="${ENGINE_LOCATION}" \
            --collection="default_collection" >/dev/null 2>&1; then
        echo "   ✅ Search engine '${APP_ID}' exists"
    else
        echo "   📦 Creating search engine '${APP_ID}' linked to '${DATASTORE_ID}'"
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

    echo "   Discovery Engine resources ready:"
    echo "     • ${DATASTORE_RESOURCE}"
    echo "     • ${APP_RESOURCE}"
fi

echo "🪣 Ensuring staging bucket gs://${STAGING_BUCKET} exists..."
if gsutil ls -b "gs://${STAGING_BUCKET}" >/dev/null 2>&1; then
    echo "   ✅ Bucket already present"
else
    echo "   📦 Creating bucket..."
    gsutil mb -p "${PROJECT_ID}" -l "us-central1" "gs://${STAGING_BUCKET}"
    echo "   ✅ Bucket created"
fi

echo ""
echo "✅ Google Cloud validation complete"
echo ""
echo "📋 Summary"
echo "  Project ID              : ${PROJECT_ID}"
echo "  Active Account          : ${CURRENT_USER}"
echo "  Discovery Engine Locale : ${ENGINE_LOCATION}"
echo "  Data Store ID           : ${DATASTORE_ID}"
echo "  Search Engine ID        : ${APP_ID}"
echo "  Staging Bucket          : gs://${STAGING_BUCKET}"

echo ""
echo "📝 Add the following to backend/.env (replace if you customise IDs):"
echo "  GCP_PROJECT_ID=${PROJECT_ID}"
echo "  GCP_LOCATION=${ENGINE_LOCATION}"
echo "  VERTEX_SEARCH_DATASTORE_ID=${DATASTORE_ID}"
echo "  VERTEX_SEARCH_APP_ID=${APP_ID}"
echo "  GCS_STAGING_BUCKET=${STAGING_BUCKET}"

echo ""
echo "Next steps"
echo "  1. Run ./scripts/setup-dev.sh to install backend/frontend dependencies."
echo "  2. Start the backend (uvicorn) and frontend (npm run dev)."
echo "  3. Upload documents at http://localhost:3000 to verify ingestion."

echo "🚀 Google Cloud environment is ready."
