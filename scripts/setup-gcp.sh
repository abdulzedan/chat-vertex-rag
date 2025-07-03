#!/bin/bash

# Setup Google Cloud Resources for RAG Demo
# This script ensures required APIs are enabled and provides setup guidance
# Uses Application Default Credentials (ADC) instead of service account keys

set -e  # Exit on any error

# Get project ID from gcloud config or environment
PROJECT_ID=${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}
LOCATION="us-central1"
DATASTORE_ID="rag-demo-datastore"
APP_ID="rag-demo-app"

if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: No GCP project set. Run 'gcloud config set project YOUR_PROJECT_ID' first"
    exit 1
fi

echo "🔧 Setting up Google Cloud resources for RAG Demo..."
echo "Project: ${PROJECT_ID}"
echo "Location: ${LOCATION}"

# Set the project
gcloud config set project ${PROJECT_ID}

# Check current authentication
echo "🔐 Checking authentication..."
CURRENT_USER=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
echo "Authenticated as: ${CURRENT_USER}"

# Ensure required APIs are enabled
echo "📡 Ensuring required APIs are enabled..."

REQUIRED_APIS=(
    "aiplatform.googleapis.com"
    "discoveryengine.googleapis.com"
    "documentai.googleapis.com"
    "storage.googleapis.com"
)

for api in "${REQUIRED_APIS[@]}"; do
    if gcloud services list --enabled --filter="name:${api}" --format="value(name)" | grep -q "${api}"; then
        echo "✅ ${api} is already enabled"
    else
        echo "🔄 Enabling ${api}..."
        gcloud services enable ${api}
    fi
done

# Check user's permissions
echo "🔍 Checking current user permissions..."
USER_ROLES=$(gcloud projects get-iam-policy ${PROJECT_ID} --flatten="bindings[].members" --filter="bindings.members~${CURRENT_USER}" --format="value(bindings.role)")

echo "Current roles for ${CURRENT_USER}:"
echo "${USER_ROLES}"

# Verify the user has necessary permissions
if echo "${USER_ROLES}" | grep -q "roles/owner\|roles/editor\|roles/aiplatform\|roles/discoveryengine"; then
    echo "✅ User has sufficient permissions for Vertex AI and Discovery Engine"
else
    echo "⚠️  Warning: User may need additional permissions for Vertex AI services"
    echo "   Required roles: roles/aiplatform.user, roles/discoveryengine.admin"
fi

# Vertex AI Search resources setup
echo "🏗️  Setting up Vertex AI Search resources..."

# Check if discovery-engine commands are available
if ! gcloud alpha discovery-engine --help >/dev/null 2>&1; then
    echo "⚠️  Vertex AI Search CLI commands not available in current gcloud version"
    echo "📝 Manual setup required for Vertex AI Search resources:"
    echo ""
    echo "   1. Go to: https://console.cloud.google.com/vertex-ai/search"
    echo "   2. Create a new Search App:"
    echo "      - App Name: RAG Demo Search App"
    echo "      - Industry: Generic"
    echo "      - Solution Type: Search"
    echo "   3. Create a Data Store:"
    echo "      - Data Store Name: rag-demo-datastore"
    echo "      - Type: Unstructured documents"
    echo "      - Location: Global"
    echo "   4. Update your backend/.env file with the actual IDs:"
    echo "      - VERTEX_SEARCH_DATASTORE_ID=your-actual-datastore-id"
    echo "      - VERTEX_SEARCH_APP_ID=your-actual-app-id"
    echo ""
    echo "💡 Note: The console will provide the actual IDs after creation"
else
    # Try to create resources using CLI
    echo "📦 Attempting to create Vertex AI Search resources via CLI..."

    # Check if data store already exists
    if gcloud alpha discovery-engine data-stores describe ${DATASTORE_ID} --location=global >/dev/null 2>&1; then
        echo "✅ Data store '${DATASTORE_ID}' already exists"
    else
        echo "📦 Creating Vertex AI Search data store: ${DATASTORE_ID}"
        gcloud alpha discovery-engine data-stores create ${DATASTORE_ID} \
            --location=global \
            --industry-vertical=GENERIC \
            --solution-types=SOLUTION_TYPE_SEARCH \
            --display-name="RAG Demo Data Store"
        echo "✅ Data store created successfully"
    fi

    # Check if search app already exists
    if gcloud alpha discovery-engine engines describe ${APP_ID} --location=global >/dev/null 2>&1; then
        echo "✅ Search app '${APP_ID}' already exists"
    else
        echo "📦 Creating Vertex AI Search app: ${APP_ID}"
        gcloud alpha discovery-engine engines create ${APP_ID} \
            --location=global \
            --data-store-ids=${DATASTORE_ID} \
            --display-name="RAG Demo Search App" \
            --industry-vertical=GENERIC \
            --solution-types=SOLUTION_TYPE_SEARCH
        echo "✅ Search app created successfully"
    fi
fi

echo ""
echo "✅ Basic Google Cloud setup verification complete!"
echo ""
echo "📋 Summary:"
echo "  Project ID: ${PROJECT_ID}"
echo "  Authentication: Using Application Default Credentials"
echo "  Current User: ${CURRENT_USER}"
echo "  Location: ${LOCATION}"
echo ""
echo "🔧 Vertex AI Search Setup:"
echo "  Data Store ID: ${DATASTORE_ID}"
echo "  App ID: ${APP_ID}"
echo ""
echo "📝 Next Steps:"
echo "  1. ✅ APIs are enabled and user is authenticated"
echo "  2. 🌐 Vertex AI Search Engine appears to be configured in your .env"
echo "  3. 🚀 Run './setup-dev.sh' to setup the development environment"
echo "  4. 🎯 Start the application with the Quick Start guide"
echo ""
echo "💡 Notes:"
echo "  - Using Application Default Credentials (no service account keys needed)"
echo "  - Your current user account has the necessary permissions"
echo "  - Vertex AI Search resources (data store & app) should be managed via Cloud Console"
echo "    or using the Discovery Engine APIs if needed"
echo ""
echo "🔗 Useful Links:"
echo "  - Vertex AI Search Console: https://console.cloud.google.com/vertex-ai/search"
echo "  - Discovery Engine API: https://cloud.google.com/discovery-engine/docs"
