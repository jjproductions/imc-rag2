#!/usr/bin/env bash
# ==============================================================================
# Azure Container Apps Deployment Script for imc-rag2
# ==============================================================================
# Designed to be run either locally or inside GitHub Actions CI/CD workflows.
# ==============================================================================

set -euo pipefail

# --- Configuration & Defaults ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load .env if it exists locally for convenient local execution
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  echo "Loading local environment variables from .env..."
  # Export variables from .env (ignoring comments)
  export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

# Set default deployment configurations (override via env variables or .env)
RESOURCE_GROUP="${RESOURCE_GROUP:-imc-rag-rg}"
LOCATION="${LOCATION:-eastus}"
ACR_NAME="${ACR_NAME:-imcregistry}"
ENV_NAME="${ENV_NAME:-imc-rag-env}"
CONTAINER_APP_NAME="${CONTAINER_APP_NAME:-rag-api-app}"
IMAGE_NAME="${IMAGE_NAME:-imc-rag-api}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

ACR_LOGIN_SERVER="${ACR_NAME}.azurecr.io"
FULL_IMAGE_NAME="${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}"

# --- Helper Functions ---
log() {
  echo -e "\033[1;34m[INFO]\033[0m $1"
}

warn() {
  echo -e "\033[1;33m[WARN]\033[0m $1"
}

error_exit() {
  echo -e "\033[1;31m[ERROR]\033[0m $1" >&2
  exit 1
}

check_cli_dependencies() {
  local needs_docker=${1:-false}
  
  log "Checking tool dependencies..."
  if ! command -v az &>/dev/null; then
    error_exit "Azure CLI ('az') is required but not installed."
  fi
  
  if [[ "$needs_docker" == "true" ]]; then
    if ! command -v docker &>/dev/null; then
      error_exit "Docker CLI ('docker') is required but not installed/running."
    fi
  fi
}

check_azure_auth() {
  log "Checking Azure CLI authentication status..."
  if ! az account show &>/dev/null; then
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
      error_exit "Azure CLI is not authenticated in this GitHub Actions runner. Please run azure/login first."
    else
      warn "Azure CLI is not logged in. Attempting interactive login..."
      az login
    fi
  fi
  
  # Log current subscription for verification
  local sub_name
  sub_name=$(az account show --query name -o tsv)
  log "Authenticated to Azure Subscription: $sub_name"
}

# --- Action Implementations ---

build_image() {
  log "Starting Docker build for platform linux/amd64..."
  log "Image target: $FULL_IMAGE_NAME"
  
  # Ensure we are building from the project root directory
  cd "$PROJECT_ROOT"
  
  # Check if docker daemon is running
  check_cli_dependencies true
  
  # Build the image using standard docker build
  # Note: The Dockerfile expects requirements.txt and app in specific structures
  docker build \
    --platform linux/amd64 \
    -t "$FULL_IMAGE_NAME" \
    -f rag_api/Dockerfile .
    
  log "Docker image built successfully: $FULL_IMAGE_NAME"
}

push_image() {
  check_cli_dependencies true
  check_azure_auth
  
  log "Logging into Azure Container Registry: $ACR_NAME..."
  az acr login --name "$ACR_NAME"
  
  log "Pushing image to ACR..."
  docker push "$FULL_IMAGE_NAME"
  log "Docker image successfully pushed to registry."
}

deploy_container_app() {
  check_cli_dependencies false
  check_azure_auth
  
  log "Verifying Resource Group '$RESOURCE_GROUP' exists..."
  if ! az group show --name "$RESOURCE_GROUP" &>/dev/null; then
    error_exit "Resource Group '$RESOURCE_GROUP' does not exist. Please create shared infrastructure first."
  fi

  log "Verifying ACA Environment '$ENV_NAME' exists..."
  if ! az containerapp env show --name "$ENV_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
    error_exit "Container App Environment '$ENV_NAME' does not exist in resource group '$RESOURCE_GROUP'."
  fi

  log "Checking if Container App '$CONTAINER_APP_NAME' exists..."
  if az containerapp show --name "$CONTAINER_APP_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
    # App exists, perform an in-place update of the image
    log "Container App '$CONTAINER_APP_NAME' exists. Triggering image-only update (rolling revision deployment)..."
    
    az containerapp update \
      --name "$CONTAINER_APP_NAME" \
      --resource-group "$RESOURCE_GROUP" \
      --image "$FULL_IMAGE_NAME"
      
    log "Rolling deployment triggered successfully!"
  else
    # App does not exist, trigger initial creation
    warn "Container App '$CONTAINER_APP_NAME' was not found."
    log "Creating new Container App from scratch..."
    
    # Retrieve ACR Credentials
    local acr_password
    acr_password=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)
    
    # Establish default/fallback variables for initial creation if they aren't configured
    local api_key="${API_KEY:-sk-rag-demo}"
    local qdrant_url="${QDRANT_URL:-http://qdrant-app}"
    local qdrant_port="${QDRANT_PORT:-80}"
    local qdrant_collection="${QDRANT_COLLECTION:-board-policies-hybrid}"
    local llm_provider="${LLM_PROVIDER:-azure_openai}"
    local aoai_key="${AZURE_OPENAI_API_KEY:-}"
    local aoai_endpoint="${AZURE_OPENAI_ENDPOINT:-}"
    local aoai_version="${AZURE_OPENAI_API_VERSION:-2024-07-18}"
    local aoai_deployment="${AZURE_OPENAI_DEPLOYMENT_NAME:-gpt-4o-mini}"
    local embedding_model="${EMBEDDING_MODEL:-BAAI/bge-m3}"
    
    az containerapp create \
      --name "$CONTAINER_APP_NAME" \
      --resource-group "$RESOURCE_GROUP" \
      --environment "$ENV_NAME" \
      --image "$FULL_IMAGE_NAME" \
      --registry-server "$ACR_LOGIN_SERVER" \
      --registry-username "$ACR_NAME" \
      --registry-password "$acr_password" \
      --cpu 0.5 --memory 1.0Gi \
      --min-replicas 0 \
      --max-replicas 3 \
      --ingress external \
      --target-port 8000 \
      --env-vars QDRANT_URL="$qdrant_url" \
                 QDRANT_PORT="$qdrant_port" \
                 QDRANT_COLLECTION="$qdrant_collection" \
                 API_KEY="$api_key" \
                 LLM_PROVIDER="$llm_provider" \
                 AZURE_OPENAI_API_KEY="$aoai_key" \
                 AZURE_OPENAI_ENDPOINT="$aoai_endpoint" \
                 AZURE_OPENAI_API_VERSION="$aoai_version" \
                 AZURE_OPENAI_DEPLOYMENT_NAME="$aoai_deployment" \
                 EMBEDDING_MODEL="$embedding_model"
                 
    log "Container App '$CONTAINER_APP_NAME' created and deployed successfully!"
  fi
}

# --- Command Router ---

usage() {
  cat <<EOF
Usage: $0 [command]

Commands:
  build             Build the Docker image locally for linux/amd64.
  push              Login to ACR and push the local Docker image.
  deploy            Deploy the image to Azure Container Apps (updates if exists, creates if new).
  all (default)     Run build, push, and deploy.
  help              Display this help message.

Environment Variables (overridable):
  RESOURCE_GROUP       Azure Resource Group name (default: imc-rag-rg)
  LOCATION             Azure region location (default: eastus)
  ACR_NAME             Azure Container Registry name (default: imcregistry)
  ENV_NAME             Azure Container App Environment name (default: imc-rag-env)
  CONTAINER_APP_NAME   Target Container App name (default: rag-api-app)
  IMAGE_TAG            The tag to use for the image (default: latest)
EOF
}

main() {
  local command="${1:-all}"
  
  case "$command" in
    build)
      build_image
      ;;
    push)
      push_image
      ;;
    deploy)
      deploy_container_app
      ;;
    all)
      build_image
      push_image
      deploy_container_app
      ;;
    help|-h|--help)
      usage
      ;;
    *)
      error_exit "Unknown command: '$command'. Run '$0 help' for usage."
      ;;
  esac
}

main "$@"
