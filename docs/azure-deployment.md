# Azure Container Apps Deployment Guide

This guide details how to deploy the modernized, GPU-free RAG API onto Azure Container Apps securely within your Microsoft Tenant boundaries.

## Prerequisites
- Azure CLI (`az login`)
- An active Azure Subscription with available Microsoft Nonprofit Grant credits.

## 1. Create Azure Resources
First, set up a Resource Group, a Container Registry (ACR), and an Azure Container Apps Environment.

```bash
RESOURCE_GROUP="imc-rag-rg"
LOCATION="eastus"
ACR_NAME="imcragregistry"
ENV_NAME="imc-rag-env"

# Create Resource Group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Container Registry
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled true

# Create Container Apps Environment
az containerapp env create --name $ENV_NAME --resource-group $RESOURCE_GROUP --location $LOCATION
```

## 2. Build and Push the RAG API Image
Locally construct the streamlined RAG API Docker container and push it to your private registry. It no longer contains Ollama or massive PyTorch overhead, making it incredibly lightweight.

```bash
# Login to ACR
az acr login --name $ACR_NAME

# Build and Push
docker build -t $ACR_NAME.azurecr.io/rag-api:latest .
docker push $ACR_NAME.azurecr.io/rag-api:latest
```

## 3. Provision Qdrant on Azure Container Apps 
Instead of relying on an expensive Managed DB like Azure AI Search, deploy standard Qdrant directly into Container Apps using an Azure File share for persistence.

```bash
# Create an Azure Storage Account and File Share for Qdrant Data
az storage account create --name imcragstorage --resource-group $RESOURCE_GROUP --location $LOCATION --sku Standard_LRS
az storage share-rm create --resource-group $RESOURCE_GROUP --storage-account imcragstorage --name qdrant-data --quota 10

# Link the storage to the ACA Environment
# (Get storage key)
STORAGE_KEY=$(az storage account keys list -g $RESOURCE_GROUP -n imcragstorage --query '[0].value' -o tsv)
az containerapp env storage set --access-mode ReadWrite --azure-file-account-name imcragstorage --azure-file-account-key $STORAGE_KEY --azure-file-share-name qdrant-data --storage-name qdrant-storage --name $ENV_NAME --resource-group $RESOURCE_GROUP

# Deploy Qdrant
az containerapp create \
  --name qdrant-app \
  --resource-group $RESOURCE_GROUP \
  --environment $ENV_NAME \
  --image qdrant/qdrant:latest \
  --cpu 0.5 --memory 1.0Gi \
  --target-port 6333 \
  --ingress internal \
  --min-replicas 1 \
  --max-replicas 1 \
  --volumes name=storage,storageType=AzureFile,storageName=qdrant-storage \
  --volume-mounts volumeName=storage,mountPath=/qdrant/storage
```

## 4. Deploy the RAG API
Finally, deploy your API container into the environment, supplying the required LLM environment variables for Azure OpenAI. This guarantees your data processing stays 100% inside your tenant.

```bash
# Deploy API
az containerapp create \
  --name rag-api-app \
  --resource-group $RESOURCE_GROUP \
  --environment $ENV_NAME \
  --image $ACR_NAME.azurecr.io/rag-api:latest \
  --cpu 0.5 --memory 1.0Gi \
  --min-replicas 0 \
  --max-replicas 3 \
  --ingress external \
  --target-port 8000 \
  --env-vars QDRANT_URL="http://qdrant-app:6333" \
             LLM_PROVIDER="azure_openai" \
             AZURE_OPENAI_API_KEY="<your_openai_key>" \
             AZURE_OPENAI_ENDPOINT="<your_openai_endpoint>" \
             EMBEDDING_MODEL="intfloat/e5-small-v2"
```

Because scaling is handled automatically and the Qdrant database is tiny, this architecture should easily cost less than $10 a month—well within your nonprofit grant limits!
