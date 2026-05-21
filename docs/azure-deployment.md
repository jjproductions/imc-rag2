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
ACR_NAME="imcregistry"
ENV_NAME="imc-rag-env"
AZURE_OPENAI_API_KEY="<Add Azure OpenAI API Key>"
AZURE_OPENAI_ENDPOINT="https://imc-rag-foundry-resource.openai.azure.com"


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

# Build and Push (Ensure AMD64 architecture for Azure Container Apps)
docker build --platform linux/amd64 -t $ACR_NAME.azurecr.io/imc-rag-api:latest -f rag_api/Dockerfile .
docker push $ACR_NAME.azurecr.io/imc-rag-api:latest
```

## 3. Provision Qdrant on Azure Container Apps 
Instead of relying on an expensive Managed DB like Azure AI Search, deploy standard Qdrant directly into Container Apps using an Azure File share for persistence.

```bash
# Create an Azure Storage Account and File Shares
az storage account create --name imcragstorage --resource-group $RESOURCE_GROUP --location $LOCATION --sku Standard_LRS
az storage share-rm create --resource-group $RESOURCE_GROUP --storage-account imcragstorage --name qdrant-data --quota 10
az storage share-rm create --resource-group $RESOURCE_GROUP --storage-account imcragstorage --name openwebui-data --quota 10

# Link the storage to the ACA Environment
# (Get storage key)
STORAGE_KEY=$(az storage account keys list -g $RESOURCE_GROUP -n imcragstorage --query '[0].value' -o tsv)
az containerapp env storage set --access-mode ReadWrite --azure-file-account-name imcragstorage --azure-file-account-key $STORAGE_KEY --azure-file-share-name qdrant-data --storage-name qdrant-storage --name $ENV_NAME --resource-group $RESOURCE_GROUP
az containerapp env storage set --access-mode ReadWrite --azure-file-account-name imcragstorage --azure-file-account-key $STORAGE_KEY --azure-file-share-name openwebui-data --storage-name openwebui-storage --name $ENV_NAME --resource-group $RESOURCE_GROUP

# Deploy Qdrant (Without volumes first)
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
  --allow-insecure \
  --env-vars QDRANT__STORAGE__STORAGE_PATH="/qdrant/storage" \
             QDRANT__STORAGE__OPTIMIZERS__MEMMAP_THRESHOLD="0" \
             QDRANT__LOG_LEVEL="INFO"

# Export the config, inject the Azure Files volume, and update!
az containerapp show --name qdrant-app --resource-group $RESOURCE_GROUP -o json > qdrant.json
python3 -c "import json; f=open('qdrant.json', 'r'); config = json.load(f); f.close(); config['properties']['template']['volumes'] = [{'name': 'storage', 'storageType': 'AzureFile', 'storageName': 'qdrant-storage'}]; config['properties']['template']['containers'][0]['volumeMounts'] = [{'volumeName': 'storage', 'mountPath': '/qdrant/storage'}]; f=open('qdrant.json', 'w'); json.dump(config, f); f.close()"
az containerapp update --name qdrant-app --resource-group $RESOURCE_GROUP --yaml qdrant.json
```

## 4. Deploy the RAG API
Finally, deploy your API container into the environment, supplying the required LLM environment variables for Azure OpenAI. This guarantees your data processing stays 100% inside your tenant.

```bash
# Deploy API
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv)

az containerapp create \
  --name rag-api-app \
  --resource-group $RESOURCE_GROUP \
  --environment $ENV_NAME \
  --image $ACR_NAME.azurecr.io/imc-rag-api:latest \
  --registry-server $ACR_NAME.azurecr.io \
  --registry-username $ACR_NAME \
  --registry-password $ACR_PASSWORD \
  --cpu 0.5 --memory 1.0Gi \
  --min-replicas 0 \
  --max-replicas 3 \
  --ingress external \
  --target-port 8000 \
  --env-vars QDRANT_URL="http://qdrant-app" \
             QDRANT_PORT="80" \
             QDRANT_COLLECTION="board-policies-hybrid" \
             API_KEY="sk-rag-demo" \
             LLM_PROVIDER="azure_openai" \
             AZURE_OPENAI_API_KEY="$AZURE_OPENAI_API_KEY" \
             AZURE_OPENAI_ENDPOINT="$AZURE_OPENAI_ENDPOINT" \
             AZURE_OPENAI_API_VERSION="2024-07-18" \
             AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o-mini" \
             EMBEDDING_MODEL="BAAI/bge-m3"
```

## 5. Deploy OpenWeb UI (Frontend)
Finally, deploy the OpenWeb UI container so you have a beautiful ChatGPT-like interface to interact with your RAG API. We will fetch the RAG API's domain name automatically and link the UI to it!

```bash
# Deploy OpenWeb UI (Without volumes first)
az containerapp create --name openwebui-app --resource-group $RESOURCE_GROUP --environment $ENV_NAME --image ghcr.io/open-webui/open-webui:main --cpu 1.0 --memory 2.0Gi --min-replicas 1 --max-replicas 1 --ingress external --target-port 8080 --env-vars OPENAI_API_BASE_URL="http://rag-api-app/v1" OPENAI_API_KEY="sk-rag-demo"

# Export the config, inject the Azure Files volume, and update!
az containerapp show --name openwebui-app --resource-group $RESOURCE_GROUP -o json > openwebui.json
python3 -c "import json; f=open('openwebui.json', 'r'); config = json.load(f); f.close(); config['properties']['template']['volumes'] = [{'name': 'storage', 'storageType': 'AzureFile', 'storageName': 'openwebui-storage'}]; config['properties']['template']['containers'][0]['volumeMounts'] = [{'volumeName': 'storage', 'mountPath': '/app/backend/data'}]; f=open('openwebui.json', 'w'); json.dump(config, f); f.close()"
az containerapp update --name openwebui-app --resource-group $RESOURCE_GROUP --yaml openwebui.json
```

Once this finishes, you can get the frontend URL by running:
```bash
az containerapp show -n openwebui-app -g $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv
```

Because scaling is handled automatically and the Qdrant database is tiny, this architecture should easily cost less than $10 a month—well within your nonprofit grant limits!
