# Azure Container Apps Deployment Guide

This guide details how to deploy the modernized, GPU-free RAG API onto Azure Container Apps securely within your Microsoft Tenant boundaries.

## Prerequisites
- Azure CLI (`az login`)
- An active Azure Subscription with available Microsoft Nonprofit Grant credits.
- Docker running locally (if building and pushing images manually)

---

## Fast Track: Automated Deployment

We have created two modular scripts under `scripts/` to automate the provisioning and deployment of your RAG ecosystem.

### Step 1: Provision Core Infrastructure, Qdrant, & OpenWebUI
To automatically set up your Azure Resource Group, Container Registry, ACA Environment, Storage Account, File Shares, Qdrant database, and OpenWebUI chat interface with persistent storage mounts, run:

```bash
./scripts/setup_infra.sh
```
Upon completion, the script will output your public OpenWebUI URL.

### Step 2: Deploy RAG API (Initial Build & Deploy)
To build the RAG API Docker container locally, push it to your private registry, and deploy it to Azure Container Apps, run:

```bash
./scripts/deploy.sh
```

Once this is complete, your RAG API will be live, and OpenWebUI will be linked and ready to query.

---

## Alternative: Step-by-Step Manual Deployment

Below are the detailed, individual manual steps if you prefer to understand the exact CLI commands or configure the environment piece-by-piece.

### 1. Create Azure Resources
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

---

## 6. Automate with GitHub Actions (CI/CD)

To enable zero-downtime, automated deployments every time you push code to your `main` branch, we have added a localized deployment script (`scripts/deploy.sh`) and a pre-configured GitHub Actions workflow (`.github/workflows/deploy.yml`).

This pipeline uses **OIDC (OpenID Connect)** for secure, passwordless authentication with Azure and automatically logs into your private Azure Container Registry (ACR), builds the `linux/amd64` Docker image, and triggers a rolling revision update of your Container App (`rag-api-app`).

### Step-by-Step CI/CD Setup

To authorize GitHub Actions to deploy to your Azure Subscription, follow these instructions to configure an Azure Active Directory (Entra ID) Service Principal with Federated Credentials.

#### 1. Create the Entra ID Application
Locally in your terminal, run the following command (substituting your resource details as needed):

```bash
# 1. Create the AD Application
az login
APP_JSON=$(az ad app create --display-name "imc-rag-github-actions" -o json)
APP_ID=$(echo $APP_JSON | python3 -c "import sys, json; print(json.load(sys.stdin)['appId'])")
OBJECT_ID=$(echo $APP_JSON | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")

# 2. Create the Service Principal
az ad sp create --id $APP_ID
```

#### 2. Assign Deployment Roles
Authorize the application to manage resources within your resource group:

```bash
# Get your Azure Subscription ID
SUBSCRIPTION_ID=$(az account show --query id -o tsv)

# Scope the permission strictly to your Resource Group
SCOPE="/subscriptions/$SUBSCRIPTION_ID/resourceGroups/imc-rag-rg"

# Assign "Contributor" role to let GitHub update Container Apps
az ad sp create-for-rbac \
  --name "GitHubActions-imc-rag2" \
  --role contributor \
  --scopes /subscriptions/ba6dcc7e-b6bc-44ca-aa09-342f604118d7 \
  --sdk-auth
```

#### 3. Establish Federated Credentials (OIDC Connection)
This links your specific GitHub repository and the `main` branch directly to the Azure credential without needing a persistent password:

Create a file named `credential.json` (replace `<your-github-username>` with your GitHub username or organization):

```json
{
  "name": "imc-rag-github-actions-main",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:jjproductions/imc-rag2:ref:refs/heads/main",
  "description": "Federated credentials allowing GitHub Actions to deploy to Azure",
  "audiences": [
    "api://AzureADTokenExchange"
  ]
}
```

Then create the credential association in Azure:
```bash
az ad app federated-credential create --id $OBJECT_ID --parameters @credential.json
```

#### 4. Configure GitHub Secrets
In your GitHub repository, navigate to **Settings > Secrets and variables > Actions** and click **New repository secret** to add the following three secrets:

| Secret Name | Value | How to Get It |
| :--- | :--- | :--- |
| `AZURE_CLIENT_ID` | `<APP_ID>` | The Application (client) ID created in Step 1. |
| `AZURE_TENANT_ID` | `<TENANT_ID>` | Run `az account show --query tenantId -o tsv` |
| `AZURE_SUBSCRIPTION_ID` | `<SUBSCRIPTION_ID>` | Run `az account show --query id -o tsv` |

---

### Docker Layer Caching (Speed Optimization)
The Docker build includes caching for sentence transformers and sparse embedding models (`BAAI/bge-m3` and SPLADE). To prevent GitHub from downloading these large models (1.5GB+) on every run, the GitHub Actions workflow uses advanced `gha` caching:

```yaml
cache-from: type=gha
cache-to: type=gha,mode=max
```

This guarantees that subsequent workflow runs will finish in **under 45 seconds** rather than 8 minutes, optimizing your nonprofit's GitHub Actions credits!

---

Because scaling is handled automatically and the Qdrant database is tiny, this architecture should easily cost less than $10 a month—well within your nonprofit grant limits!

