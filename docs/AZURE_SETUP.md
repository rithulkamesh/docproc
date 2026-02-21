# Azure Setup Guide

DocProc can use two Azure services for document processing and RAG:

1. **Azure OpenAI** — chat, embeddings, optional vision chat (`gpt-4o`, `text-embedding-ada-002`)
2. **Azure AI Vision (Computer Vision)** — image extraction: **Describe API** (captions/tags) + **Read API** (OCR for text in images, e.g. equations and labels)

## Environment variables

Set in `.env` or your environment:

| Variable | Description |
|----------|-------------|
| `AZURE_OPENAI_API_KEY` | API key (same key often works for OpenAI + Vision in one resource) |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint, e.g. `https://<resource>.openai.azure.com/` |
| `AZURE_OPENAI_DEPLOYMENT` | Chat model deployment name (e.g. `gpt-4o`) |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding deployment (e.g. `text-embedding-ada-002`) |
| `AZURE_VISION_ENDPOINT` | Computer Vision endpoint, e.g. `https://<resource>.cognitiveservices.azure.com/` |

If Vision is in the same Cognitive Services resource as OpenAI, use the same key for both.

## Credentials via Azure CLI

Use the script to fetch endpoint and key from an existing Cognitive Services resource:

```bash
./scripts/azure_env.sh [resource-name] [resource-group]
# Append output to .env: ./scripts/azure_env.sh myresource myrg >> .env
```

Requires [Azure CLI](https://docs.microsoft.com/cli/azure/install-azure-cli) and `az login`.

## DeploymentNotFound (404)

If you see `DeploymentNotFound: The API deployment for this resource does not exist`:

- Azure OpenAI requires **deployments** for each model. Create them in the Azure OpenAI portal.
- Ensure `AZURE_OPENAI_DEPLOYMENT` matches the deployment name (e.g. `gpt-4o`, `gpt-4`, `gpt-4o-mini`).

## Using Azure AI Vision for images

If you have **Azure AI Vision** (Computer Vision) deployed:

1. Set `AZURE_VISION_ENDPOINT` to your Computer Vision endpoint (e.g. `https://<resource>.cognitiveservices.azure.com/`).
2. Use the same key as `AZURE_OPENAI_API_KEY` when both are in the same resource.
3. Image extraction will use:
   - **Describe API** — image captions and tags
   - **Read API (v3.2)** — OCR for text inside images (equations, labels)

Chat and embeddings continue to use Azure OpenAI. See [CONFIGURATION.md](CONFIGURATION.md) for `ingest.use_vision` and related options.

## Microsoft Foundry

For Foundry projects, the endpoint format may differ. Use the endpoint and key from your project's "Models + endpoints" page.
