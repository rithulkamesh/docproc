# Azure Setup Guide

docproc **CLI** can use Azure for document extraction:

1. **Azure OpenAI** — chat (for LLM refinement) and optional vision chat for PDF embedded images.
2. **Azure AI Vision (Computer Vision)** — image extraction: **Describe API** (captions/tags) + **Read API** (OCR for text in images, e.g. equations and labels).

## Environment variables

Set in `.env` or your environment (then use `docproc init-config --env .env` to generate `~/.config/docproc/docproc.yml`, or set them when running the CLI):

| Variable | Description |
|----------|-------------|
| `AZURE_OPENAI_API_KEY` | API key (same key often works for OpenAI + Vision in one resource) |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint, e.g. `https://<resource>.openai.azure.com/` |
| `AZURE_OPENAI_DEPLOYMENT` | Chat model deployment name (e.g. `gpt-4o`) |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Optional; used only if you had RAG (not used by CLI). |
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
- **Chat / vision:** Ensure `AZURE_OPENAI_DEPLOYMENT` matches the deployment name (e.g. `gpt-4o`, `gpt-4o-mini`).

## Using Azure AI Vision for images

If you have **Azure AI Vision** (Computer Vision) deployed:

1. Set `AZURE_VISION_ENDPOINT` to your Computer Vision endpoint (e.g. `https://<resource>.cognitiveservices.azure.com/`).
2. Use the same key as `AZURE_OPENAI_API_KEY` when both are in the same resource.
3. Image extraction will use:
   - **Describe API** — image captions and tags
   - **Read API (v3.2)** — OCR for text inside images (equations, labels)

See [CONFIGURATION.md](CONFIGURATION.md) for `ingest.use_vision` and related options.

## Demo (Go app)

The **demo** uses `OPENAI_API_KEY` for RAG and grading. To use Azure from the demo you would need to extend the Go app to support an Azure provider; out of the box it uses OpenAI.
