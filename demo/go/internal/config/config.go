package config

import (
	"log"
	"os"
	"strings"

	"github.com/rithulkamesh/docproc/demo/internal/secure"
	"github.com/sashabaranov/go-openai"
)

// DefaultEncryptionPassphrase is used when DOCPROC_ENCRYPTION_KEY is not set (dev only; set your own in production).
const DefaultEncryptionPassphrase = "docproc-dev-default-change-in-production"

// Config holds demo app configuration (env or file).
type Config struct {
	DatabaseURL string // PostgreSQL (documents, projects, pgvector)
	S3Endpoint  string // LocalStack or real S3 (e.g. http://localhost:4566)
	S3Bucket    string
	S3Region    string
	MQURL       string // RabbitMQ AMQP URL
	DocprocCLI  string // Path to docproc binary (default: docproc)
	// EncryptionKey is 32 bytes for encrypting/decrypting AI API keys in DB. From DOCPROC_ENCRYPTION_KEY.
	EncryptionKey []byte
	OpenAIKey     string // For embeddings + LLM (RAG) — fallback when no DB config
	OpenAIModel   string
	// Azure OpenAI (used when OPENAI_API_KEY is not set)
	AzureAPIKey             string
	AzureEndpoint           string
	AzureDeployment          string
	AzureEmbeddingDeployment string
}

// Load reads config from environment. Uses OPENAI_API_KEY if set; otherwise AZURE_OPENAI_*.
// DOCPROC_ENCRYPTION_KEY is used to encrypt AI keys stored in the DB (32 bytes or passphrase).
func Load() (*Config, error) {
	c := &Config{
		DatabaseURL: getEnv("DATABASE_URL", "postgresql://docproc:docproc@localhost:5432/docproc?sslmode=disable"),
		S3Endpoint:  getEnv("S3_ENDPOINT", "http://localhost:4566"),
		S3Bucket:    getEnv("S3_BUCKET", "docproc-demo"),
		S3Region:    getEnv("AWS_REGION", "us-east-1"),
		MQURL:       getEnv("MQ_URL", "amqp://docproc:docproc@localhost:5672/"),
		DocprocCLI:  getEnv("DOCPROC_CLI", "docproc"),
		EncryptionKey: keyFromEnvOrDefault(os.Getenv("DOCPROC_ENCRYPTION_KEY")),
		OpenAIKey:   os.Getenv("OPENAI_API_KEY"),
		OpenAIModel: getEnv("OPENAI_MODEL", "gpt-4o-mini"),
		AzureAPIKey:             os.Getenv("AZURE_OPENAI_API_KEY"),
		AzureEndpoint:           os.Getenv("AZURE_OPENAI_ENDPOINT"),
		AzureDeployment:         getEnv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
		AzureEmbeddingDeployment: getEnv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
	}
	return c, nil
}

// HasAI returns true if either OpenAI or Azure is configured (for RAG/grading).
func (c *Config) HasAI() bool {
	return c.OpenAIKey != "" || (c.AzureAPIKey != "" && c.AzureEndpoint != "")
}

// PrimaryAI returns the active provider id for status: "openai" or "azure".
func (c *Config) PrimaryAI() string {
	if c.OpenAIKey != "" {
		return "openai"
	}
	if c.AzureAPIKey != "" && c.AzureEndpoint != "" {
		return "azure"
	}
	return "openai"
}

// DefaultRAGModel returns the chat model name used for RAG (OpenAI or Azure deployment).
func (c *Config) DefaultRAGModel() string {
	if c.OpenAIKey != "" {
		return c.OpenAIModel
	}
	if c.AzureAPIKey != "" && c.AzureEndpoint != "" {
		return c.AzureDeployment
	}
	return c.OpenAIModel
}

// DefaultEmbeddingDeployment returns the Azure embedding deployment name when Azure is primary, else empty.
// Used only for /status so the frontend can show server defaults; never exposes keys or endpoints.
func (c *Config) DefaultEmbeddingDeployment() string {
	if c.PrimaryAI() == "azure" && c.AzureEmbeddingDeployment != "" {
		return c.AzureEmbeddingDeployment
	}
	return ""
}

// AIClient returns an OpenAI-compatible client and model names (chat, embedding) using the default provider:
// OPENAI_API_KEY if set, else AZURE_OPENAI_* if set. Returns (nil, "", "") when neither is configured.
func (c *Config) AIClient() (client *openai.Client, chatModel, embeddingModel string) {
	if c.OpenAIKey != "" {
		return openai.NewClient(c.OpenAIKey), c.OpenAIModel, string(openai.AdaEmbeddingV2)
	}
	if c.AzureAPIKey != "" && c.AzureEndpoint != "" {
		cfg := openai.DefaultAzureConfig(c.AzureAPIKey, c.AzureEndpoint)
		return openai.NewClientWithConfig(cfg), c.AzureDeployment, c.AzureEmbeddingDeployment
	}
	return nil, "", ""
}

func getEnv(key, defaultVal string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultVal
}

// keyFromEnvOrDefault returns a 32-byte key derived from passphrase. If passphrase is empty, uses a default (dev only; log warning).
func keyFromEnvOrDefault(passphrase string) []byte {
	if strings.TrimSpace(passphrase) == "" {
		log.Print("[config] DOCPROC_ENCRYPTION_KEY not set; using default (fine for dev; set your own in production)")
		return secure.KeyFromEnv(DefaultEncryptionPassphrase)
	}
	return secure.KeyFromEnv(passphrase)
}
