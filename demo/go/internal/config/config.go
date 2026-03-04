package config

import (
	"os"

	"github.com/sashabaranov/go-openai"
)

// Config holds demo app configuration (env or file).
type Config struct {
	DatabaseURL string // PostgreSQL (documents, projects, pgvector)
	S3Endpoint  string // LocalStack or real S3 (e.g. http://localhost:4566)
	S3Bucket    string
	S3Region    string
	MQURL       string // RabbitMQ AMQP URL
	DocprocCLI  string // Path to docproc binary (default: docproc)
	OpenAIKey   string // For embeddings + LLM (RAG)
	OpenAIModel string
	// Azure OpenAI (used when OPENAI_API_KEY is not set)
	AzureAPIKey               string
	AzureEndpoint             string
	AzureDeployment            string
	AzureEmbeddingDeployment   string
}

// Load reads config from environment. Uses OPENAI_API_KEY if set; otherwise AZURE_OPENAI_*.
func Load() (*Config, error) {
	c := &Config{
		DatabaseURL: getEnv("DATABASE_URL", "postgresql://docproc:docproc@localhost:5432/docproc?sslmode=disable"),
		S3Endpoint:  getEnv("S3_ENDPOINT", "http://localhost:4566"),
		S3Bucket:    getEnv("S3_BUCKET", "docproc-demo"),
		S3Region:    getEnv("AWS_REGION", "us-east-1"),
		MQURL:       getEnv("MQ_URL", "amqp://docproc:docproc@localhost:5672/"),
		DocprocCLI:  getEnv("DOCPROC_CLI", "docproc"),
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
