package config

import (
	"os"
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
}

// Load reads config from environment.
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
	}
	return c, nil
}

func getEnv(key, defaultVal string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultVal
}
