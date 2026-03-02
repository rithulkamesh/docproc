package worker

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/docproc/demo/internal/blob"
	"github.com/docproc/demo/internal/config"
	"github.com/docproc/demo/internal/db"
	"github.com/docproc/demo/internal/mq"
	"github.com/docproc/demo/internal/rag"
)

// Run consumes document jobs from MQ, runs docproc CLI, updates DB and RAG.
func Run(ctx context.Context, cfg *config.Config) error {
	pool, err := db.NewPool(ctx, cfg.DatabaseURL)
	if err != nil {
		return fmt.Errorf("db: %w", err)
	}
	defer pool.Close()

	store, err := blob.NewStore(ctx, cfg.S3Endpoint, cfg.S3Region, cfg.S3Bucket)
	if err != nil {
		return fmt.Errorf("blob: %w", err)
	}

	var ragClient *rag.RAG
	if cfg.OpenAIKey != "" {
		ragClient = rag.New(pool, cfg.OpenAIKey, cfg.OpenAIModel)
	}

	handler := func(ctx context.Context, job mq.DocumentJob) error {
		return processJob(ctx, cfg, pool, store, ragClient, job)
	}

	consumer, err := mq.NewConsumer(ctx, cfg.MQURL, handler)
	if err != nil {
		return fmt.Errorf("mq consumer: %w", err)
	}
	defer consumer.Close()

	log.Println("Worker started. Consuming document jobs.")
	return consumer.Run(ctx)
}

func processJob(ctx context.Context, cfg *config.Config, pool *db.Pool, store *blob.Store, ragClient *rag.RAG, job mq.DocumentJob) error {
	data, err := store.Get(ctx, job.BlobKey)
	if err != nil {
		return fmt.Errorf("get blob: %w", err)
	}
	ext := filepath.Ext(job.BlobKey)
	tmpDir := os.TempDir()
	inputPath := filepath.Join(tmpDir, "docproc_"+job.DocID+ext)
	outputPath := filepath.Join(tmpDir, "docproc_"+job.DocID+".md")
	if err := os.WriteFile(inputPath, data, 0600); err != nil {
		return fmt.Errorf("write temp file: %w", err)
	}
	defer os.Remove(inputPath)
	defer os.Remove(outputPath)

	cmd := exec.CommandContext(ctx, cfg.DocprocCLI, "--file", inputPath, "-o", outputPath)
	cmd.Env = os.Environ()
	if err := cmd.Run(); err != nil {
		_ = pool.UpdateDocumentFailed(ctx, job.DocID, err.Error())
		return fmt.Errorf("docproc: %w", err)
	}

	md, err := os.ReadFile(outputPath)
	if err != nil {
		_ = pool.UpdateDocumentFailed(ctx, job.DocID, err.Error())
		return fmt.Errorf("read output: %w", err)
	}
	fullText := string(md)

	if err := pool.UpdateDocumentCompleted(ctx, job.DocID, fullText, 0, nil); err != nil {
		return fmt.Errorf("update doc: %w", err)
	}

	if ragClient != nil && fullText != "" {
		if err := ragClient.Index(ctx, job.DocID, fullText); err != nil {
			_ = pool.SetDocumentIndexError(ctx, job.DocID, err.Error())
			log.Printf("RAG index failed for %s: %v", job.DocID, err)
		}
	}

	return nil
}
