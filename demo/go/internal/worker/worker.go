package worker

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/rithulkamesh/docproc/demo/internal/blob"
	"github.com/rithulkamesh/docproc/demo/internal/config"
	"github.com/rithulkamesh/docproc/demo/internal/db"
	"github.com/rithulkamesh/docproc/demo/internal/mq"
	"github.com/rithulkamesh/docproc/demo/internal/rag"
)

// docprocTimeout is the max time allowed for docproc CLI to process a document.
const docprocTimeout = 10 * time.Minute

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
	// Use a timeout for docproc CLI to avoid hanging on large/slow documents
	procCtx, cancel := context.WithTimeout(ctx, docprocTimeout)
	defer cancel()

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

	cmd := exec.CommandContext(procCtx, cfg.DocprocCLI, "--file", inputPath, "-o", outputPath)
	cmd.Env = os.Environ()
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		errMsg := err.Error()
		if stderr.Len() > 0 {
			errMsg = fmt.Sprintf("%s (stderr: %s)", errMsg, stderr.String())
		}
		// When running in Docker without docproc CLI, mark as completed with placeholder so UI doesn't show "Failed".
		if errors.Is(err, exec.ErrNotFound) || strings.Contains(errMsg, "executable file not found") || strings.Contains(errMsg, "no such file") {
			log.Printf("docproc CLI not available, marking document %s as stored (no extraction)", job.DocID)
			placeholder := "# Document stored\n\nProcessing was skipped because the docproc CLI is not available in this environment (e.g. Docker-only). Install docproc and run the worker locally for full extraction."
			if updateErr := pool.UpdateDocumentCompleted(ctx, job.DocID, placeholder, 0, nil); updateErr != nil {
				_ = pool.UpdateDocumentFailed(ctx, job.DocID, errMsg)
			}
			return nil
		}
		_ = pool.UpdateDocumentFailed(ctx, job.DocID, errMsg)
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
