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
	log.Printf("[worker] job received doc_id=%s blob_key=%s project_id=%s", job.DocID, job.BlobKey, job.ProjectID)
	// Use a timeout for docproc CLI to avoid hanging on large/slow documents
	procCtx, cancel := context.WithTimeout(ctx, docprocTimeout)
	defer cancel()

	data, err := store.Get(ctx, job.BlobKey)
	if err != nil {
		log.Printf("[worker] doc_id=%s get blob failed: %v", job.DocID, err)
		return fmt.Errorf("get blob: %w", err)
	}
	log.Printf("[worker] doc_id=%s blob fetched size=%d", job.DocID, len(data))

	ext := filepath.Ext(job.BlobKey)
	tmpDir := os.TempDir()
	inputPath := filepath.Join(tmpDir, "docproc_"+job.DocID+ext)
	outputPath := filepath.Join(tmpDir, "docproc_"+job.DocID+".md")
	if err := os.WriteFile(inputPath, data, 0600); err != nil {
		log.Printf("[worker] doc_id=%s write temp file failed: %v", job.DocID, err)
		return fmt.Errorf("write temp file: %w", err)
	}
	defer os.Remove(inputPath)
	defer os.Remove(outputPath)
	log.Printf("[worker] doc_id=%s temp file written path=%s", job.DocID, inputPath)

	cmd := exec.CommandContext(procCtx, cfg.DocprocCLI, "--file", inputPath, "-o", outputPath)
	cmd.Env = os.Environ()
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	log.Printf("[worker] doc_id=%s running docproc cli=%s", job.DocID, cfg.DocprocCLI)
	if err := cmd.Run(); err != nil {
		errMsg := err.Error()
		if stderr.Len() > 0 {
			errMsg = fmt.Sprintf("%s (stderr: %s)", errMsg, stderr.String())
		}
		log.Printf("[worker] doc_id=%s docproc failed: %s", job.DocID, errMsg)
		// When running in Docker without docproc CLI, mark as completed with placeholder so UI doesn't show "Failed".
		if errors.Is(err, exec.ErrNotFound) || strings.Contains(errMsg, "executable file not found") || strings.Contains(errMsg, "no such file") {
			log.Printf("[worker] doc_id=%s docproc CLI not available, marking as stored (no extraction)", job.DocID)
			placeholder := "# Document stored\n\nProcessing was skipped because the docproc CLI is not available in this environment (e.g. Docker-only). Install docproc and run the worker locally for full extraction."
			if updateErr := pool.UpdateDocumentCompleted(ctx, job.DocID, placeholder, 0, nil); updateErr != nil {
				_ = pool.UpdateDocumentFailed(ctx, job.DocID, errMsg)
			}
			return nil
		}
		_ = pool.UpdateDocumentFailed(ctx, job.DocID, errMsg)
		log.Printf("[worker] doc_id=%s status set to failed in DB", job.DocID)
		return fmt.Errorf("docproc: %w", err)
	}

	md, err := os.ReadFile(outputPath)
	if err != nil {
		log.Printf("[worker] doc_id=%s read output failed: %v", job.DocID, err)
		_ = pool.UpdateDocumentFailed(ctx, job.DocID, err.Error())
		return fmt.Errorf("read output: %w", err)
	}
	fullText := string(md)
	log.Printf("[worker] doc_id=%s docproc completed output_len=%d", job.DocID, len(fullText))

	if err := pool.UpdateDocumentCompleted(ctx, job.DocID, fullText, 0, nil); err != nil {
		log.Printf("[worker] doc_id=%s UpdateDocumentCompleted failed: %v", job.DocID, err)
		return fmt.Errorf("update doc: %w", err)
	}
	log.Printf("[worker] doc_id=%s status set to completed in DB", job.DocID)

	if ragClient != nil && fullText != "" {
		log.Printf("[worker] doc_id=%s indexing to RAG", job.DocID)
		if err := ragClient.Index(ctx, job.DocID, fullText); err != nil {
			_ = pool.SetDocumentIndexError(ctx, job.DocID, err.Error())
			log.Printf("[worker] doc_id=%s RAG index failed: %v", job.DocID, err)
		} else {
			log.Printf("[worker] doc_id=%s RAG index done", job.DocID)
		}
	}

	log.Printf("[worker] doc_id=%s job finished successfully", job.DocID)
	return nil
}
