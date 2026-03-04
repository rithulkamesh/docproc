package worker

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/rithulkamesh/docproc/demo/internal/blob"
	"github.com/rithulkamesh/docproc/demo/internal/config"
	"github.com/rithulkamesh/docproc/demo/internal/db"
	"github.com/rithulkamesh/docproc/demo/internal/mq"
	"github.com/rithulkamesh/docproc/demo/internal/rag"
)

// docprocTimeout is the max time allowed for docproc CLI to process a document.
const docprocTimeout = 10 * time.Minute

// readProgressFile reads JSON lines from r (page, total, message), updates DB progress, and logs %.
// It also updates lastProgress so a heartbeat goroutine can refresh heartbeat without wiping percent.
func readProgressFile(ctx context.Context, r io.Reader, pool *db.Pool, docID string, done <-chan struct{}, lastProgress *sync.Map) {
	scanner := bufio.NewScanner(r)
	var lastLoggedPct int = -1
	for scanner.Scan() {
		select {
		case <-done:
			return
		default:
		}
		var line struct {
			Page    int    `json:"page"`
			Total   int    `json:"total"`
			Message string `json:"message"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &line); err != nil {
			continue
		}
		total := line.Total
		if total < 1 {
			total = 1
		}
		page := line.Page
		if page < 0 {
			page = 0
		}
		if page >= total {
			page = total - 1
		}
		pct := 0
		if total > 0 {
			pct = page * 100 / total
		}
		progress := map[string]interface{}{
			"page":      page,
			"total":     total,
			"message":   line.Message,
			"heartbeat": time.Now().Format(time.RFC3339),
			"percent":   pct,
		}
		if lastProgress != nil {
			lastProgress.Store(docID, progress)
		}
		if err := pool.UpdateDocumentProgress(ctx, docID, progress); err != nil {
			log.Printf("[worker] doc_id=%s UpdateDocumentProgress: %v", docID, err)
		}
		// Log every ~10% to avoid spam
		if pct >= lastLoggedPct+10 || pct == 100 || lastLoggedPct < 0 {
			log.Printf("[worker] doc_id=%s progress %d%% (%d/%d)", docID, pct, page, total)
			lastLoggedPct = pct
		}
	}
}

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
	if client, chatModel, embeddingModel := cfg.AIClient(); client != nil {
		ragClient = rag.New(pool, client, chatModel, embeddingModel)
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

	progressFile, err := os.CreateTemp(tmpDir, "docproc_progress_")
	if err != nil {
		log.Printf("[worker] doc_id=%s create progress file failed: %v", job.DocID, err)
		return fmt.Errorf("progress file: %w", err)
	}
	progressPath := progressFile.Name()
	progressFile.Close()
	defer os.Remove(progressPath)

	progressRead, err := os.Open(progressPath)
	if err != nil {
		log.Printf("[worker] doc_id=%s open progress file for read failed: %v", job.DocID, err)
		return fmt.Errorf("progress file read: %w", err)
	}
	defer progressRead.Close()

	done := make(chan struct{})
	var lastProgress sync.Map
	go readProgressFile(ctx, progressRead, pool, job.DocID, done, &lastProgress)
	// Fallback heartbeat so UI shows "live" while CLI is loading (e.g. before first progress line).
	go func() {
		ticker := time.NewTicker(3 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				heartbeat := time.Now().Format(time.RFC3339)
				progress := map[string]interface{}{
					"message":   "Extracting…",
					"heartbeat": heartbeat,
				}
				if v, ok := lastProgress.Load(job.DocID); ok {
					if m, ok := v.(map[string]interface{}); ok {
						for k, val := range m {
							progress[k] = val
						}
					}
				}
				progress["heartbeat"] = heartbeat
				_ = pool.UpdateDocumentProgress(ctx, job.DocID, progress)
			}
		}
	}()

	cmd := exec.CommandContext(procCtx, cfg.DocprocCLI, "--file", inputPath, "-o", outputPath, "--progress-file", progressPath)
	cmd.Env = os.Environ()
	// Prevent CLI from blocking on stdin (e.g. accidental prompt).
	stdinFile, err := os.Open(os.DevNull)
	if err != nil {
		log.Printf("[worker] doc_id=%s failed to open stdin: %v", job.DocID, err)
		return fmt.Errorf("open stdin: %w", err)
	}
	defer stdinFile.Close()
	cmd.Stdin = stdinFile
	// Drain stderr in a goroutine so docproc's tqdm/progress writes don't fill the pipe and block.
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		log.Printf("[worker] doc_id=%s StderrPipe: %v", job.DocID, err)
		return fmt.Errorf("stderr pipe: %w", err)
	}
	var stderr bytes.Buffer
	go func() { _, _ = io.Copy(&stderr, stderrPipe) }()
	log.Printf("[worker] doc_id=%s running docproc cli=%s", job.DocID, cfg.DocprocCLI)
	if err := cmd.Start(); err != nil {
		log.Printf("[worker] doc_id=%s cmd.Start failed: %v", job.DocID, err)
		return fmt.Errorf("start docproc: %w", err)
	}
	err = cmd.Wait()
	close(done)
	if err != nil {
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
	err = nil

	md, err := os.ReadFile(outputPath)
	if err != nil {
		log.Printf("[worker] doc_id=%s read output failed: %v", job.DocID, err)
		_ = pool.UpdateDocumentFailed(ctx, job.DocID, err.Error())
		return fmt.Errorf("read output: %w", err)
	}
	fullText := string(md)
	log.Printf("[worker] doc_id=%s docproc completed output_len=%d", job.DocID, len(fullText))

	pages := parsePageCountFromMarkdown(fullText)
	if pages > 0 {
		fullText = stripPageCountComment(fullText)
	}
	if err := pool.UpdateDocumentCompleted(ctx, job.DocID, fullText, pages, nil); err != nil {
		log.Printf("[worker] doc_id=%s UpdateDocumentCompleted failed: %v", job.DocID, err)
		return fmt.Errorf("update doc: %w", err)
	}
	log.Printf("[worker] doc_id=%s status set to completed in DB", job.DocID)

	if ragClient != nil && fullText != "" {
		if title, err := ragClient.SuggestDocumentTitle(ctx, fullText); err == nil && title != "" {
			if updateErr := pool.UpdateDocumentDisplayName(ctx, job.DocID, title); updateErr != nil {
				log.Printf("[worker] doc_id=%s UpdateDocumentDisplayName failed: %v", job.DocID, updateErr)
			} else {
				log.Printf("[worker] doc_id=%s display_name=%s", job.DocID, title)
			}
		} else if err != nil {
			log.Printf("[worker] doc_id=%s suggest title failed: %v", job.DocID, err)
		}
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

var pagesCommentRE = regexp.MustCompile(`^\s*<!--\s*PAGES:\s*(\d+)\s*-->\s*\n?`)

// parsePageCountFromMarkdown returns page count if the markdown starts with <!-- PAGES: N -->, else 0.
func parsePageCountFromMarkdown(md string) int {
	loc := pagesCommentRE.FindStringSubmatchIndex(md)
	if loc == nil {
		return 0
	}
	n, _ := strconv.Atoi(md[loc[2]:loc[3]])
	return n
}

// stripPageCountComment removes a leading <!-- PAGES: N --> line from markdown.
func stripPageCountComment(md string) string {
	return pagesCommentRE.ReplaceAllString(md, "")
}
