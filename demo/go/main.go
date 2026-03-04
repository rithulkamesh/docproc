// Demo app: HTTP API + optional worker. Documents are processed via docproc CLI.
package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/rithulkamesh/docproc/demo/internal/api"
	"github.com/rithulkamesh/docproc/demo/internal/blob"
	"github.com/rithulkamesh/docproc/demo/internal/config"
	"github.com/rithulkamesh/docproc/demo/internal/db"
	"github.com/rithulkamesh/docproc/demo/internal/grade"
	"github.com/rithulkamesh/docproc/demo/internal/mq"
	"github.com/rithulkamesh/docproc/demo/internal/rag"
	"github.com/rithulkamesh/docproc/demo/internal/worker"
)

func main() {
	workerMode := flag.Bool("worker", false, "Run as document job worker (consume MQ, run docproc CLI)")
	flag.Parse()

	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("config: %v", err)
	}

	if *workerMode {
		ctx := context.Background()
		if err := worker.Run(ctx, cfg); err != nil {
			log.Fatalf("worker: %v", err)
		}
		return
	}

	ctx := context.Background()
	pool, err := db.NewPool(ctx, cfg.DatabaseURL)
	if err != nil {
		log.Fatalf("db: %v", err)
	}
	defer pool.Close()

	store, err := blob.NewStore(ctx, cfg.S3Endpoint, cfg.S3Region, cfg.S3Bucket)
	if err != nil {
		log.Fatalf("blob: %v", err)
	}

	pub, err := mq.NewPublisher(ctx, cfg.MQURL)
	if err != nil {
		log.Fatalf("mq: %v", err)
	}
	defer pub.Close()

	var ragClient *rag.RAG
	var grader *grade.Grader
	if client, chatModel, embeddingModel := cfg.AIClient(); client != nil {
		ragClient = rag.New(pool, client, chatModel, embeddingModel)
		grader = grade.NewGrader(client, chatModel)
	}

	// HTTP server with request timeout (60s)
	handler := api.NewHandler(cfg, pool, store, pub, ragClient, grader)
	addr := os.Getenv("PORT")
	if addr == "" {
		addr = "8080"
	}
	server := &http.Server{
		Addr:         ":" + addr,
		Handler:      handler,
		ReadTimeout:  60 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}
	log.Printf("Listening on :%s", addr)
	if err := server.ListenAndServe(); err != nil {
		log.Fatalf("server: %v", err)
	}
}

