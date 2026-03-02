// Demo app: HTTP API + optional worker. Documents are processed via docproc CLI.
package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"

	"github.com/docproc/demo/internal/api"
	"github.com/docproc/demo/internal/blob"
	"github.com/docproc/demo/internal/config"
	"github.com/docproc/demo/internal/db"
	"github.com/docproc/demo/internal/grade"
	"github.com/docproc/demo/internal/mq"
	"github.com/docproc/demo/internal/rag"
	"github.com/docproc/demo/internal/worker"
	"github.com/sashabaranov/go-openai"
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
	if cfg.OpenAIKey != "" {
		ragClient = rag.New(pool, cfg.OpenAIKey, cfg.OpenAIModel)
		grader = grade.NewGrader(openai.NewClient(cfg.OpenAIKey), cfg.OpenAIModel)
	}

	// HTTP server
	handler := api.NewHandler(cfg, pool, store, pub, ragClient, grader)
	addr := os.Getenv("PORT")
	if addr == "" {
		addr = "8080"
	}
	log.Printf("Listening on :%s", addr)
	if err := http.ListenAndServe(":"+addr, handler); err != nil {
		log.Fatalf("server: %v", err)
	}
}

