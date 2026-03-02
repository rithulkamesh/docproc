package mq

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	amqp "github.com/rabbitmq/amqp091-go"
)

const (
	queueName = "document-jobs"
)

// DocumentJob is the message body for a document processing job.
type DocumentJob struct {
	DocID     string `json:"doc_id"`
	BlobKey   string `json:"blob_key"`
	ProjectID string `json:"project_id"`
}

// Publisher publishes document jobs to the queue.
type Publisher struct {
	ch    *amqp.Channel
	queue string
}

// NewPublisher connects to RabbitMQ and declares the queue.
func NewPublisher(ctx context.Context, amqpURL string) (*Publisher, error) {
	conn, err := amqp.Dial(amqpURL)
	if err != nil {
		return nil, fmt.Errorf("amqp dial: %w", err)
	}
	ch, err := conn.Channel()
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("channel: %w", err)
	}
	_, err = ch.QueueDeclare(queueName, true, false, false, false, nil)
	if err != nil {
		ch.Close()
		conn.Close()
		return nil, fmt.Errorf("queue declare: %w", err)
	}
	return &Publisher{ch: ch, queue: queueName}, nil
}

// Publish sends a document job.
func (p *Publisher) Publish(ctx context.Context, job DocumentJob) error {
	body, err := json.Marshal(job)
	if err != nil {
		return err
	}
	return p.ch.PublishWithContext(ctx, "", p.queue, false, false, amqp.Publishing{
		ContentType: "application/json",
		Body:        body,
	})
}

// Close closes the channel (caller may close connection separately if needed).
func (p *Publisher) Close() error {
	return p.ch.Close()
}

// Consumer consumes document jobs and calls the handler.
type Consumer struct {
	conn    *amqp.Connection
	ch      *amqp.Channel
	queue   string
	handler func(ctx context.Context, job DocumentJob) error
}

// NewConsumer connects and declares the queue.
func NewConsumer(ctx context.Context, amqpURL string, handler func(ctx context.Context, job DocumentJob) error) (*Consumer, error) {
	conn, err := amqp.Dial(amqpURL)
	if err != nil {
		return nil, fmt.Errorf("amqp dial: %w", err)
	}
	ch, err := conn.Channel()
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("channel: %w", err)
	}
	_, err = ch.QueueDeclare(queueName, true, false, false, false, nil)
	if err != nil {
		ch.Close()
		conn.Close()
		return nil, fmt.Errorf("queue declare: %w", err)
	}
	return &Consumer{conn: conn, ch: ch, queue: queueName, handler: handler}, nil
}

// Run consumes messages until context is cancelled.
func (c *Consumer) Run(ctx context.Context) error {
	deliveries, err := c.ch.Consume(c.queue, "demo-worker", false, false, false, false, nil)
	if err != nil {
		return fmt.Errorf("consume: %w", err)
	}
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case d, ok := <-deliveries:
			if !ok {
				return fmt.Errorf("delivery channel closed")
			}
			var job DocumentJob
			if err := json.Unmarshal(d.Body, &job); err != nil {
				log.Printf("mq: invalid job body: %v", err)
				_ = d.Nack(false, false)
				continue
			}
			if err := c.handler(ctx, job); err != nil {
				log.Printf("mq: handle job %s: %v", job.DocID, err)
				_ = d.Nack(false, true)
				continue
			}
			_ = d.Ack(false)
		}
	}
}

// Close closes channel and connection.
func (c *Consumer) Close() error {
	_ = c.ch.Close()
	return c.conn.Close()
}
