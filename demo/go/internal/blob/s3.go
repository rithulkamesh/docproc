package blob

import (
	"bytes"
	"context"
	"fmt"
	"io"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

// Store is an S3-compatible blob store (LocalStack or AWS).
type Store struct {
	client *s3.Client
	bucket string
}

// NewStore creates an S3 store with custom endpoint (e.g. LocalStack).
func NewStore(ctx context.Context, endpoint, region, bucket string) (*Store, error) {
	resolver := aws.EndpointResolverWithOptionsFunc(func(service, region string, options ...interface{}) (aws.Endpoint, error) {
		return aws.Endpoint{URL: endpoint}, nil
	})
	cfg := aws.Config{
		Region:                      region,
		EndpointResolverWithOptions: resolver,
		Credentials:                 credentials.NewStaticCredentialsProvider("test", "test", ""),
	}
	client := s3.NewFromConfig(cfg, func(o *s3.Options) {
		o.UsePathStyle = true
	})
	_, err := client.CreateBucket(ctx, &s3.CreateBucketInput{Bucket: aws.String(bucket)})
	if err != nil {
		// Bucket may already exist; continue
	}
	return &Store{client: client, bucket: bucket}, nil
}

// Put uploads body to key.
func (s *Store) Put(ctx context.Context, key string, body []byte) error {
	_, err := s.client.PutObject(ctx, &s3.PutObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(key),
		Body:   bytes.NewReader(body),
	})
	return err
}

// PutReader uploads from reader to key.
func (s *Store) PutReader(ctx context.Context, key string, body io.Reader, contentLength int64) error {
	_, err := s.client.PutObject(ctx, &s3.PutObjectInput{
		Bucket:        aws.String(s.bucket),
		Key:           aws.String(key),
		Body:          body,
		ContentLength: &contentLength,
	})
	return err
}

// Get downloads object at key.
func (s *Store) Get(ctx context.Context, key string) ([]byte, error) {
	out, err := s.client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		return nil, err
	}
	defer out.Body.Close()
	return io.ReadAll(out.Body)
}

// Delete removes object at key.
func (s *Store) Delete(ctx context.Context, key string) error {
	_, err := s.client.DeleteObject(ctx, &s3.DeleteObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(key),
	})
	return err
}

// UploadKey returns the blob key for an uploaded document file.
func UploadKey(docID, ext string) string {
	return "uploads/" + docID + ext
}

// TextKey returns the blob key for extracted text (.md).
func TextKey(docID string) string {
	return "texts/" + docID + ".md"
}
