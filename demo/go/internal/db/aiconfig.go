package db

import (
	"context"
	"errors"

	"github.com/jackc/pgx/v5"
	"github.com/rithulkamesh/docproc/demo/internal/secure"
)

const aiConfigID = "default"

// AIConfigRow is the stored row (api_key_encrypted is never exposed to the client).
type AIConfigRow struct {
	Provider             string
	Model                string
	APIKeyEncrypted      string
	BaseURL              string
	Endpoint             string
	Deployment           string
	EmbeddingDeployment  string
}

// AIConfigForAPI is returned by GET /ai-config (api_key never sent to client).
type AIConfigForAPI struct {
	Provider            string `json:"provider"`
	Model               string `json:"model"`
	APIKeyConfigured    bool   `json:"api_key_configured"`
	BaseURL             string `json:"base_url,omitempty"`
	Endpoint            string `json:"endpoint,omitempty"`
	Deployment          string `json:"deployment,omitempty"`
	EmbeddingDeployment string `json:"embedding_deployment,omitempty"`
}

// AIConfigDecrypted is for internal use (query/stream) with decrypted api_key.
type AIConfigDecrypted struct {
	Provider            string
	Model               string
	APIKey              string
	BaseURL             string
	Endpoint            string
	Deployment          string
	EmbeddingDeployment string
}

// GetAIConfig returns the stored config for API response (no decryption).
func (p *Pool) GetAIConfig(ctx context.Context) (*AIConfigForAPI, error) {
	var row struct {
		Provider            string
		Model               string
		APIKeyEncrypted     *string
		BaseURL             *string
		Endpoint            *string
		Deployment          *string
		EmbeddingDeployment *string
	}
	err := p.QueryRow(ctx, `SELECT provider, model, api_key_encrypted, base_url, endpoint, deployment, embedding_deployment
		FROM docproc_ai_config WHERE id = $1`, aiConfigID).Scan(
		&row.Provider, &row.Model, &row.APIKeyEncrypted, &row.BaseURL, &row.Endpoint, &row.Deployment, &row.EmbeddingDeployment,
	)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, nil
		}
		return nil, err
	}
	out := &AIConfigForAPI{
		Provider:         row.Provider,
		Model:            row.Model,
		APIKeyConfigured: row.APIKeyEncrypted != nil && *row.APIKeyEncrypted != "",
	}
	if row.BaseURL != nil {
		out.BaseURL = *row.BaseURL
	}
	if row.Endpoint != nil {
		out.Endpoint = *row.Endpoint
	}
	if row.Deployment != nil {
		out.Deployment = *row.Deployment
	}
	if row.EmbeddingDeployment != nil {
		out.EmbeddingDeployment = *row.EmbeddingDeployment
	}
	return out, nil
}

// GetAIConfigDecrypted loads config and decrypts api_key for internal use (query/stream).
// encKey is the 32-byte encryption key (e.g. from secure.KeyFromEnv(os.Getenv("DOCPROC_ENCRYPTION_KEY"))).
func (p *Pool) GetAIConfigDecrypted(ctx context.Context, encKey []byte) (*AIConfigDecrypted, error) {
	var row struct {
		Provider            string
		Model               string
		APIKeyEncrypted     *string
		BaseURL             *string
		Endpoint            *string
		Deployment          *string
		EmbeddingDeployment *string
	}
	err := p.QueryRow(ctx, `SELECT provider, model, api_key_encrypted, base_url, endpoint, deployment, embedding_deployment
		FROM docproc_ai_config WHERE id = $1`, aiConfigID).Scan(
		&row.Provider, &row.Model, &row.APIKeyEncrypted, &row.BaseURL, &row.Endpoint, &row.Deployment, &row.EmbeddingDeployment,
	)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, nil
		}
		return nil, err
	}
	out := &AIConfigDecrypted{Provider: row.Provider, Model: row.Model}
	if row.APIKeyEncrypted != nil && *row.APIKeyEncrypted != "" && len(encKey) == 32 {
		dec, err := secure.Decrypt(encKey, *row.APIKeyEncrypted)
		if err != nil {
			return nil, err
		}
		out.APIKey = dec
	}
	if row.BaseURL != nil {
		out.BaseURL = *row.BaseURL
	}
	if row.Endpoint != nil {
		out.Endpoint = *row.Endpoint
	}
	if row.Deployment != nil {
		out.Deployment = *row.Deployment
	}
	if row.EmbeddingDeployment != nil {
		out.EmbeddingDeployment = *row.EmbeddingDeployment
	}
	return out, nil
}

// SaveAIConfig saves AI config. When in.APIKey is nil, stored key is left unchanged; when *in.APIKey is "", key is cleared; otherwise encrypted and stored. encKey must be 32 bytes.
func (p *Pool) SaveAIConfig(ctx context.Context, encKey []byte, in *AIConfigSaveInput) error {
	var apiKeyEncrypted *string
	if in.APIKey != nil {
		if *in.APIKey != "" && len(encKey) == 32 {
			enc, err := secure.Encrypt(encKey, *in.APIKey)
			if err != nil {
				return err
			}
			apiKeyEncrypted = &enc
		}
		// else *in.APIKey == "" -> clear (apiKeyEncrypted stays nil)
	}

	if in.APIKey != nil {
		// Key was sent: upsert including api_key_encrypted (may set to NULL to clear)
		_, err := p.Exec(ctx, `
			INSERT INTO docproc_ai_config (id, provider, model, api_key_encrypted, base_url, endpoint, deployment, embedding_deployment, updated_at)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
			ON CONFLICT (id) DO UPDATE SET
				provider = EXCLUDED.provider,
				model = EXCLUDED.model,
				api_key_encrypted = EXCLUDED.api_key_encrypted,
				base_url = EXCLUDED.base_url,
				endpoint = EXCLUDED.endpoint,
				deployment = EXCLUDED.deployment,
				embedding_deployment = EXCLUDED.embedding_deployment,
				updated_at = NOW()`,
			aiConfigID, in.Provider, in.Model, apiKeyEncrypted, ptr(in.BaseURL), ptr(in.Endpoint), ptr(in.Deployment), ptr(in.EmbeddingDeployment),
		)
		return err
	}
	// Key not sent: update other fields only, leave api_key_encrypted unchanged
	_, err := p.Exec(ctx, `
		INSERT INTO docproc_ai_config (id, provider, model, base_url, endpoint, deployment, embedding_deployment, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
		ON CONFLICT (id) DO UPDATE SET
			provider = EXCLUDED.provider,
			model = EXCLUDED.model,
			base_url = EXCLUDED.base_url,
			endpoint = EXCLUDED.endpoint,
			deployment = EXCLUDED.deployment,
			embedding_deployment = EXCLUDED.embedding_deployment,
			updated_at = NOW()`,
		aiConfigID, in.Provider, in.Model, ptr(in.BaseURL), ptr(in.Endpoint), ptr(in.Deployment), ptr(in.EmbeddingDeployment),
	)
	return err
}

// AIConfigSaveInput is the input for saving AI config (e.g. from PUT /ai-config).
// APIKey: nil = do not change stored key, ptr to "" = clear key, ptr to "sk-..." = set key.
type AIConfigSaveInput struct {
	Provider            string  `json:"provider"`
	Model               string  `json:"model"`
	APIKey              *string `json:"api_key"` // nil = not sent, don't change; "" = clear; "sk-..." = set
	BaseURL             string  `json:"base_url"`
	Endpoint            string  `json:"endpoint"`
	Deployment          string  `json:"deployment"`
	EmbeddingDeployment string  `json:"embedding_deployment"`
}

func ptr(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}