package secure

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"io"
)

// KeyFromEnv derives a 32-byte AES-256 key from a passphrase (e.g. DOCPROC_ENCRYPTION_KEY).
// If passphrase is already 32 bytes, it is used as-is; otherwise SHA256 is used.
func KeyFromEnv(passphrase string) []byte {
	if len(passphrase) == 32 {
		return []byte(passphrase)
	}
	h := sha256.Sum256([]byte(passphrase))
	return h[:]
}

// Encrypt encrypts plaintext with AES-256-GCM. Key must be 32 bytes.
// Returns base64(nonce || ciphertext) for storage.
func Encrypt(key []byte, plaintext string) (string, error) {
	if len(key) != 32 {
		return "", errors.New("key must be 32 bytes")
	}
	if plaintext == "" {
		return "", nil
	}
	block, err := aes.NewCipher(key)
	if err != nil {
		return "", err
	}
	aesgcm, err := cipher.NewGCM(block)
	if err != nil {
		return "", err
	}
	nonce := make([]byte, aesgcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return "", err
	}
	ciphertext := aesgcm.Seal(nil, nonce, []byte(plaintext), nil)
	blob := append(nonce, ciphertext...)
	return base64.StdEncoding.EncodeToString(blob), nil
}

// Decrypt decrypts a value produced by Encrypt. Key must be 32 bytes.
func Decrypt(key []byte, encoded string) (string, error) {
	if len(key) != 32 {
		return "", errors.New("key must be 32 bytes")
	}
	if encoded == "" {
		return "", nil
	}
	blob, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return "", err
	}
	block, err := aes.NewCipher(key)
	if err != nil {
		return "", err
	}
	aesgcm, err := cipher.NewGCM(block)
	if err != nil {
		return "", err
	}
	nonceSize := aesgcm.NonceSize()
	if len(blob) < nonceSize {
		return "", errors.New("ciphertext too short")
	}
	nonce, ciphertext := blob[:nonceSize], blob[nonceSize:]
	plain, err := aesgcm.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return "", err
	}
	return string(plain), nil
}
