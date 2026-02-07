// Package llm provides LLM integration via Ollama API.
package llm

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

// Client is an Ollama API client for LLM interactions.
type Client struct {
	client      *api.Client   // Official Ollama Go client
	model       string        // LLM model name (e.g., "gemma3:1b")
	history     []api.Message // Conversation history (system prompt at index 0)
	verbose     bool          // Enable verbose logging
	maxHistory  int           // Maximum conversation history length
	temperature float32       // LLM temperature
}

// Config holds LLM client configuration.
type Config struct {
	Host         string
	Model        string
	SystemPrompt string
	Verbose      bool
	MaxHistory   int
	Temperature  float32 // LLM temperature for controlling randomness
}

// NewClient creates a new Ollama client with optimized connection pooling.
// The HTTP client is configured for low-latency repeated requests to local LLM.
func NewClient(cfg *Config) (*Client, error) {
	maxHistory := cfg.MaxHistory
	if maxHistory <= 0 {
		maxHistory = 10 // Default to 10 message pairs
	}

	// Parse host URL
	host := strings.TrimSuffix(cfg.Host, "/")
	parsedURL, err := url.Parse(host)
	if err != nil {
		return nil, fmt.Errorf("invalid host URL: %w", err)
	}

	// Create official Ollama client with optimized http.Client
	// Configure connection pooling to reduce latency on repeated requests
	httpClient := &http.Client{
		Timeout: 60 * time.Second,
		Transport: &http.Transport{
			MaxIdleConns:        10,
			MaxIdleConnsPerHost: 10,
			IdleConnTimeout:     90 * time.Second,
			DisableCompression:  false,
		},
	}
	client := api.NewClient(parsedURL, httpClient)

	// Initialize history with system prompt at index 0
	history := make([]api.Message, 0, 1)
	history = append(history, api.Message{
		Role:    "system",
		Content: cfg.SystemPrompt,
	})

	return &Client{
		client:      client,
		model:       cfg.Model,
		history:     history,
		verbose:     cfg.Verbose,
		maxHistory:  maxHistory,
		temperature: cfg.Temperature,
	}, nil
}

// Chat sends a message and returns the response.
func (c *Client) Chat(ctx context.Context, userMessage string) (string, error) {
	// Append user message to history (system prompt already at index 0)
	c.history = append(c.history, api.Message{
		Role:    "user",
		Content: userMessage,
	})

	// Non-streaming mode
	stream := false

	var response api.ChatResponse
	err := c.client.Chat(ctx, &api.ChatRequest{
		Model:    c.model,
		Messages: c.history, // Pass history directly (includes system prompt)
		Stream:   &stream,
		Options: map[string]any{
			"temperature": c.temperature,
			"num_predict": 150,  // Limit response length for voice output
			"num_ctx":     1024, // Reduced context window to save GPU memory
		},
	}, func(resp api.ChatResponse) error {
		response = resp
		return nil
	})
	if err != nil {
		return "", fmt.Errorf("chat request failed: %w", err)
	}

	finalResponse := strings.TrimSpace(response.Message.Content)

	// Append assistant response to history
	c.history = append(c.history, api.Message{
		Role:    "assistant",
		Content: finalResponse,
	})

	// Trim history if too long
	c.trimHistory()

	return finalResponse, nil
}

// ClearHistory clears the conversation history (preserves system prompt).
func (c *Client) ClearHistory() {
	c.history = c.history[:1] // Keep only system prompt at index 0
}

// trimHistory keeps only the last N message pairs (preserves system prompt).
func (c *Client) trimHistory() {
	maxMessages := 1 + c.maxHistory*2 // system + user/assistant pairs
	if len(c.history) > maxMessages {
		// Keep system prompt (index 0) and last N pairs
		systemMsg := c.history[0]
		c.history = append([]api.Message{systemMsg}, c.history[len(c.history)-c.maxHistory*2:]...)
	}
}

// HealthCheck verifies the Ollama server is reachable.
func (c *Client) HealthCheck(ctx context.Context) error {
	// Use the Heartbeat method to check connectivity
	if err := c.client.Heartbeat(ctx); err != nil {
		return fmt.Errorf("cannot reach Ollama: %w", err)
	}
	return nil
}
