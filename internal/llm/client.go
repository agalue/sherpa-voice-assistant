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

// Client is an Ollama API client for LLM interactions with agentic tool support.
type Client struct {
	client      *api.Client   // Official Ollama Go client
	model       string        // LLM model name (e.g., "qwen2.5:3b")
	history     []api.Message // Conversation history (system prompt at index 0)
	verbose     bool          // Enable verbose logging
	maxHistory  int           // Maximum conversation history length
	temperature float32       // LLM temperature
	tools       []api.Tool    // Available tools for the agent
	registry    ToolRegistry  // Tool execution registry
}

// Config holds LLM client configuration.
type Config struct {
	Host         string
	Model        string
	SystemPrompt string
	Verbose      bool
	MaxHistory   int
	Temperature  float32 // LLM temperature for controlling randomness
	SearxngURL   string  // Optional SearXNG URL for web search
}

// NewClient creates a new Ollama client with optimized connection pooling and agentic tool support.
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

	// Build system prompt with tool usage instructions
	systemPrompt := cfg.SystemPrompt + " CRITICAL: You have two tools available: get_weather and search_web. " +
		"When asked about current events, news, facts, sports results, or anything you don't know: " +
		"IMMEDIATELY use search_web tool - DO NOT say you lack information or capabilities. " +
		"For weather queries: use get_weather tool. Always use tools proactively."

	// Initialize history with enhanced system prompt at index 0
	history := make([]api.Message, 0, 1)
	history = append(history, api.Message{
		Role:    "system",
		Content: systemPrompt,
	})

	// Create tool registry and definitions
	registry := CreateToolRegistry(cfg.SearxngURL)
	tools := GetToolDefinitions()

	return &Client{
		client:      client,
		model:       cfg.Model,
		history:     history,
		verbose:     cfg.Verbose,
		maxHistory:  maxHistory,
		temperature: cfg.Temperature,
		tools:       tools,
		registry:    registry,
	}, nil
}

// Chat sends a message and returns the response using agentic loop with tool calling.
// This method implements the agentic loop: LLM → Tool Calls → Tool Results → LLM → Final Answer
func (c *Client) Chat(ctx context.Context, userMessage string) (string, error) {
	// Append user message to history (system prompt already at index 0)
	c.history = append(c.history, api.Message{
		Role:    "user",
		Content: userMessage,
	})

	// Non-streaming mode
	stream := false

	// Agentic loop: keep calling LLM until no more tools are needed
	maxIterations := 5 // Prevent infinite loops
	for iteration := 0; iteration < maxIterations; iteration++ {
		var response api.ChatResponse
		err := c.client.Chat(ctx, &api.ChatRequest{
			Model:    c.model,
			Messages: c.history, // Pass history directly (includes system prompt)
			Tools:    c.tools,   // Provide available tools
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

		// Check if LLM wants to use tools
		if len(response.Message.ToolCalls) == 0 {
			// No tools needed, we have the final answer
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

		// Execute tools and collect results
		// First, append the assistant's message with tool calls to history
		c.history = append(c.history, response.Message)

		// Execute each tool call
		for _, toolCall := range response.Message.ToolCalls {
			toolFunc, exists := c.registry[toolCall.Function.Name]
			if !exists {
				// Unknown tool, add error message
				c.history = append(c.history, api.Message{
					Role:    "tool",
					Content: fmt.Sprintf("Error: Unknown tool '%s'", toolCall.Function.Name),
				})
				continue
			}

			// Execute the tool (convert Arguments to JSON string)
			argJSON := toolCall.Function.Arguments.String()
			result, err := toolFunc(ctx, argJSON)
			if err != nil {
				// Tool execution failed, add error message
				result = fmt.Sprintf("Error executing tool: %v", err)
			}

			// Add tool result to history
			c.history = append(c.history, api.Message{
				Role:    "tool",
				Content: result,
			})
		}

		// Loop continues: LLM will see tool results and generate final response
	}

	// If we hit max iterations, return the last response
	return "I apologize, but I couldn't complete the task within the allowed time.", nil
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
