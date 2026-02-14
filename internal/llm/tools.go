// Package llm provides LLM integration via Ollama API.
package llm

import (
	"context"

	"github.com/ollama/ollama/api"
)

// ToolFunc is a function that executes a tool with JSON arguments.
type ToolFunc func(ctx context.Context, jsonArgs string) (string, error)

// ToolRegistry maps tool names to their execution functions.
type ToolRegistry map[string]ToolFunc

// GetToolDefinitions returns the Ollama tool definitions for weather and search.
func GetToolDefinitions() []api.Tool {
	// Create properties maps for tool parameters
	weatherProps := api.NewToolPropertiesMap()
	weatherProps.Set("city", api.ToolProperty{
		Type:        []string{"string"},
		Description: "City name. Leave empty for IP-based current location.",
	})

	searchProps := api.NewToolPropertiesMap()
	searchProps.Set("query", api.ToolProperty{
		Type:        []string{"string"},
		Description: "The search query (be specific, use keywords)",
	})

	return []api.Tool{
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "get_weather",
				Description: "Get current weather for any location. Use this when user asks about weather, temperature, or climate. Leave city empty for user's current location via IP geolocation.",
				Parameters: api.ToolFunctionParameters{
					Type:       "object",
					Properties: weatherProps,
				},
			},
		},
		{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "search_web",
				Description: "Search the web for current information, news, events, facts you don't know. ALWAYS use this tool when you lack information about current events, recent news, or real-time data. Returns top search results.",
				Parameters: api.ToolFunctionParameters{
					Type:       "object",
					Properties: searchProps,
					Required:   []string{"query"},
				},
			},
		},
	}
}

// CreateToolRegistry creates a tool registry with weather and search tools.
//
// Parameters:
//   - searxngURL: Optional SearXNG URL for web search (empty string uses DuckDuckGo)
//
// Returns:
//   - A ToolRegistry mapping tool names to execution functions
func CreateToolRegistry(searxngURL string) ToolRegistry {
	return ToolRegistry{
		"get_weather": func(ctx context.Context, jsonArgs string) (string, error) {
			return ExecuteWeatherTool(ctx, jsonArgs)
		},
		"search_web": func(ctx context.Context, jsonArgs string) (string, error) {
			return ExecuteSearchTool(ctx, jsonArgs, searxngURL)
		},
	}
}
