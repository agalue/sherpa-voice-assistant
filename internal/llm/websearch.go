// Package llm provides LLM integration via Ollama API.
package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// SearchArgs is the arguments for the search tool.
type SearchArgs struct {
	Query string `json:"query"`
}

// SearxngResult represents a single search result from SearXNG.
type SearxngResult struct {
	Title   string `json:"title"`
	Content string `json:"content"`
}

// SearxngResponse is the response from SearXNG API.
type SearxngResponse struct {
	Results []SearxngResult `json:"results"`
}

// SearchSearxng performs a search using SearXNG instance.
func SearchSearxng(ctx context.Context, query, searxngURL string) (string, error) {
	log.Printf("Using SearXNG at %s for query: '%s'", searxngURL, query)

	client := &http.Client{Timeout: 10 * time.Second}
	encodedQuery := url.QueryEscape(query)
	searchURL := fmt.Sprintf("%s/search?q=%s&format=json&categories=general", searxngURL, encodedQuery)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, searchURL, nil)
	if err != nil {
		return "", fmt.Errorf("failed to create SearXNG request: %w", err)
	}
	req.Header.Set("Accept", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		log.Printf("SearXNG request failed: %v", err)
		return "", fmt.Errorf("SearXNG request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Printf("SearXNG returned status: %d", resp.StatusCode)
		return "", fmt.Errorf("SearXNG returned HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("Failed to read SearXNG response: %v", err)
		return "", fmt.Errorf("failed to read SearXNG response: %w", err)
	}

	var parsed SearxngResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		log.Printf("SearXNG JSON parse error: %v", err)
		return "", fmt.Errorf("SearXNG JSON parse error: %w", err)
	}

	log.Printf("SearXNG returned %d results", len(parsed.Results))

	if len(parsed.Results) == 0 {
		return "No search results found.", nil
	}

	return FormatResults(parsed.Results), nil
}

// SearchDuckDuckGo performs a search using DuckDuckGo HTML (fallback).
func SearchDuckDuckGo(ctx context.Context, query string) (string, error) {
	log.Printf("Using DuckDuckGo for query: '%s'", query)

	client := &http.Client{Timeout: 10 * time.Second}
	encodedQuery := url.QueryEscape(query)
	searchURL := fmt.Sprintf("https://html.duckduckgo.com/html/?q=%s", encodedQuery)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, searchURL, nil)
	if err != nil {
		return "", fmt.Errorf("failed to create DuckDuckGo request: %w", err)
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

	resp, err := client.Do(req)
	if err != nil {
		log.Printf("DuckDuckGo request failed: %v", err)
		return "", fmt.Errorf("DuckDuckGo request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Printf("DuckDuckGo returned status: %d", resp.StatusCode)
		return "", fmt.Errorf("DuckDuckGo returned HTTP %d", resp.StatusCode)
	}

	html, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("Failed to read DuckDuckGo response: %v", err)
		return "", fmt.Errorf("failed to read DuckDuckGo response: %w", err)
	}

	results := ParseDuckDuckGoHTML(string(html))

	if len(results) == 0 {
		log.Println("DuckDuckGo parsing found no results")
		return "No search results found. Try rephrasing your query.", nil
	}

	log.Printf("DuckDuckGo found %d results", len(results))
	return FormatResults(results), nil
}

// ParseDuckDuckGoHTML parses DuckDuckGo HTML response.
func ParseDuckDuckGoHTML(html string) []SearxngResult {
	var results []SearxngResult

	// Check for anti-bot page or errors
	if strings.Contains(html, "blocked") || strings.Contains(html, "captcha") {
		log.Println("DuckDuckGo may be blocking requests (captcha/blocked detected)")
	}

	// Simple parsing - look for result divs (take top 3)
	sections := strings.Split(html, `class="result"`)
	for i := 1; i < len(sections) && len(results) < 3; i++ {
		section := sections[i]

		// Extract title (between result__a tags)
		title := extractBetween(section, `class="result__a"`, "</a>")
		title = strings.TrimSpace(stripHTML(title))

		// Extract snippet (between result__snippet tags)
		snippet := extractBetween(section, `class="result__snippet"`, "</div>")
		snippet = strings.TrimSpace(stripHTML(snippet))

		if title != "" {
			results = append(results, SearxngResult{
				Title:   htmlUnescape(title),
				Content: htmlUnescape(snippet),
			})
		}
	}

	return results
}

// extractBetween extracts text between start and end markers.
func extractBetween(text, start, end string) string {
	startIdx := strings.Index(text, start)
	if startIdx == -1 {
		return ""
	}
	text = text[startIdx+len(start):]

	endIdx := strings.Index(text, end)
	if endIdx == -1 {
		return text
	}
	return text[:endIdx]
}

// stripHTML removes HTML tags from text (simple implementation).
func stripHTML(text string) string {
	var result strings.Builder
	inTag := false
	for _, c := range text {
		if c == '<' {
			inTag = true
		} else if c == '>' {
			inTag = false
		} else if !inTag {
			result.WriteRune(c)
		}
	}
	return result.String()
}

// htmlUnescape unescapes basic HTML entities.
func htmlUnescape(s string) string {
	s = strings.ReplaceAll(s, "&amp;", "&")
	s = strings.ReplaceAll(s, "&lt;", "<")
	s = strings.ReplaceAll(s, "&gt;", ">")
	s = strings.ReplaceAll(s, "&quot;", "\"")
	s = strings.ReplaceAll(s, "&#x27;", "'")
	s = strings.ReplaceAll(s, "&#39;", "'")
	return s
}

// FormatResults formats search results for voice output.
// Limited to 3 results with 200 char snippets for better voice output and token efficiency.
func FormatResults(results []SearxngResult) string {
	var output strings.Builder
	maxResults := 3
	if len(results) < maxResults {
		maxResults = len(results)
	}

	for i := 0; i < maxResults; i++ {
		result := results[i]
		// More direct: just title and content without numbering/preamble
		content := result.Content
		// Truncate by runes (Unicode code points), not bytes, to avoid breaking UTF-8 characters
		runes := []rune(content)
		if len(runes) > 200 {
			content = string(runes[:200])
		}
		output.WriteString(fmt.Sprintf("%s. %s. ", result.Title, content))
	}

	return strings.TrimSpace(output.String())
}

// ExecuteSearchTool executes the web search tool with the given JSON arguments.
func ExecuteSearchTool(ctx context.Context, jsonArgs string, searxngURL string) (string, error) {
	var args SearchArgs
	if err := json.Unmarshal([]byte(jsonArgs), &args); err != nil {
		return "", fmt.Errorf("failed to parse search tool arguments: %w", err)
	}

	log.Printf("🔍 Searching web for: %s", args.Query)

	if searxngURL != "" {
		return SearchSearxng(ctx, args.Query, searxngURL)
	}
	return SearchDuckDuckGo(ctx, args.Query)
}
