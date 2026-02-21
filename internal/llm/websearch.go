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
	"regexp"
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

// ParseDuckDuckGoHTML parses DuckDuckGo HTML response using regex for robust extraction.
func ParseDuckDuckGoHTML(html string) []SearxngResult {
	var results []SearxngResult

	// Check for anti-bot page or errors
	if strings.Contains(html, "blocked") || strings.Contains(html, "captcha") {
		log.Println("DuckDuckGo may be blocking requests (captcha/blocked detected)")
	}

	// Extract result links with regex: <a class="result__a" href="...">Title</a>
	linkRegex := regexp.MustCompile(`<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)</a>`)
	linkMatches := linkRegex.FindAllStringSubmatch(html, 5) // Get top 5 to ensure we have 3 good ones

	// Extract snippets: <a class="result__snippet">...</a>
	snippetRegex := regexp.MustCompile(`<a class="result__snippet[^"]*"[^>]*>([\s\S]*?)</a>`)
	snippetMatches := snippetRegex.FindAllStringSubmatch(html, 5)

	if len(linkMatches) == 0 {
		log.Println("No result links found in DuckDuckGo HTML")
		return results
	}

	// Take top 3 results
	maxResults := 3
	if len(linkMatches) < maxResults {
		maxResults = len(linkMatches)
	}

	for i := 0; i < maxResults; i++ {
		match := linkMatches[i]
		if len(match) < 3 {
			continue
		}

		// Decode DuckDuckGo redirect URL to get actual destination
		_ = decodeDDGRedirectURL(match[1]) // URL not used in voice output, but decoded for accuracy

		// Extract and clean title
		title := strings.TrimSpace(stripHTML(match[2]))

		// Extract snippet if available
		snippet := ""
		if i < len(snippetMatches) && len(snippetMatches[i]) > 1 {
			snippet = strings.TrimSpace(stripHTML(snippetMatches[i][1]))
		}

		if title != "" {
			results = append(results, SearxngResult{
				Title:   htmlUnescape(title),
				Content: htmlUnescape(snippet),
			})
		}
	}

	log.Printf("Parsed %d results from DuckDuckGo", len(results))
	return results
}

// decodeDDGRedirectURL decodes DuckDuckGo redirect URLs to extract actual destination.
// DuckDuckGo wraps URLs like: https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com&rut=...
func decodeDDGRedirectURL(rawURL string) string {
	// Look for uddg= parameter which contains the actual URL
	if idx := strings.Index(rawURL, "uddg="); idx != -1 {
		encoded := rawURL[idx+5:]
		// Split on & to get just the encoded URL part
		if ampIdx := strings.Index(encoded, "&"); ampIdx != -1 {
			encoded = encoded[:ampIdx]
		}
		// Decode the URL
		if decoded, err := url.QueryUnescape(encoded); err == nil {
			return decoded
		}
	}
	return rawURL
}

// stripHTML removes HTML tags from text using regex.
func stripHTML(text string) string {
	tagRegex := regexp.MustCompile(`<[^>]+>`)
	return tagRegex.ReplaceAllString(text, "")
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
