package llm

import (
	"strings"
	"testing"
)

func TestStripHTML(t *testing.T) {
	html := "<b>Hello</b> <i>World</i>"
	expected := "Hello World"
	result := stripHTML(html)
	if result != expected {
		t.Errorf("stripHTML() = %q, want %q", result, expected)
	}
}

func TestStripHTMLNested(t *testing.T) {
	html := "<div><span>Test</span></div>"
	expected := "Test"
	result := stripHTML(html)
	if result != expected {
		t.Errorf("stripHTML() = %q, want %q", result, expected)
	}
}

func TestHTMLUnescape(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"Hello &amp; World", "Hello & World"},
		{"&lt;test&gt;", "<test>"},
		{"&quot;quoted&quot;", "\"quoted\""},
		{"&#x27;apostrophe&#39;", "'apostrophe'"},
	}

	for _, tt := range tests {
		result := htmlUnescape(tt.input)
		if result != tt.expected {
			t.Errorf("htmlUnescape(%q) = %q, want %q", tt.input, result, tt.expected)
		}
	}
}

func TestDecodeDDGRedirectURL(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "redirect URL with uddg parameter",
			input:    "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpath%3Fa%3D1&rut=test",
			expected: "https://example.com/path?a=1",
		},
		{
			name:     "redirect URL without trailing parameters",
			input:    "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com",
			expected: "https://example.com",
		},
		{
			name:     "non-redirect URL",
			input:    "https://example.com/direct",
			expected: "https://example.com/direct",
		},
		{
			name:     "empty string",
			input:    "",
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := decodeDDGRedirectURL(tt.input)
			if result != tt.expected {
				t.Errorf("decodeDDGRedirectURL(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

func TestParseDuckDuckGoHTMLEmpty(t *testing.T) {
	html := "<html>No results here</html>"
	results := ParseDuckDuckGoHTML(html)
	if len(results) != 0 {
		t.Errorf("ParseDuckDuckGoHTML() returned %d results, want 0", len(results))
	}
}

func TestParseDuckDuckGoHTMLWithData(t *testing.T) {
	// Real DuckDuckGo HTML structure
	html := `
		<div class="result__body">
			<h2 class="result__title">
				<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgo.dev%2F&amp;rut=76be0d5c">The Go Programming Language</a>
			</h2>
			<a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgo.dev%2F&amp;rut=76be0d5c">Go is an open source programming language that makes it simple to build secure, scalable systems.</a>
		</div>
	`
	results := ParseDuckDuckGoHTML(html)
	if len(results) == 0 {
		t.Fatal("ParseDuckDuckGoHTML() returned no results")
	}
	if !strings.Contains(results[0].Title, "The Go Programming Language") {
		t.Errorf("Title = %q, want to contain 'The Go Programming Language'", results[0].Title)
	}
	if !strings.Contains(results[0].Content, "open source programming language") {
		t.Errorf("Content = %q, want to contain 'open source programming language'", results[0].Content)
	}
}

func TestParseDuckDuckGoHTMLDecodesRedirectURL(t *testing.T) {
	// Real DuckDuckGo redirect URL structure
	html := `
		<div class="result__body">
			<h2 class="result__title">
				<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpath%3Fquery%3D1&amp;rut=test123">Example Site</a>
			</h2>
			<a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpath%3Fquery%3D1&amp;rut=test123">Example description with content</a>
		</div>
	`
	results := ParseDuckDuckGoHTML(html)
	if len(results) == 0 {
		t.Fatal("ParseDuckDuckGoHTML() returned no results")
	}
	if !strings.Contains(results[0].Title, "Example Site") {
		t.Errorf("Title = %q, want to contain 'Example Site'", results[0].Title)
	}
	// URL should be decoded internally (even though not returned in voice output)
	decodedURL := decodeDDGRedirectURL("//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpath%3Fquery%3D1&rut=test123")
	if !strings.Contains(decodedURL, "https://example.com/path?query=1") {
		t.Errorf("decodeDDGRedirectURL() = %q, want to contain 'https://example.com/path?query=1'", decodedURL)
	}
}

func TestParseDuckDuckGoHTMLMaxResults(t *testing.T) {
	// Create HTML with 5 results using real DuckDuckGo structure
	html := `
		<div class="result__body">
			<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample1.com">Title 1</a>
			<a class="result__snippet">Snippet 1</a>
		</div>
		<div class="result__body">
			<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample2.com">Title 2</a>
			<a class="result__snippet">Snippet 2</a>
		</div>
		<div class="result__body">
			<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample3.com">Title 3</a>
			<a class="result__snippet">Snippet 3</a>
		</div>
		<div class="result__body">
			<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample4.com">Title 4</a>
			<a class="result__snippet">Snippet 4</a>
		</div>
		<div class="result__body">
			<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample5.com">Title 5</a>
			<a class="result__snippet">Snippet 5</a>
		</div>
	`
	results := ParseDuckDuckGoHTML(html)
	// Should limit to 3 results
	if len(results) != 3 {
		t.Errorf("ParseDuckDuckGoHTML() returned %d results, want 3", len(results))
	}
}

func TestParseDuckDuckGoHTMLStripsHTMLInSnippet(t *testing.T) {
	// Real DuckDuckGo sometimes includes bold tags in snippets
	html := `
		<div class="result__body">
			<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com">Example Title</a>
			<a class="result__snippet">This is a <b>bold</b> snippet with <i>italic</i> text</a>
		</div>
	`
	results := ParseDuckDuckGoHTML(html)
	if len(results) == 0 {
		t.Fatal("ParseDuckDuckGoHTML() returned no results")
	}
	if strings.Contains(results[0].Content, "<b>") || strings.Contains(results[0].Content, "</b>") {
		t.Errorf("Content still contains HTML tags: %q", results[0].Content)
	}
	if !strings.Contains(results[0].Content, "bold snippet") {
		t.Errorf("Content = %q, want to contain 'bold snippet'", results[0].Content)
	}
}

func TestFormatResultsTruncatesLongContent(t *testing.T) {
	// Create content with 300 characters (should be truncated to 200)
	longContent := strings.Repeat("a", 300)
	results := []SearxngResult{
		{Title: "Test", Content: longContent},
	}
	output := FormatResults(results)

	// The formatted output should not contain all 300 characters
	// (200 char limit + title + formatting should be less than 300)
	if len(output) > 250 {
		t.Errorf("FormatResults() did not truncate long content, length = %d", len(output))
	}
}
