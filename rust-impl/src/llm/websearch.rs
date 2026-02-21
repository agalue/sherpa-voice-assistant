//! Web search tool with SearXNG and DuckDuckGo support.
//!
//! Provides web search capabilities with two backends:
//! - SearXNG: Privacy-respecting local metasearch engine (preferred)
//! - DuckDuckGo: Fallback for convenience when SearXNG unavailable

use regex::Regex;
use reqwest::Client;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::OnceLock;
use tracing::{debug, info, warn};

/// SearXNG search result.
#[derive(Debug, Deserialize)]
struct SearxngResult {
    title: String,
    #[serde(default)]
    content: String,
}

/// SearXNG API response.
#[derive(Debug, Deserialize)]
struct SearxngResponse {
    results: Vec<SearxngResult>,
}

/// Search tool error type.
#[derive(Debug, thiserror::Error)]
pub enum SearchError {
    #[error("Failed to perform web search: {0}")]
    RequestFailed(String),
    #[error("Failed to parse search results: {0}")]
    ParseFailed(String),
}

/// Arguments for the search tool.
#[derive(Deserialize)]
pub struct SearchArgs {
    query: String,
}

/// Web search tool with SearXNG and DuckDuckGo backends.
///
/// Uses SearXNG if URL is provided, otherwise falls back to DuckDuckGo.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WebSearchTool {
    searxng_url: Option<String>,
    #[serde(skip)]
    client: Client,
}

impl WebSearchTool {
    /// Create a new web search tool.
    ///
    /// # Arguments
    /// * `searxng_url` - Optional base URL of SearXNG instance. If None/empty, uses DuckDuckGo.
    ///
    /// # Returns
    /// A new `WebSearchTool` instance.
    pub fn new(searxng_url: Option<String>) -> Self {
        let client = Client::builder().timeout(std::time::Duration::from_secs(10)).build().expect("Failed to build HTTP client");
        Self { searxng_url, client }
    }

    /// Search using SearXNG instance.
    ///
    /// # Arguments
    /// * `query` - Search query string
    ///
    /// # Returns
    /// Formatted search results string.
    ///
    /// # Errors
    /// Returns `SearchError` if the request fails or results cannot be parsed.
    async fn search_searxng(&self, query: &str) -> Result<String, SearchError> {
        let searxng_url = match &self.searxng_url {
            Some(url) => url,
            None => return Err(SearchError::RequestFailed("SearXNG URL not configured".to_string())),
        };
        info!("Using SearXNG at {} for query: '{}'", searxng_url, query);

        let url = format!("{}/search?q={}&format=json&categories=general", searxng_url, urlencoding::encode(query));
        debug!("SearXNG request URL: {}", url);

        let response = self.client.get(&url).header("Accept", "application/json").send().await.map_err(|e| {
            info!("SearXNG request failed: {}", e);
            SearchError::RequestFailed(format!("SearXNG request failed: {}", e))
        })?;

        let status = response.status();
        if !status.is_success() {
            info!("SearXNG returned status: {}", status);
            return Err(SearchError::RequestFailed(format!("SearXNG returned HTTP {}", status)));
        }

        let response_text = response.text().await.map_err(|e| {
            info!("Failed to read SearXNG response: {}", e);
            SearchError::RequestFailed(format!("Failed to read SearXNG response: {}", e))
        })?;

        debug!("SearXNG response ({} bytes): {}", response_text.len(), &response_text[..response_text.len().min(200)]);

        let parsed: SearxngResponse = serde_json::from_str(&response_text).map_err(|e| {
            info!("SearXNG JSON parse error: {}. Response start: {}", e, &response_text[..response_text.len().min(500)]);
            SearchError::ParseFailed(format!("SearXNG JSON parse error: {}", e))
        })?;

        info!("SearXNG returned {} results", parsed.results.len());

        if parsed.results.is_empty() {
            return Ok("No search results found.".to_string());
        }

        self.format_results(&parsed.results.iter().map(|r| (r.title.as_str(), r.content.as_str())).collect::<Vec<_>>())
    }

    /// Search using DuckDuckGo HTML (fallback when SearXNG unavailable).
    ///
    /// # Arguments
    /// * `query` - Search query string
    ///
    /// # Returns
    /// Formatted search results string.
    ///
    /// # Errors
    /// Returns `SearchError` if the request fails or results cannot be parsed.
    async fn search_duckduckgo(&self, query: &str) -> Result<String, SearchError> {
        info!("Using DuckDuckGo for query: '{}'", query);

        let url = format!("https://html.duckduckgo.com/html/?q={}", urlencoding::encode(query));
        debug!("DuckDuckGo URL: {}", url);

        // DuckDuckGo requires a browser-like user agent to avoid blocking
        let response = self
            .client
            .get(&url)
            .header("User-Agent", "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
            .send()
            .await
            .map_err(|e| {
                info!("DuckDuckGo request failed: {}", e);
                SearchError::RequestFailed(format!("DuckDuckGo request failed: {}", e))
            })?;

        let status = response.status();
        if !status.is_success() {
            info!("DuckDuckGo returned status: {}", status);
            return Err(SearchError::RequestFailed(format!("DuckDuckGo returned HTTP {}", status)));
        }

        let html = response.text().await.map_err(|e| {
            info!("Failed to read DuckDuckGo response: {}", e);
            SearchError::RequestFailed(format!("Failed to read DuckDuckGo response: {}", e))
        })?;

        debug!("DuckDuckGo HTML response length: {} bytes", html.len());

        // Simple HTML parsing for results (fragile but works for basic use)
        let results = self.parse_duckduckgo_html(&html)?;

        if results.is_empty() {
            info!("DuckDuckGo parsing found no results");
            return Ok("No search results found. Try rephrasing your query.".to_string());
        }

        info!("DuckDuckGo found {} results", results.len());
        self.format_results(&results.iter().map(|(t, c)| (t.as_str(), c.as_str())).collect::<Vec<_>>())
    }

    /// Parse DuckDuckGo HTML response using regex for robust extraction.
    ///
    /// # Arguments
    /// * `html` - HTML response text
    ///
    /// # Returns
    /// Vector of (title, snippet) tuples.
    ///
    /// # Errors
    /// Returns `SearchError` if parsing fails.
    fn parse_duckduckgo_html(&self, html: &str) -> Result<Vec<(String, String)>, SearchError> {
        let mut results = Vec::new();

        debug!("Parsing DuckDuckGo HTML ({} bytes)", html.len());

        // Check for anti-bot page or errors
        if html.contains("blocked") || html.contains("captcha") {
            info!("DuckDuckGo may be blocking requests (captcha/blocked detected)");
        }

        // Extract result links with regex: <a class="result__a" href="...">Title</a>
        let link_regex = link_regex().ok_or_else(|| SearchError::ParseFailed("Failed to initialize link regex pattern".to_string()))?;
        let link_matches: Vec<_> = link_regex.captures_iter(html).take(5).collect(); // Get top 5 to ensure we have 3 good ones

        // Extract snippets: <a class="result__snippet">...</a>
        let snippet_regex = snippet_regex().ok_or_else(|| SearchError::ParseFailed("Failed to initialize snippet regex pattern".to_string()))?;
        let snippet_matches: Vec<_> = snippet_regex.captures_iter(html).take(5).collect();

        if link_matches.is_empty() {
            info!("No result links found in DuckDuckGo HTML");
            return Ok(results);
        }

        debug!("Found {} link matches and {} snippet matches", link_matches.len(), snippet_matches.len());

        // Take top 3 results
        let max_results = 3.min(link_matches.len());

        for i in 0..max_results {
            let caps = &link_matches[i];

            // Decode DuckDuckGo redirect URL to get actual destination
            let _url = decode_ddg_redirect_url(&caps[1]); // URL not used in voice output, but decoded for accuracy

            // Extract and clean title
            let title = strip_html_tags(&caps[2]).trim().to_string();

            // Extract snippet if available
            let snippet = if i < snippet_matches.len() {
                strip_html_tags(&snippet_matches[i][1]).trim().to_string()
            } else {
                String::new()
            };

            if !title.is_empty() {
                results.push((html_unescape(&title), html_unescape(&snippet)));
            }
        }

        info!("Parsed {} results from DuckDuckGo", results.len());
        Ok(results)
    }

    /// Format search results for voice output.
    /// Limited to 3 results with 200 char snippets for better voice output and token efficiency.
    ///
    /// # Arguments
    /// * `results` - Vector of (title, content) references
    ///
    /// # Returns
    /// Formatted results string.
    fn format_results(&self, results: &[(&str, &str)]) -> Result<String, SearchError> {
        let mut output = String::new();
        let max_results = 3.min(results.len());

        for (title, content) in results.iter().take(max_results) {
            // More direct: just title and content without numbering/preamble
            output.push_str(&format!("{}.  {}. ", title, content.chars().take(200).collect::<String>()));
        }

        Ok(output.trim().to_string())
    }

    /// Perform a web search.
    ///
    /// # Arguments
    /// * `query` - Search query string
    ///
    /// # Returns
    /// Formatted search results string.
    ///
    /// # Errors
    /// Returns `SearchError` if the request fails or results cannot be parsed.
    async fn search(&self, query: &str) -> Result<String, SearchError> {
        if self.searxng_url.is_some() {
            self.search_searxng(query).await
        } else {
            self.search_duckduckgo(query).await
        }
    }
}

/// Returns the lazily-initialized regex for DuckDuckGo result link anchors.
fn link_regex() -> Option<&'static Regex> {
    static LINK_REGEX: OnceLock<Option<Regex>> = OnceLock::new();
    LINK_REGEX
        .get_or_init(|| Regex::new(r#"<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)</a>"#).ok())
        .as_ref()
}

/// Returns the lazily-initialized regex for DuckDuckGo result snippet anchors.
fn snippet_regex() -> Option<&'static Regex> {
    static SNIPPET_REGEX: OnceLock<Option<Regex>> = OnceLock::new();
    SNIPPET_REGEX.get_or_init(|| Regex::new(r#"<a class="result__snippet[^"]*"[^>]*>([\s\S]*?)</a>"#).ok()).as_ref()
}

/// Returns the lazily-initialized regex for stripping HTML tags.
fn tag_regex() -> Option<&'static Regex> {
    static TAG_REGEX: OnceLock<Option<Regex>> = OnceLock::new();
    TAG_REGEX.get_or_init(|| Regex::new(r"<[^>]+>").ok()).as_ref()
}

/// Decode DuckDuckGo redirect URLs to extract actual destination.
/// DuckDuckGo wraps URLs like: //duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com&rut=...
fn decode_ddg_redirect_url(raw_url: &str) -> String {
    // Look for uddg= parameter which contains the actual URL
    if let Some(idx) = raw_url.find("uddg=") {
        let encoded = &raw_url[idx + 5..];
        // Split on & to get just the encoded URL part
        let encoded = if let Some(amp_idx) = encoded.find('&') { &encoded[..amp_idx] } else { encoded };
        // Decode the URL
        if let Ok(decoded) = urlencoding::decode(encoded) {
            return decoded.into_owned();
        }
    }
    raw_url.to_string()
}

/// Strip HTML tags from text using a lazily-initialized static regex.
fn strip_html_tags(text: &str) -> String {
    match tag_regex() {
        Some(re) => re.replace_all(text, "").to_string(),
        None => {
            warn!("HTML tag regex unavailable; returning text unstripped");
            text.to_string()
        }
    }
}

/// Unescape basic HTML entities.
fn html_unescape(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
}

impl Tool for WebSearchTool {
    const NAME: &'static str = "search_web";
    type Error = SearchError;
    type Args = SearchArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "search_web".to_string(),
            description: "Search the web for current information, news, events, facts you don't know. ALWAYS use this tool when you lack information about current events, recent news, or real-time data. Returns top search results.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (be specific, use keywords)"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        info!("🔍 Searching web for: {}", args.query);
        self.search(&args.query).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_html_tags() {
        let html = "<b>Hello</b> <i>World</i>";
        assert_eq!(strip_html_tags(html), "Hello World");
    }

    #[test]
    fn test_strip_html_tags_nested() {
        let html = "<div><span>Test</span></div>";
        assert_eq!(strip_html_tags(html), "Test");
    }

    #[test]
    fn test_html_unescape() {
        assert_eq!(html_unescape("Hello &amp; World"), "Hello & World");
        assert_eq!(html_unescape("&lt;test&gt;"), "<test>");
        assert_eq!(html_unescape("&quot;quoted&quot;"), "\"quoted\"");
        assert_eq!(html_unescape("&#x27;apostrophe&#39;"), "'apostrophe'");
    }

    #[test]
    fn test_decode_ddg_redirect_url() {
        // With uddg parameter and trailing params
        let url1 = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpath%3Fa%3D1&rut=test";
        assert_eq!(decode_ddg_redirect_url(url1), "https://example.com/path?a=1");

        // Without trailing parameters
        let url2 = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com";
        assert_eq!(decode_ddg_redirect_url(url2), "https://example.com");

        // Non-redirect URL (should return as-is)
        let url3 = "https://example.com/direct";
        assert_eq!(decode_ddg_redirect_url(url3), "https://example.com/direct");

        // Empty string
        assert_eq!(decode_ddg_redirect_url(""), "");
    }

    #[test]
    fn test_parse_duckduckgo_html_empty() {
        let tool = WebSearchTool::new(None);
        let html = "<html>No results here</html>";
        let results = tool.parse_duckduckgo_html(html).expect("Failed to parse");
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_parse_duckduckgo_html_with_data() {
        let tool = WebSearchTool::new(None);
        // Real DuckDuckGo HTML structure
        let html = r#"
            <div class="result__body">
                <h2 class="result__title">
                    <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgo.dev%2F&amp;rut=76be0d5c">The Go Programming Language</a>
                </h2>
                <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgo.dev%2F&amp;rut=76be0d5c">Go is an open source programming language that makes it simple to build secure, scalable systems.</a>
            </div>
        "#;
        let results = tool.parse_duckduckgo_html(html).expect("Failed to parse");
        assert_eq!(results.len(), 1);
        assert!(results[0].0.contains("The Go Programming Language"));
        assert!(results[0].1.contains("open source programming language"));
    }

    #[test]
    fn test_parse_duckduckgo_html_decodes_redirect_url() {
        let tool = WebSearchTool::new(None);
        // Real DuckDuckGo redirect URL structure
        let html = r#"
            <div class="result__body">
                <h2 class="result__title">
                    <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpath%3Fquery%3D1&amp;rut=test123">Example Site</a>
                </h2>
                <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpath%3Fquery%3D1&amp;rut=test123">Example description with content</a>
            </div>
        "#;
        let results = tool.parse_duckduckgo_html(html).expect("Failed to parse");
        assert_eq!(results.len(), 1);
        assert!(results[0].0.contains("Example Site"));

        // Verify URL decoding works
        let decoded = decode_ddg_redirect_url("//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpath%3Fquery%3D1&rut=test123");
        assert!(decoded.contains("https://example.com/path?query=1"));
    }

    #[test]
    fn test_parse_duckduckgo_html_max_results() {
        let tool = WebSearchTool::new(None);
        // Create HTML with 5 results using real DuckDuckGo structure
        let html = r#"
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
        "#;
        let results = tool.parse_duckduckgo_html(html).expect("Failed to parse");
        // Should limit to 3 results
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_parse_duckduckgo_html_strips_html_in_snippet() {
        let tool = WebSearchTool::new(None);
        // Real DuckDuckGo sometimes includes bold tags in snippets
        let html = r#"
            <div class="result__body">
                <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com">Example Title</a>
                <a class="result__snippet">This is a <b>bold</b> snippet with <i>italic</i> text</a>
            </div>
        "#;
        let results = tool.parse_duckduckgo_html(html).expect("Failed to parse");
        assert_eq!(results.len(), 1);
        assert!(!results[0].1.contains("<b>"));
        assert!(!results[0].1.contains("</b>"));
        assert!(results[0].1.contains("bold snippet"));
    }

    #[test]
    fn test_format_results_truncates_long_content() {
        let tool = WebSearchTool::new(None);
        // Create content with 300 characters (should be truncated to 200)
        let long_content = "a".repeat(300);
        let results = vec![("Test", long_content.as_str())];
        let output = tool.format_results(&results).expect("Failed to format");
        // The formatted output should not contain all 300 characters
        // (200 char limit + title + formatting should be less than 300)
        assert!(output.len() < 250);
    }
}
