//! Web search tool with SearXNG and DuckDuckGo support.
//!
//! Provides web search capabilities with two backends:
//! - SearXNG: Privacy-respecting local metasearch engine (preferred)
//! - DuckDuckGo: Fallback for convenience when SearXNG unavailable

use reqwest::Client;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{debug, info};

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
        Self { searxng_url: searxng_url.filter(|url| !url.is_empty()) }
    }

    /// Search using SearXNG instance.
    ///
    /// # Arguments
    /// * `query` - Search query string
    /// * `searxng_url` - Base URL of SearXNG instance
    ///
    /// # Returns
    /// Formatted search results string.
    ///
    /// # Errors
    /// Returns `SearchError` if the request fails or results cannot be parsed.
    async fn search_searxng(&self, query: &str, searxng_url: &str) -> Result<String, SearchError> {
        info!("Using SearXNG at {} for query: '{}'", searxng_url, query);
        // No custom user agent needed for our own SearXNG instance
        let cli = Client::builder().timeout(std::time::Duration::from_secs(10)).build().map_err(|e| {
            info!("Failed to build HTTP client: {}", e);
            SearchError::RequestFailed(format!("Failed to build HTTP client: {}", e))
        })?;

        let url = format!("{}/search?q={}&format=json&categories=general", searxng_url, urlencoding::encode(query));
        debug!("SearXNG request URL: {}", url);

        let response = cli.get(&url).header("Accept", "application/json").send().await.map_err(|e| {
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
        // Generic user agent without OS specifics (works on any platform)
        let cli = Client::builder()
            .user_agent("Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|e| {
                info!("Failed to build HTTP client: {}", e);
                SearchError::RequestFailed(format!("Failed to build HTTP client: {}", e))
            })?;

        let url = format!("https://html.duckduckgo.com/html/?q={}", urlencoding::encode(query));
        debug!("DuckDuckGo URL: {}", url);

        let response = cli.get(&url).send().await.map_err(|e| {
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

    /// Parse DuckDuckGo HTML response.
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

        // Count potential result divs
        let result_count = html.matches("class=\"result\"").count();
        debug!("Found {} result divs in HTML", result_count);

        // Simple regex-free parsing - look for result divs (take top 2)
        for (idx, section) in html.split("class=\"result\"").skip(1).take(2).enumerate() {
            // Extract title (between result__a tags)
            let title = section
                .split("class=\"result__a\"")
                .nth(1)
                .and_then(|s| s.split('>').nth(1))
                .and_then(|s| s.split('<').next())
                .map(html_unescape)
                .unwrap_or_default();

            // Extract snippet (between result__snippet tags)
            let snippet = section
                .split("class=\"result__snippet\"")
                .nth(1)
                .and_then(|s| s.split('>').nth(1))
                .and_then(|s| s.split('<').next())
                .map(html_unescape)
                .unwrap_or_default();

            debug!("DuckDuckGo result {}: title='{}', snippet_len={}", idx + 1, title, snippet.len());

            if !title.is_empty() {
                results.push((title, snippet));
            }
        }

        if results.is_empty() {
            debug!("No results parsed from DuckDuckGo HTML");
        }

        Ok(results)
    }

    /// Format search results for voice output.
    /// Limited to 2 results with 150 char snippets for better voice output and token efficiency.
    ///
    /// # Arguments
    /// * `results` - Vector of (title, content) references
    ///
    /// # Returns
    /// Formatted results string.
    fn format_results(&self, results: &[(&str, &str)]) -> Result<String, SearchError> {
        let mut output = String::new();
        let max_results = 2.min(results.len());

        for (title, content) in results.iter().take(max_results) {
            // More direct: just title and content without numbering/preamble
            output.push_str(&format!("{}.  {}. ", title, content.chars().take(150).collect::<String>()));
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
        if let Some(ref url) = self.searxng_url {
            self.search_searxng(query, url).await
        } else {
            self.search_duckduckgo(query).await
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
