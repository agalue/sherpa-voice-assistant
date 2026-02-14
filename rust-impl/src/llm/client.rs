//! LLM client using RIG with Ollama provider and agentic tool calling.

use anyhow::{Context, Result};
use rig::agent::Agent;
use rig::client::{CompletionClient, Nothing};
use rig::completion::Chat;
use rig::message::Message;
use rig::providers::ollama;
use serde_json::json;
use tracing::{debug, info};

use super::{WeatherTool, WebSearchTool};
use crate::config::AppConfig;

/// LLM client for conversation with Ollama using agentic mode with tools.
/// Uses RIG framework with tool calling for weather and web search.
pub struct LlmClient {
    agent: Agent<ollama::CompletionModel>, // RIG agent with Ollama backend
    history: Vec<Message>,                 // Conversation history
    max_history: usize,                    // Maximum history length
}

impl LlmClient {
    /// Create a new LLM client with agentic tool support.
    ///
    /// # Arguments
    /// * `config` - Application configuration
    ///
    /// # Returns
    /// A new `LlmClient` instance with weather and search tools configured.
    ///
    /// # Errors
    /// Returns an error if failed to create Ollama client.
    pub fn new(config: &AppConfig) -> Result<Self> {
        info!("Connecting to Ollama at {}", config.ollama_url);
        info!("Using model: {}", config.ollama_model);

        let client = ollama::Client::builder()
            .api_key(Nothing)
            .base_url(&config.ollama_url)
            .build()
            .context("Failed to create Ollama client")?;

        // Build system prompt with tool usage instructions
        let mut system_prompt = config.system_prompt.clone();
        system_prompt.push_str(
            " CRITICAL: You have two tools available: get_weather and search_web. \
            When asked about current events, news, facts, sports results, or anything you don't know: \
            IMMEDIATELY use search_web tool - DO NOT say you lack information or capabilities. \
            For weather queries: use get_weather tool. Always use tools proactively.",
        );

        // Create search tool with optional SearXNG URL (fallback to DuckDuckGo if not provided)
        let search_tool = WebSearchTool::new(config.searxng_url.clone());
        if let Some(ref url) = config.searxng_url {
            info!("Web search: Using SearXNG at {}", url);
        } else {
            info!("Web search: Using DuckDuckGo (no SearXNG URL configured)");
        }

        // Reduce context window and token limits to save GPU memory on
        // resource-constrained devices running STT/TTS models alongside LLM
        let agent = client
            .agent(&config.ollama_model)
            .preamble(&system_prompt)
            .temperature(config.temperature as f64)
            .additional_params(json!({
                "num_ctx": 1024,
                "num_predict": 150
            }))
            .tool(WeatherTool)
            .tool(search_tool)
            .build();

        Ok(Self { agent, history: Vec::new(), max_history: config.max_history })
    }

    /// Send a message and get complete response (may invoke tools).
    /// This is simpler and faster than streaming when TTS is the bottleneck.
    ///
    /// # Arguments
    /// * `message` - The user's message
    ///
    /// # Returns
    /// The assistant's complete response.
    ///
    /// # Errors
    /// Returns an error if LLM request fails.
    pub async fn chat(&mut self, message: &str) -> Result<String> {
        debug!("User: {}", message);

        // Get complete response using Chat trait - RIG handles tool calling automatically
        let response = self.agent.chat(message, self.history.clone()).await.context("LLM request failed")?;

        debug!("Assistant: {}", response);

        // Update history
        self.history.push(Message::user(message));
        self.history.push(Message::assistant(&response));

        // Trim history if needed
        while self.history.len() > self.max_history * 2 {
            self.history.remove(0);
            if !self.history.is_empty() {
                self.history.remove(0);
            }
        }

        Ok(response)
    }
}
