//! LLM client using RIG with Ollama provider.

use anyhow::{Context, Result};
use rig::agent::Agent;
use rig::client::{CompletionClient, Nothing};
use rig::message::Message;
use rig::providers::ollama;
use serde_json::json;
use tracing::{debug, info};

use crate::config::AppConfig;

/// LLM client for conversation with Ollama.
/// Uses RIG framework for simplified LLM interactions.
pub struct LlmClient {
    agent: Agent<ollama::CompletionModel>, // RIG agent with Ollama backend
    history: Vec<Message>,                 // Conversation history
    max_history: usize,                    // Maximum history length
}

impl LlmClient {
    /// Create a new LLM client.
    ///
    /// # Arguments
    /// * `config` - Application configuration
    ///
    /// # Returns
    /// A new `LlmClient` instance.
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

        // Reduce context window and token limits to save GPU memory on
        // resource-constrained devices running STT/TTS models alongside LLM
        let agent = client
            .agent(&config.ollama_model)
            .preamble(&config.system_prompt)
            .temperature(0.7)
            .additional_params(json!({
                "num_ctx": 1024,
                "num_predict": 150
            }))
            .build();

        Ok(Self { agent, history: Vec::new(), max_history: config.max_history })
    }

    /// Send a message and get complete response.
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

        use rig::completion::Chat;

        // Get complete response using Chat trait
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
