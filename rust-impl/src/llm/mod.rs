//! LLM client module for interacting with language models.
//!
//! Uses RIG with Ollama provider for local LLM inference and supports
//! agentic tool calling for weather and web search capabilities.

mod client;
mod weather;
mod websearch;

pub use client::LlmClient;
pub use weather::WeatherTool;
pub use websearch::WebSearchTool;
