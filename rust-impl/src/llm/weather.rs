//! Weather tool for retrieving current weather information using Open-Meteo API.
//!
//! Supports both city-based queries and IP-based geolocation for automatic location detection.

use reqwest::{Client, header::USER_AGENT};
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::info;

/// IP geolocation response from ip-api.com.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct IPGeolocation {
    country_code: String,
    region: String,
    city: String,
    lat: f64,
    lon: f64,
}

/// Geolocation response from OpenStreetMap Nominatim API.
#[derive(Debug, Deserialize)]
struct Geolocation {
    display_name: String,
    lat: String,
    lon: String,
}

/// Weather API response from Open-Meteo.
#[derive(Debug, Deserialize)]
struct WeatherResponse {
    current: CurrentWeather,
    current_units: CurrentWeatherUnits,
}

/// Current weather data.
#[derive(Debug, Deserialize)]
struct CurrentWeather {
    temperature_2m: f64,
    apparent_temperature: f64,
    relative_humidity_2m: f64,
    rain: f64,
    snowfall: f64,
}

/// Units for current weather data.
#[derive(Debug, Deserialize)]
struct CurrentWeatherUnits {
    temperature_2m: String,
    apparent_temperature: String,
    relative_humidity_2m: String,
    rain: String,
    snowfall: String,
}

impl WeatherResponse {
    /// Format weather data with location for speech output.
    ///
    /// # Arguments
    /// * `location` - Location name
    ///
    /// # Returns
    /// Formatted weather string suitable for TTS.
    fn format_with_location(&self, location: &str) -> String {
        let mut output = format!("Weather for {}: ", location);

        // Format for TTS: spell out units like % as "percent" for better voice synthesis
        output.push_str(&format!(
            "Temperature is {}{}, feels like {}{}. ",
            self.current.temperature_2m, self.current_units.temperature_2m, self.current.apparent_temperature, self.current_units.apparent_temperature
        ));

        // Convert % to "percent" for TTS (with space for better pronunciation)
        let humidity_unit = if self.current_units.relative_humidity_2m == "%" {
            " percent"
        } else {
            &self.current_units.relative_humidity_2m
        };
        output.push_str(&format!("Humidity is {}{}. ", self.current.relative_humidity_2m, humidity_unit));

        if self.current.rain > 0.0 {
            output.push_str(&format!("Rain: {}{}. ", self.current.rain, self.current_units.rain));
        }

        if self.current.snowfall > 0.0 {
            output.push_str(&format!("Snowfall: {}{}. ", self.current.snowfall, self.current_units.snowfall));
        }

        output
    }
}

/// Weather tool error type.
#[derive(Debug, thiserror::Error)]
#[error("Weather API error")]
pub struct WeatherError;

/// Arguments for the weather tool.
#[derive(Deserialize)]
pub struct WeatherArgs {
    city: Option<String>,
}

/// Weather tool for retrieving current weather information.
///
/// Uses Open-Meteo API for weather data, OpenStreetMap for geocoding,
/// and ip-api.com for IP-based geolocation when no city is specified.
#[derive(Deserialize, Serialize)]
pub struct WeatherTool;

impl WeatherTool {
    /// Get current public IP address.
    ///
    /// # Returns
    /// IP address string.
    ///
    /// # Errors
    /// Returns `WeatherError` if request fails.
    async fn get_current_ip(&self) -> Result<String, WeatherError> {
        let cli = Client::builder().timeout(std::time::Duration::from_secs(5)).build().map_err(|e| {
            info!("Failed to build HTTP client: {}", e);
            WeatherError
        })?;
        cli.get("https://ifconfig.me/ip")
            .send()
            .await
            .map_err(|e| {
                info!("Failed to get IP address: {}", e);
                WeatherError
            })?
            .text()
            .await
            .map_err(|e| {
                info!("Failed to read IP response: {}", e);
                WeatherError
            })
    }

    /// Get coordinates from IP address using geolocation API.
    ///
    /// # Arguments
    /// * `ipaddr` - IP address
    ///
    /// # Returns
    /// Tuple of (latitude, longitude, location_name).
    ///
    /// # Errors
    /// Returns `WeatherError` if request fails.
    async fn get_coords_from_ip(&self, ipaddr: &str) -> Result<(f64, f64, String), WeatherError> {
        let cli = Client::builder().timeout(std::time::Duration::from_secs(5)).build().map_err(|e| {
            info!("Failed to build HTTP client: {}", e);
            WeatherError
        })?;
        // Note: ip-api.com free tier doesn't support HTTPS
        let ipgeo = cli
            .get(format!("http://ip-api.com/json/{}", ipaddr))
            .send()
            .await
            .map_err(|e| {
                info!("IP geolocation request failed: {}", e);
                WeatherError
            })?
            .json::<IPGeolocation>()
            .await
            .map_err(|e| {
                info!("Failed to parse IP geolocation response: {}", e);
                WeatherError
            })?;

        let location = format!("{}, {}, {}", ipgeo.city, ipgeo.region, ipgeo.country_code);
        Ok((ipgeo.lat, ipgeo.lon, location))
    }

    /// Get coordinates from city name using OpenStreetMap Nominatim API.
    ///
    /// # Arguments
    /// * `city` - City name
    ///
    /// # Returns
    /// Tuple of (latitude, longitude, location_name).
    ///
    /// # Errors
    /// Returns `WeatherError` if request fails or city not found.
    async fn get_coords_from_city(&self, city: &str) -> Result<(f64, f64, String), WeatherError> {
        if city.is_empty() {
            return Err(WeatherError);
        }
        let cli = Client::builder().timeout(std::time::Duration::from_secs(5)).build().map_err(|e| {
            info!("Failed to build HTTP client: {}", e);
            WeatherError
        })?;
        // URL encode city name for special characters
        let encoded_city = urlencoding::encode(city);
        let response = cli
            .get(format!("https://nominatim.openstreetmap.org/search?q={}&format=json&limit=1", encoded_city))
            .header(USER_AGENT, "Mozilla/5.0 (compatible; VoiceAssistant/1.0)")
            .send()
            .await
            .map_err(|e| {
                info!("Nominatim geocoding request failed for '{}': {}", city, e);
                WeatherError
            })?
            .json::<Vec<Geolocation>>()
            .await
            .map_err(|e| {
                info!("Failed to parse Nominatim response: {}", e);
                WeatherError
            })?;

        if response.is_empty() {
            Err(WeatherError)
        } else {
            Ok((
                response[0].lat.parse().map_err(|_| WeatherError)?,
                response[0].lon.parse().map_err(|_| WeatherError)?,
                response[0].display_name.clone(),
            ))
        }
    }

    /// Get weather data from Open-Meteo API.
    ///
    /// # Arguments
    /// * `lat` - Latitude
    /// * `lon` - Longitude
    /// * `location` - Location name for display
    ///
    /// # Returns
    /// Formatted weather string.
    ///
    /// # Errors
    /// Returns `WeatherError` if request fails.
    async fn get_weather_data(&self, lat: f64, lon: f64, location: &str) -> Result<String, WeatherError> {
        let cli = Client::builder().timeout(std::time::Duration::from_secs(5)).build().map_err(|e| {
            info!("Failed to build HTTP client: {}", e);
            WeatherError
        })?;
        let weather = cli
            .get(format!(
                "https://api.open-meteo.com/v1/forecast?latitude={}&longitude={}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,showers,snowfall",
                lat, lon
            ))
            .header(USER_AGENT, "Mozilla/5.0 (compatible; VoiceAssistant/1.0)")
            .send()
            .await
            .map_err(|e| {
                info!("Weather API request failed: {}", e);
                WeatherError
            })?
            .json::<WeatherResponse>()
            .await
            .map_err(|e| {
                info!("Failed to parse weather response: {}", e);
                WeatherError
            })?;

        Ok(weather.format_with_location(location))
    }
}

impl Tool for WeatherTool {
    const NAME: &'static str = "get_weather";
    type Error = WeatherError;
    type Args = WeatherArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get current weather for any location. Use this when user asks about weather, temperature, or climate. Leave city empty for user's current location via IP geolocation.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name. Leave empty for IP-based current location."
                    }
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        info!("🌤️  Fetching weather data...");

        let (lat, lon, location) = if let Some(city_name) = args.city {
            // Treat "current", empty, or similar phrases as request for IP-based location
            let city_lower = city_name.to_lowercase();
            if city_lower.is_empty() || city_lower == "current" || city_lower == "here" || city_lower == "my location" || city_lower.contains("current location") {
                let ipaddr = self.get_current_ip().await?;
                self.get_coords_from_ip(&ipaddr).await?
            } else {
                self.get_coords_from_city(&city_name).await?
            }
        } else {
            let ipaddr = self.get_current_ip().await?;
            self.get_coords_from_ip(&ipaddr).await?
        };

        self.get_weather_data(lat, lon, &location).await
    }
}
