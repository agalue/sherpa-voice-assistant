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
pub enum WeatherError {
    /// HTTP or network-level error when calling an external API.
    #[error("Weather API HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// Failed to parse the weather or geolocation API response.
    #[error("Weather API response parse error: {0}")]
    Parse(#[from] serde_json::Error),

    /// Generic weather tool error with a custom message.
    #[error("Weather API error: {0}")]
    Message(String),
}

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
pub struct WeatherTool {
    #[serde(skip)]
    client: Client,
}

impl WeatherTool {
    /// Create a new weather tool instance with a default HTTP client.
    pub fn new() -> Self {
        WeatherTool {
            client: Client::builder().timeout(std::time::Duration::from_secs(5)).build().expect("Failed to build HTTP client"),
        }
    }

    /// Get current public IP address.
    ///
    /// # Returns
    /// IP address string.
    ///
    /// # Errors
    /// Returns `WeatherError` if request fails.
    async fn get_current_ip(&self) -> Result<String, WeatherError> {
        let ip = self.client.get("https://ifconfig.me/ip").send().await?.text().await?;

        // Trim whitespace (ifconfig.me includes trailing newline)
        Ok(ip.trim().to_string())
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
        // Note: ip-api.com free tier doesn't support HTTPS
        let ipgeo = self.client.get(format!("http://ip-api.com/json/{}", ipaddr)).send().await?.json::<IPGeolocation>().await?;

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
            return Err(WeatherError::Message("City name is empty".to_string()));
        }
        // URL encode city name for special characters
        let encoded_city = urlencoding::encode(city);
        let response = self
            .client
            .get(format!("https://nominatim.openstreetmap.org/search?q={}&format=json&limit=1", encoded_city))
            .header(USER_AGENT, "Mozilla/5.0 (compatible; VoiceAssistant/1.0)")
            .send()
            .await?
            .json::<Vec<Geolocation>>()
            .await?;

        if response.is_empty() {
            Err(WeatherError::Message(format!("City not found: {}", city)))
        } else {
            Ok((
                response[0].lat.parse().map_err(|_| WeatherError::Message("Invalid latitude".to_string()))?,
                response[0].lon.parse().map_err(|_| WeatherError::Message("Invalid longitude".to_string()))?,
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
        let weather = self.client
            .get(format!(
                "https://api.open-meteo.com/v1/forecast?latitude={}&longitude={}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,showers,snowfall",
                lat, lon
            ))
            .header(USER_AGENT, "Mozilla/5.0 (compatible; VoiceAssistant/1.0)")
            .send()
            .await?
            .json::<WeatherResponse>()
            .await?;

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create test weather response
    fn create_weather_response(temp: f64, apparent_temp: f64, humidity: f64, rain: f64, snowfall: f64, humidity_unit: &str) -> WeatherResponse {
        WeatherResponse {
            current: CurrentWeather {
                temperature_2m: temp,
                apparent_temperature: apparent_temp,
                relative_humidity_2m: humidity,
                rain,
                snowfall,
            },
            current_units: CurrentWeatherUnits {
                temperature_2m: "°C".to_string(),
                apparent_temperature: "°C".to_string(),
                relative_humidity_2m: humidity_unit.to_string(),
                rain: "mm".to_string(),
                snowfall: "cm".to_string(),
            },
        }
    }

    #[test]
    fn test_weather_format_with_location() {
        let weather = create_weather_response(22.5, 21.0, 65.0, 0.0, 0.0, "%");
        let result = weather.format_with_location("San Francisco");

        // Verify location is included
        assert!(result.contains("San Francisco"), "Result should contain location");

        // Verify temperature is included
        assert!(result.contains("22.5"), "Result should contain temperature");

        // Verify apparent temperature is included
        assert!(result.contains("21"), "Result should contain apparent temperature");

        // Verify humidity is included
        assert!(result.contains("65"), "Result should contain humidity");

        // Verify rain and snowfall are NOT included (both are 0.0)
        assert!(!result.contains("Rain:"), "Result should not contain rain when 0.0");
        assert!(!result.contains("Snowfall:"), "Result should not contain snowfall when 0.0");
    }

    #[test]
    fn test_weather_unit_conversion() {
        let weather = create_weather_response(20.0, 19.0, 80.0, 0.0, 0.0, "%");
        let result = weather.format_with_location("London");

        // Verify % is converted to " percent" (with space for TTS)
        assert!(result.contains("80 percent"), "Result should contain '80 percent': {}", result);

        // Verify % symbol is not present
        assert!(!result.contains("80%"), "Result should not contain '80%'");
    }

    #[test]
    fn test_weather_conditional_fields() {
        // No precipitation
        let weather = create_weather_response(15.0, 14.0, 70.0, 0.0, 0.0, "%");
        let result = weather.format_with_location("New York");
        assert!(!result.contains("Rain:"), "Should not show rain when 0.0");
        assert!(!result.contains("Snowfall:"), "Should not show snowfall when 0.0");

        // Only rain
        let weather = create_weather_response(15.0, 14.0, 70.0, 5.2, 0.0, "%");
        let result = weather.format_with_location("New York");
        assert!(result.contains("Rain:"), "Should show rain when > 0.0");
        assert!(!result.contains("Snowfall:"), "Should not show snowfall when 0.0");

        // Only snow
        let weather = create_weather_response(15.0, 14.0, 70.0, 0.0, 3.1, "%");
        let result = weather.format_with_location("New York");
        assert!(!result.contains("Rain:"), "Should not show rain when 0.0");
        assert!(result.contains("Snowfall:"), "Should show snowfall when > 0.0");

        // Both
        let weather = create_weather_response(15.0, 14.0, 70.0, 2.5, 1.8, "%");
        let result = weather.format_with_location("New York");
        assert!(result.contains("Rain:"), "Should show rain when > 0.0");
        assert!(result.contains("Snowfall:"), "Should show snowfall when > 0.0");
    }

    #[test]
    fn test_weather_negative_temperature() {
        let weather = create_weather_response(-5.2, -8.0, 90.0, 0.0, 0.0, "%");
        let result = weather.format_with_location("Oslo");

        // Should handle negative temperature without panic
        assert!(!result.is_empty(), "Result should not be empty");
        assert!(result.contains("Oslo"), "Result should contain location");
        assert!(result.contains("-5.2") || result.contains("−5.2"), "Result should contain negative temperature");
    }

    #[test]
    fn test_weather_zero_temperature() {
        let weather = create_weather_response(0.0, -2.0, 100.0, 0.0, 5.0, "%");
        let result = weather.format_with_location("Reykjavik");

        assert!(!result.is_empty(), "Result should not be empty");
        assert!(result.contains("Reykjavik"), "Result should contain location");
        assert!(result.contains("Temperature"), "Result should contain temperature");
        assert!(result.contains("Snowfall:"), "Should show snowfall");
    }

    #[test]
    fn test_weather_high_humidity() {
        let weather = create_weather_response(25.0, 28.0, 100.0, 0.0, 0.0, "%");
        let result = weather.format_with_location("Miami");

        assert!(result.contains("100 percent"), "Should show 100 percent humidity");
    }

    #[test]
    fn test_weather_multibyte_location() {
        let locations = vec!["São Paulo", "Zürich", "北京", "Reykjavík", "Montréal"];

        for location in locations {
            let weather = create_weather_response(30.0, 32.0, 75.0, 0.0, 0.0, "%");
            let result = weather.format_with_location(location);

            // Should handle UTF-8 without panic
            assert!(!result.is_empty(), "Result should not be empty for UTF-8 location: {}", location);

            // Should contain the location
            assert!(result.contains(location), "Result should contain location '{}': {}", location, result);

            // Should still format weather data correctly
            assert!(result.contains("30"), "Result should contain temperature for {}", location);
        }
    }

    #[test]
    fn test_weather_different_humidity_unit() {
        // Test with "percent" instead of "%"
        let weather = WeatherResponse {
            current: CurrentWeather {
                temperature_2m: 20.0,
                apparent_temperature: 19.0,
                relative_humidity_2m: 60.0,
                rain: 0.0,
                snowfall: 0.0,
            },
            current_units: CurrentWeatherUnits {
                temperature_2m: "°C".to_string(),
                apparent_temperature: "°C".to_string(),
                relative_humidity_2m: "percent".to_string(),
                rain: "mm".to_string(),
                snowfall: "cm".to_string(),
            },
        };

        let result = weather.format_with_location("Test City");

        // Should use "percent" unit as-is (not convert)
        assert!(result.contains("60percent"), "Should use original non-% unit");
    }

    #[test]
    fn test_weather_extreme_values() {
        // Very hot temperature
        let weather = create_weather_response(52.0, 58.0, 5.0, 100.5, 50.2, "%");
        let result = weather.format_with_location("Death Valley");

        assert!(result.contains("52"), "Should handle high temperature");
        assert!(result.contains("5 percent"), "Should handle low humidity");
        assert!(result.contains("Rain:"), "Should show heavy rain");
        assert!(result.contains("Snowfall:"), "Should show heavy snow");
    }

    #[test]
    fn test_weather_small_precipitation() {
        // Very small but non-zero precipitation
        let weather = create_weather_response(10.0, 9.0, 80.0, 0.1, 0.0, "%");
        let result = weather.format_with_location("Portland");

        assert!(result.contains("Rain:"), "Should show rain even when very small (0.1)");
    }
}
