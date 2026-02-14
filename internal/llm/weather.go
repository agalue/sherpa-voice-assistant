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

// WeatherArgs is the arguments for the weather tool.
type WeatherArgs struct {
	City string `json:"city,omitempty"`
}

// IPGeolocation is the response from ip-api.com.
type IPGeolocation struct {
	CountryCode string  `json:"countryCode"`
	Region      string  `json:"region"`
	City        string  `json:"city"`
	Lat         float64 `json:"lat"`
	Lon         float64 `json:"lon"`
}

// Geolocation is the response from OpenStreetMap Nominatim API.
type Geolocation struct {
	DisplayName string `json:"display_name"`
	Lat         string `json:"lat"`
	Lon         string `json:"lon"`
}

// WeatherResponse is the response from Open-Meteo API.
type WeatherResponse struct {
	Current      CurrentWeather      `json:"current"`
	CurrentUnits CurrentWeatherUnits `json:"current_units"`
}

// CurrentWeather contains current weather data.
type CurrentWeather struct {
	Temperature2m       float64 `json:"temperature_2m"`
	ApparentTemperature float64 `json:"apparent_temperature"`
	RelativeHumidity2m  float64 `json:"relative_humidity_2m"`
	Rain                float64 `json:"rain"`
	Snowfall            float64 `json:"snowfall"`
}

// CurrentWeatherUnits contains units for current weather data.
type CurrentWeatherUnits struct {
	Temperature2m       string `json:"temperature_2m"`
	ApparentTemperature string `json:"apparent_temperature"`
	RelativeHumidity2m  string `json:"relative_humidity_2m"`
	Rain                string `json:"rain"`
	Snowfall            string `json:"snowfall"`
}

// FormatWithLocation formats weather data with location for speech output.
func (w *WeatherResponse) FormatWithLocation(location string) string {
	var output strings.Builder
	output.WriteString(fmt.Sprintf("Weather for %s: ", location))

	// Format for TTS: spell out units for better voice synthesis
	output.WriteString(fmt.Sprintf(
		"Temperature is %v%s, feels like %v%s. ",
		w.Current.Temperature2m, w.CurrentUnits.Temperature2m,
		w.Current.ApparentTemperature, w.CurrentUnits.ApparentTemperature,
	))

	// Convert % to "percent" for TTS (with space for better pronunciation)
	humidityUnit := w.CurrentUnits.RelativeHumidity2m
	if humidityUnit == "%" {
		humidityUnit = " percent"
	}
	output.WriteString(fmt.Sprintf("Humidity is %v%s. ", w.Current.RelativeHumidity2m, humidityUnit))

	if w.Current.Rain > 0.0 {
		output.WriteString(fmt.Sprintf("Rain: %v%s. ", w.Current.Rain, w.CurrentUnits.Rain))
	}

	if w.Current.Snowfall > 0.0 {
		output.WriteString(fmt.Sprintf("Snowfall: %v%s. ", w.Current.Snowfall, w.CurrentUnits.Snowfall))
	}

	return output.String()
}

// GetCurrentIP fetches the current public IP address.
func GetCurrentIP(ctx context.Context) (string, error) {
	client := &http.Client{Timeout: 5 * time.Second}
	req, err := http.NewRequestWithContext(ctx, "GET", "https://ifconfig.me/ip", nil)
	if err != nil {
		return "", fmt.Errorf("failed to create IP request: %w", err)
	}

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to get IP address: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read IP response: %w", err)
	}

	return strings.TrimSpace(string(body)), nil
}

// GetCoordsFromIP gets coordinates from IP address using geolocation API.
func GetCoordsFromIP(ctx context.Context, ipAddr string) (lat, lon float64, location string, err error) {
	client := &http.Client{Timeout: 5 * time.Second}
	url := fmt.Sprintf("http://ip-api.com/json/%s", ipAddr)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return 0, 0, "", fmt.Errorf("failed to create geolocation request: %w", err)
	}

	resp, err := client.Do(req)
	if err != nil {
		return 0, 0, "", fmt.Errorf("IP geolocation request failed: %w", err)
	}
	defer resp.Body.Close()

	var ipGeo IPGeolocation
	if err := json.NewDecoder(resp.Body).Decode(&ipGeo); err != nil {
		return 0, 0, "", fmt.Errorf("failed to parse IP geolocation response: %w", err)
	}

	location = fmt.Sprintf("%s, %s, %s", ipGeo.City, ipGeo.Region, ipGeo.CountryCode)
	return ipGeo.Lat, ipGeo.Lon, location, nil
}

// GetCoordsFromCity gets coordinates from city name using OpenStreetMap Nominatim API.
func GetCoordsFromCity(ctx context.Context, city string) (lat, lon float64, location string, err error) {
	if city == "" {
		return 0, 0, "", fmt.Errorf("city name is empty")
	}

	client := &http.Client{Timeout: 5 * time.Second}
	encodedCity := url.QueryEscape(city)
	url := fmt.Sprintf("https://nominatim.openstreetmap.org/search?q=%s&format=json&limit=1", encodedCity)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return 0, 0, "", fmt.Errorf("failed to create geocoding request: %w", err)
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; VoiceAssistant/1.0)")

	resp, err := client.Do(req)
	if err != nil {
		return 0, 0, "", fmt.Errorf("Nominatim geocoding request failed for '%s': %w", city, err)
	}
	defer resp.Body.Close()

	var geoResults []Geolocation
	if err := json.NewDecoder(resp.Body).Decode(&geoResults); err != nil {
		return 0, 0, "", fmt.Errorf("failed to parse Nominatim response: %w", err)
	}

	if len(geoResults) == 0 {
		return 0, 0, "", fmt.Errorf("city not found: %s", city)
	}

	var latVal, lonVal float64
	if _, err := fmt.Sscanf(geoResults[0].Lat, "%f", &latVal); err != nil {
		return 0, 0, "", fmt.Errorf("failed to parse latitude: %w", err)
	}
	if _, err := fmt.Sscanf(geoResults[0].Lon, "%f", &lonVal); err != nil {
		return 0, 0, "", fmt.Errorf("failed to parse longitude: %w", err)
	}

	return latVal, lonVal, geoResults[0].DisplayName, nil
}

// GetWeatherData fetches weather data from Open-Meteo API.
func GetWeatherData(ctx context.Context, lat, lon float64, location string) (string, error) {
	client := &http.Client{Timeout: 5 * time.Second}
	url := fmt.Sprintf(
		"https://api.open-meteo.com/v1/forecast?latitude=%v&longitude=%v&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,showers,snowfall",
		lat, lon,
	)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return "", fmt.Errorf("failed to create weather request: %w", err)
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; VoiceAssistant/1.0)")

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("weather API request failed: %w", err)
	}
	defer resp.Body.Close()

	var weather WeatherResponse
	if err := json.NewDecoder(resp.Body).Decode(&weather); err != nil {
		return "", fmt.Errorf("failed to parse weather response: %w", err)
	}

	return weather.FormatWithLocation(location), nil
}

// ExecuteWeatherTool executes the weather tool with the given JSON arguments.
func ExecuteWeatherTool(ctx context.Context, jsonArgs string) (string, error) {
	log.Println("🌤️  Fetching weather data...")

	var args WeatherArgs
	if err := json.Unmarshal([]byte(jsonArgs), &args); err != nil {
		return "", fmt.Errorf("failed to parse weather tool arguments: %w", err)
	}

	var lat, lon float64
	var location string
	var err error

	// Determine location
	if args.City != "" {
		// Treat "current", empty, or similar phrases as request for IP-based location
		cityLower := strings.ToLower(args.City)
		if cityLower == "current" || cityLower == "here" || cityLower == "my location" || strings.Contains(cityLower, "current location") {
			ipAddr, err := GetCurrentIP(ctx)
			if err != nil {
				return "", err
			}
			lat, lon, location, err = GetCoordsFromIP(ctx, ipAddr)
			if err != nil {
				return "", err
			}
		} else {
			lat, lon, location, err = GetCoordsFromCity(ctx, args.City)
			if err != nil {
				return "", err
			}
		}
	} else {
		// No city specified, use IP-based location
		ipAddr, err := GetCurrentIP(ctx)
		if err != nil {
			return "", err
		}
		lat, lon, location, err = GetCoordsFromIP(ctx, ipAddr)
		if err != nil {
			return "", err
		}
	}

	return GetWeatherData(ctx, lat, lon, location)
}
