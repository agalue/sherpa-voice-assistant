package llm

import (
	"strings"
	"testing"
)

// TestWeatherFormatWithLocation validates basic weather formatting.
func TestWeatherFormatWithLocation(t *testing.T) {
	weather := &WeatherResponse{
		Current: CurrentWeather{
			Temperature2m:       22.5,
			ApparentTemperature: 21.0,
			RelativeHumidity2m:  65.0,
			Rain:                0.0,
			Snowfall:            0.0,
		},
		CurrentUnits: CurrentWeatherUnits{
			Temperature2m:       "°C",
			ApparentTemperature: "°C",
			RelativeHumidity2m:  "%",
			Rain:                "mm",
			Snowfall:            "cm",
		},
	}

	result := weather.FormatWithLocation("San Francisco")

	// Verify location is included
	if !strings.Contains(result, "San Francisco") {
		t.Errorf("Result should contain location: %s", result)
	}

	// Verify temperature is included
	if !strings.Contains(result, "22.5") {
		t.Errorf("Result should contain temperature: %s", result)
	}

	// Verify apparent temperature is included
	if !strings.Contains(result, "21") {
		t.Errorf("Result should contain apparent temperature: %s", result)
	}

	// Verify humidity is included
	if !strings.Contains(result, "65") {
		t.Errorf("Result should contain humidity: %s", result)
	}

	// Verify rain and snowfall are NOT included (both are 0.0)
	if strings.Contains(result, "Rain:") {
		t.Errorf("Result should not contain rain when 0.0: %s", result)
	}
	if strings.Contains(result, "Snowfall:") {
		t.Errorf("Result should not contain snowfall when 0.0: %s", result)
	}
}

// TestWeatherUnitConversion validates % to "percent" conversion.
func TestWeatherUnitConversion(t *testing.T) {
	weather := &WeatherResponse{
		Current: CurrentWeather{
			Temperature2m:       20.0,
			ApparentTemperature: 19.0,
			RelativeHumidity2m:  80.0,
			Rain:                0.0,
			Snowfall:            0.0,
		},
		CurrentUnits: CurrentWeatherUnits{
			Temperature2m:       "°C",
			ApparentTemperature: "°C",
			RelativeHumidity2m:  "%",
			Rain:                "mm",
			Snowfall:            "cm",
		},
	}

	result := weather.FormatWithLocation("London")

	// Verify % is converted to " percent" (with space for TTS)
	if !strings.Contains(result, "80 percent") {
		t.Errorf("Result should contain '80 percent': %s", result)
	}

	// Verify % symbol is not present
	if strings.Contains(result, "80%") {
		t.Errorf("Result should not contain '80%%': %s", result)
	}
}

// TestWeatherConditionalFields validates rain/snowfall only shown when > 0.0.
func TestWeatherConditionalFields(t *testing.T) {
	tests := []struct {
		name       string
		rain       float64
		snowfall   float64
		expectRain bool
		expectSnow bool
	}{
		{"No precipitation", 0.0, 0.0, false, false},
		{"Only rain", 5.2, 0.0, true, false},
		{"Only snow", 0.0, 3.1, false, true},
		{"Both", 2.5, 1.8, true, true},
		{"Very small rain", 0.1, 0.0, true, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			weather := &WeatherResponse{
				Current: CurrentWeather{
					Temperature2m:       15.0,
					ApparentTemperature: 14.0,
					RelativeHumidity2m:  70.0,
					Rain:                tt.rain,
					Snowfall:            tt.snowfall,
				},
				CurrentUnits: CurrentWeatherUnits{
					Temperature2m:       "°C",
					ApparentTemperature: "°C",
					RelativeHumidity2m:  "%",
					Rain:                "mm",
					Snowfall:            "cm",
				},
			}

			result := weather.FormatWithLocation("New York")

			hasRain := strings.Contains(result, "Rain:")
			if hasRain != tt.expectRain {
				t.Errorf("Rain presence = %v, want %v. Result: %s", hasRain, tt.expectRain, result)
			}

			hasSnow := strings.Contains(result, "Snowfall:")
			if hasSnow != tt.expectSnow {
				t.Errorf("Snowfall presence = %v, want %v. Result: %s", hasSnow, tt.expectSnow, result)
			}
		})
	}
}

// TestWeatherEdgeCases validates edge cases like negative temperature.
func TestWeatherEdgeCases(t *testing.T) {
	tests := []struct {
		name    string
		weather WeatherResponse
	}{
		{
			name: "Negative temperature",
			weather: WeatherResponse{
				Current: CurrentWeather{
					Temperature2m:       -5.2,
					ApparentTemperature: -8.0,
					RelativeHumidity2m:  90.0,
					Rain:                0.0,
					Snowfall:            0.0,
				},
				CurrentUnits: CurrentWeatherUnits{
					Temperature2m:       "°C",
					ApparentTemperature: "°C",
					RelativeHumidity2m:  "%",
					Rain:                "mm",
					Snowfall:            "cm",
				},
			},
		},
		{
			name: "Zero temperature",
			weather: WeatherResponse{
				Current: CurrentWeather{
					Temperature2m:       0.0,
					ApparentTemperature: -2.0,
					RelativeHumidity2m:  100.0,
					Rain:                0.0,
					Snowfall:            5.0,
				},
				CurrentUnits: CurrentWeatherUnits{
					Temperature2m:       "°C",
					ApparentTemperature: "°C",
					RelativeHumidity2m:  "%",
					Rain:                "mm",
					Snowfall:            "cm",
				},
			},
		},
		{
			name: "Very high humidity",
			weather: WeatherResponse{
				Current: CurrentWeather{
					Temperature2m:       25.0,
					ApparentTemperature: 28.0,
					RelativeHumidity2m:  100.0,
					Rain:                0.0,
					Snowfall:            0.0,
				},
				CurrentUnits: CurrentWeatherUnits{
					Temperature2m:       "°C",
					ApparentTemperature: "°C",
					RelativeHumidity2m:  "%",
					Rain:                "mm",
					Snowfall:            "cm",
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.weather.FormatWithLocation("Oslo")

			// Should not panic and should produce valid output
			if len(result) == 0 {
				t.Error("Result should not be empty")
			}

			// Should contain location
			if !strings.Contains(result, "Oslo") {
				t.Errorf("Result should contain location: %s", result)
			}

			// Should contain temperature (even if negative or zero)
			if !strings.Contains(result, "Temperature") {
				t.Errorf("Result should contain temperature: %s", result)
			}

			// Should contain humidity
			if !strings.Contains(result, "Humidity") {
				t.Errorf("Result should contain humidity: %s", result)
			}
		})
	}
}

// TestWeatherMultiByteLocation validates UTF-8 city names.
func TestWeatherMultiByteLocation(t *testing.T) {
	weather := &WeatherResponse{
		Current: CurrentWeather{
			Temperature2m:       30.0,
			ApparentTemperature: 32.0,
			RelativeHumidity2m:  75.0,
			Rain:                0.0,
			Snowfall:            0.0,
		},
		CurrentUnits: CurrentWeatherUnits{
			Temperature2m:       "°C",
			ApparentTemperature: "°C",
			RelativeHumidity2m:  "%",
			Rain:                "mm",
			Snowfall:            "cm",
		},
	}

	locations := []string{
		"São Paulo",
		"Zürich",
		"北京", // Beijing in Chinese
		"Reykjavík",
		"Montréal",
	}

	for _, location := range locations {
		t.Run(location, func(t *testing.T) {
			result := weather.FormatWithLocation(location)

			// Should handle UTF-8 without panic
			if len(result) == 0 {
				t.Error("Result should not be empty for UTF-8 location")
			}

			// Should contain the location
			if !strings.Contains(result, location) {
				t.Errorf("Result should contain location '%s': %s", location, result)
			}

			// Should still format weather data correctly
			if !strings.Contains(result, "30") || !strings.Contains(result, "75") {
				t.Errorf("Result should contain weather data: %s", result)
			}
		})
	}
}

// TestWeatherDifferentUnits validates formatting with different temperature units.
func TestWeatherDifferentUnits(t *testing.T) {
	// Fahrenheit test
	weatherF := &WeatherResponse{
		Current: CurrentWeather{
			Temperature2m:       72.0,
			ApparentTemperature: 70.0,
			RelativeHumidity2m:  60.0,
			Rain:                0.0,
			Snowfall:            0.0,
		},
		CurrentUnits: CurrentWeatherUnits{
			Temperature2m:       "°F",
			ApparentTemperature: "°F",
			RelativeHumidity2m:  "%",
			Rain:                "in",
			Snowfall:            "in",
		},
	}

	result := weatherF.FormatWithLocation("Miami")

	// Verify Fahrenheit unit is preserved
	if !strings.Contains(result, "°F") {
		t.Errorf("Result should contain °F: %s", result)
	}

	// Verify inches for rain unit
	if weatherF.CurrentUnits.Rain != "in" {
		t.Error("Rain unit should be 'in'")
	}
}

// TestWeatherAbnormalValues validates handling of unusual but valid values.
func TestWeatherAbnormalValues(t *testing.T) {
	weather := &WeatherResponse{
		Current: CurrentWeather{
			Temperature2m:       52.0,  // Very hot
			ApparentTemperature: 58.0,  // Heat index
			RelativeHumidity2m:  5.0,   // Very dry
			Rain:                100.5, // Heavy rain
			Snowfall:            50.2,  // Heavy snow
		},
		CurrentUnits: CurrentWeatherUnits{
			Temperature2m:       "°C",
			ApparentTemperature: "°C",
			RelativeHumidity2m:  "%",
			Rain:                "mm",
			Snowfall:            "cm",
		},
	}

	result := weather.FormatWithLocation("Death Valley")

	// Should handle extreme values without issue
	if !strings.Contains(result, "52") {
		t.Errorf("Result should contain high temperature: %s", result)
	}

	if !strings.Contains(result, "5 percent") {
		t.Errorf("Result should contain low humidity: %s", result)
	}

	// Both rain and snowfall should be present (even if unusual combination)
	if !strings.Contains(result, "Rain:") {
		t.Errorf("Result should contain rain: %s", result)
	}

	if !strings.Contains(result, "Snowfall:") {
		t.Errorf("Result should contain snowfall: %s", result)
	}
}
