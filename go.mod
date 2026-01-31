module github.com/agalue/voice-assistant

go 1.25

require (
	github.com/gen2brain/malgo v0.11.24
	// IMPORTANT: sherpa-onnx versions must stay in sync with SHERPA_VERSION in scripts/build.sh
	// For CUDA builds, the build script compiles sherpa-onnx from source and the versions MUST match.
	// The build script will fail with an error if versions drift.
	// See README.md "Upgrading Dependencies" section for the upgrade procedure.
	// Version mapping: v1.12.22 works with ONNX Runtime 1.11.0-1.18.1 (depends on CUDA version)
	github.com/k2-fsa/sherpa-onnx-go-linux v1.12.22
	github.com/k2-fsa/sherpa-onnx-go-macos v1.12.22
	github.com/ollama/ollama v0.15.2
)

require (
	github.com/bahlo/generic-list-go v0.2.0 // indirect
	github.com/buger/jsonparser v1.1.1 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/mailru/easyjson v0.9.1 // indirect
	github.com/wk8/go-ordered-map/v2 v2.1.8 // indirect
	golang.org/x/crypto v0.47.0 // indirect
	golang.org/x/sys v0.40.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
