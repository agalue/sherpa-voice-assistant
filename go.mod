module github.com/agalue/sherpa-voice-assistant

go 1.26.0

require (
	github.com/gen2brain/malgo v0.11.24
	// NOTE: The CUDA build script (scripts/build.sh) reads these versions from go.mod at build time.
	// Updating these versions here is sufficient to upgrade the CUDA build on Linux.
	// Version mapping: v1.12.x works with ONNX Runtime 1.11.0-1.18.1 (depends on CUDA version)
	github.com/k2-fsa/sherpa-onnx-go-linux v1.12.35
	github.com/k2-fsa/sherpa-onnx-go-macos v1.12.35
	github.com/ollama/ollama v0.20.2
)

require (
	github.com/bahlo/generic-list-go v0.2.0 // indirect
	github.com/buger/jsonparser v1.1.2 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/mailru/easyjson v0.9.2 // indirect
	github.com/wk8/go-ordered-map/v2 v2.1.8 // indirect
	golang.org/x/crypto v0.49.0 // indirect
	golang.org/x/sys v0.42.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
