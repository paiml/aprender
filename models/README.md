# Models

Mock models for testing and development purposes only.

## Files

- `mock.gguf` - Mock GGUF model used in unit tests, integration tests, and CLI examples

## Usage

This model is referenced by YAML training configurations in `examples/yaml/` and
by integration tests. It does not contain real trained weights.

## Note

Do not use these files for inference. They exist solely to exercise code paths
during automated testing and local development.
