# First Server

Serve a model as an HTTP API:

```bash
apr serve model.gguf --port 8080
```

Then query it:

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 2+2?", "max_tokens": 32}'
```

## OpenAI-Compatible API

The server implements the OpenAI completions API:

```bash
# Chat completions
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'
```
