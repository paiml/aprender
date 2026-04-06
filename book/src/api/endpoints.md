# API Endpoints

Realizar provides a REST API for ML inference with six core endpoints:

- **`GET /health`** - Health check
- **`POST /tokenize`** - Convert text to tokens
- **`POST /generate`** - Generate text from a prompt
- **`POST /batch/tokenize`** - Batch tokenize multiple texts
- **`POST /batch/generate`** - Batch generate from multiple prompts
- **`POST /stream/generate`** - Stream generation tokens via Server-Sent Events

All endpoints return JSON responses (except /stream/generate which uses SSE).

## Base URL

When running locally with default settings:
```
http://127.0.0.1:8080
```

## Endpoints Overview

| Endpoint | Method | Purpose | Authentication |
|----------|--------|---------|----------------|
| `/health` | GET | Check server status | None |
| `/tokenize` | POST | Tokenize input text | None |
| `/generate` | POST | Generate text | None |
| `/batch/tokenize` | POST | Batch tokenize texts | None |
| `/batch/generate` | POST | Batch generate text | None |
| `/stream/generate` | POST | Stream tokens via SSE | None |

## Health Check

**`GET /health`**

Check if the server is running and healthy.

### Request

```bash
curl http://127.0.0.1:8080/health
```

### Response

**Status**: `200 OK`

```json
{
  "status": "ok"
}
```

### Example

```bash
$ curl http://127.0.0.1:8080/health
{"status":"ok"}
```

### Use Cases

- **Health monitoring** - Check if server is responsive
- **Load balancer checks** - Verify instance health
- **CI/CD verification** - Ensure deployment succeeded

---

## Tokenize

**`POST /tokenize`**

Convert input text into token IDs using the model's tokenizer.

### Request

**Headers**:
```
Content-Type: application/json
```

**Body**:
```json
{
  "text": "string"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Text to tokenize |

### Response

**Status**: `200 OK`

```json
{
  "tokens": [integer, ...]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `tokens` | array of integers | Token IDs |

### Examples

**Basic tokenization**:
```bash
curl -X POST http://127.0.0.1:8080/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

Response:
```json
{
  "tokens": [7, 4, 11, 11, 14, 0, 22, 14, 17, 11, 3]
}
```

**Empty string**:
```bash
curl -X POST http://127.0.0.1:8080/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": ""}'
```

Response:
```json
{
  "tokens": []
}
```

**Special characters**:
```bash
curl -X POST http://127.0.0.1:8080/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world! 123"}'
```

Response:
```json
{
  "tokens": [7, 4, 11, 11, 14, 26, 0, 22, 14, 17, 11, 3, 27, 0, 1, 2, 3]
}
```

### Error Responses

**Missing text field**:
```json
{
  "error": "Missing required field: text"
}
```

---

## Generate

**`POST /generate`**

Generate text based on a prompt using the inference engine.

### Request

**Headers**:
```
Content-Type: application/json
```

**Body**:
```json
{
  "prompt": "string",
  "max_tokens": integer,
  "strategy": "string",
  "temperature": number,
  "top_k": integer,
  "top_p": number
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Input text to generate from |
| `max_tokens` | integer | No | 50 | Maximum tokens to generate |
| `strategy` | string | No | `"greedy"` | Sampling strategy: `"greedy"`, `"top_k"`, `"top_p"` |
| `temperature` | number | No | 1.0 | Temperature for sampling (0.0-2.0) |
| `top_k` | integer | No | 50 | Number of top tokens for top-k sampling |
| `top_p` | number | No | 0.9 | Cumulative probability for top-p sampling |

### Response

**Status**: `200 OK`

```json
{
  "text": "string",
  "num_generated": integer
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Generated text (including prompt) |
| `num_generated` | integer | Number of tokens generated (excluding prompt) |

### Examples

**Greedy sampling** (deterministic):
```bash
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello",
    "max_tokens": 10,
    "strategy": "greedy"
  }'
```

Response:
```json
{
  "text": "Hello world this is a test",
  "num_generated": 5
}
```

**Top-k sampling**:
```bash
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The weather today is",
    "max_tokens": 20,
    "strategy": "top_k",
    "top_k": 40,
    "temperature": 0.8
  }'
```

Response:
```json
{
  "text": "The weather today is sunny and warm with clear skies",
  "num_generated": 9
}
```

**Top-p (nucleus) sampling**:
```bash
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "strategy": "top_p",
    "top_p": 0.95,
    "temperature": 1.0
  }'
```

Response:
```json
{
  "text": "Once upon a time there was a brave knight...",
  "num_generated": 42
}
```

**Minimal request** (all defaults):
```bash
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello"}'
```

Response:
```json
{
  "text": "Hello world this is a test example of the system working properly and generating text from the given prompt with default parameters",
  "num_generated": 50
}
```

### Sampling Strategies

**Greedy** (`strategy: "greedy"`):
- Always selects the highest probability token
- Deterministic output (same input → same output)
- Best for: reproducible results, factual generation

**Top-k** (`strategy: "top_k"`):
- Samples from the top-k most likely tokens
- Controlled randomness
- Best for: creative text with some diversity

**Top-p/Nucleus** (`strategy: "top_p"`):
- Samples from tokens whose cumulative probability ≥ top_p
- Dynamic vocabulary size based on distribution
- Best for: natural, diverse text generation

### Temperature

Controls randomness of generation:

- **0.0**: Nearly deterministic (minimal randomness)
- **1.0**: Default (balanced)
- **2.0**: Very random (maximum creativity)

```bash
# Low temperature (more focused)
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "temperature": 0.3}'

# High temperature (more diverse)
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "temperature": 1.5}'
```

### Error Responses

**Empty prompt**:
```json
{
  "error": "Prompt cannot be empty"
}
```

**Invalid strategy**:
```json
{
  "error": "Invalid strategy: must be 'greedy', 'top_k', or 'top_p'"
}
```

**Invalid temperature**:
```json
{
  "error": "Temperature must be positive"
}
```

---

## Batch Tokenize

**`POST /batch/tokenize`**

Tokenize multiple texts in a single request for improved throughput.

### Request

**Headers**:
```
Content-Type: application/json
```

**Body**:
```json
{
  "texts": ["string", ...]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `texts` | array of strings | Yes | Texts to tokenize (1+ items) |

### Response

**Status**: `200 OK`

```json
{
  "results": [
    {
      "token_ids": [integer, ...],
      "num_tokens": integer
    },
    ...
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `results` | array | Results for each text in the same order |

### Examples

**Basic batch tokenization**:
```bash
curl -X POST http://127.0.0.1:8080/batch/tokenize \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello", "world", "test"]
  }'
```

Response:
```json
{
  "results": [
    {"token_ids": [7, 4, 11, 11, 14], "num_tokens": 5},
    {"token_ids": [22, 14, 17, 11, 3], "num_tokens": 5},
    {"token_ids": [19, 4, 18, 19], "num_tokens": 4}
  ]
}
```

### Error Responses

**Empty array**:
```json
{
  "error": "Texts array cannot be empty"
}
```

### Performance

Batch tokenization provides better throughput than individual requests:
- **Single requests**: 3 requests × ~100µs = ~300µs total
- **Batch request**: 1 request × ~250µs = ~250µs total (17% faster)

---

## Batch Generate

**`POST /batch/generate`**

Generate text for multiple prompts in a single request using shared generation parameters.

### Request

**Headers**:
```
Content-Type: application/json
```

**Body**:
```json
{
  "prompts": ["string", ...],
  "max_tokens": integer,
  "strategy": "string",
  "temperature": number,
  "top_k": integer,
  "top_p": number,
  "seed": integer
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompts` | array of strings | Yes | - | Input prompts (1+ items) |
| `max_tokens` | integer | No | 50 | Maximum tokens per generation |
| `strategy` | string | No | `"greedy"` | Sampling strategy (shared) |
| `temperature` | number | No | 1.0 | Temperature (shared) |
| `top_k` | integer | No | 50 | Top-k value (shared) |
| `top_p` | number | No | 0.9 | Top-p value (shared) |
| `seed` | integer | No | - | Random seed (shared) |

### Response

**Status**: `200 OK`

```json
{
  "results": [
    {
      "token_ids": [integer, ...],
      "text": "string",
      "num_generated": integer
    },
    ...
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `results` | array | Results for each prompt in the same order |

### Examples

**Basic batch generation**:
```bash
curl -X POST http://127.0.0.1:8080/batch/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["Hello", "How are"],
    "max_tokens": 10,
    "strategy": "greedy"
  }'
```

Response:
```json
{
  "results": [
    {
      "token_ids": [7, 4, 11, 11, 14, 0, 22, 14, 17, 11, 3],
      "text": "Hello world this is",
      "num_generated": 6
    },
    {
      "token_ids": [7, 14, 22, 0, 1, 17, 4],
      "text": "How are you today",
      "num_generated": 5
    }
  ]
}
```

**Batch generation with top-k**:
```bash
curl -X POST http://127.0.0.1:8080/batch/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["The weather", "Once upon", "In the future"],
    "max_tokens": 15,
    "strategy": "top_k",
    "top_k": 40,
    "temperature": 0.8,
    "seed": 42
  }'
```

### Use Cases

**1. Parallel prompt comparison**:
```bash
# Compare multiple prompt formulations
curl -X POST http://127.0.0.1:8080/batch/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      "Explain quantum computing:",
      "What is quantum computing?",
      "Quantum computing is"
    ],
    "max_tokens": 50,
    "strategy": "greedy"
  }'
```

**2. Multi-turn conversation**:
```bash
# Process multiple conversation turns
curl -X POST http://127.0.0.1:8080/batch/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      "User: Hello\nAssistant:",
      "User: How are you?\nAssistant:",
      "User: What can you do?\nAssistant:"
    ],
    "max_tokens": 30
  }'
```

**3. Data augmentation**:
```bash
# Generate variations with different seeds
curl -X POST http://127.0.0.1:8080/batch/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["The cat", "The cat", "The cat"],
    "max_tokens": 20,
    "strategy": "top_k",
    "temperature": 1.2
  }'
```

### Error Responses

**Empty prompts array**:
```json
{
  "error": "Prompts array cannot be empty"
}
```

**Invalid strategy**:
```json
{
  "error": "Invalid strategy: must be 'greedy', 'top_k', or 'top_p'"
}
```

**Empty prompt in array**:
```json
{
  "error": "Prompt 'example' tokenizes to empty sequence"
}
```

### Performance

Batch generation provides significant throughput improvements:

| Batch Size | Sequential Time | Batch Time | Improvement |
|------------|----------------|------------|-------------|
| 1 | 0.6 ms | 0.6 ms | - |
| 2 | 1.2 ms | 1.3 ms | 8% faster |
| 4 | 2.4 ms | 2.5 ms | 4% faster |
| 8 | 4.8 ms | 5.1 ms | 6% slower* |

*Note: Current implementation processes sequentially. Future optimizations (Phase 2) will enable true parallel batch processing with GPU acceleration for significant speedups.

### Best Practices

1. **Group similar prompts**: Batch prompts with similar lengths and generation requirements
2. **Shared parameters**: All prompts use the same generation config for consistency
3. **Error handling**: One failed prompt fails the entire batch (fail-fast)
4. **Batch size**: Start with batches of 4-8 prompts for optimal throughput
5. **Order preservation**: Results are always in the same order as input prompts

---

## Stream Generate

**`POST /stream/generate`**

Generate text with Server-Sent Events (SSE), streaming tokens as they're generated for real-time user feedback.

### Request

**Headers**:
```
Content-Type: application/json
```

**Body**: Same as `/generate`
```json
{
  "prompt": "string",
  "max_tokens": integer,
  "strategy": "string",
  "temperature": number,
  "top_k": integer,
  "top_p": number,
  "seed": integer
}
```

### Response

**Content-Type**: `text/event-stream`

**Events**:

1. **`token`** event (multiple):
```json
{
  "token_id": 42,
  "text": " world"
}
```

2. **`done`** event (final):
```json
{
  "num_generated": 10
}
```

### Examples

**Basic streaming**:
```bash
curl -N -X POST http://127.0.0.1:8080/stream/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello",
    "max_tokens": 10,
    "strategy": "greedy"
  }'
```

Output (SSE format):
```
event: token
data: {"token_id":7,"text":" world"}

event: token
data: {"token_id":19,"text":" this"}

event: token
data: {"token_id":8,"text":" is"}

event: done
data: {"num_generated":3}
```

### JavaScript Client Example

```javascript
const eventSource = new EventSource('http://127.0.0.1:8080/stream/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    prompt: 'Hello',
    max_tokens: 20
  })
});

eventSource.addEventListener('token', (event) => {
  const data = JSON.parse(event.data);
  console.log(`Token ${data.token_id}: ${data.text}`);
  // Update UI with new token
  document.getElementById('output').textContent += data.text;
});

eventSource.addEventListener('done', (event) => {
  const data = JSON.parse(event.data);
  console.log(`Generation complete: ${data.num_generated} tokens`);
  eventSource.close();
});

eventSource.onerror = () => {
  console.error('Streaming error');
  eventSource.close();
};
```

### Python Client Example

```python
import requests
import json

url = 'http://127.0.0.1:8080/stream/generate'
data = {
    'prompt': 'Hello',
    'max_tokens': 20,
    'strategy': 'greedy'
}

response = requests.post(url, json=data, stream=True)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('event:'):
            event_type = line.split(':', 1)[1].strip()
        elif line.startswith('data:'):
            data = json.loads(line.split(':', 1)[1].strip())
            if event_type == 'token':
                print(data['text'], end='', flush=True)
            elif event_type == 'done':
                print(f"\n\nGenerated {data['num_generated']} tokens")
```

### Use Cases

**1. Real-time chat interfaces**:
- Display tokens as they're generated
- Better perceived latency
- Progressive UI updates

**2. Live code generation**:
- Show code being written
- Early feedback for users
- Cancellable generation

**3. Long-form content**:
- Stream articles, stories, documentation
- Users can start reading immediately
- Better UX for slow generations

### Performance

Streaming provides better perceived performance:

| Metric | Standard `/generate` | Streaming `/stream/generate` |
|--------|---------------------|----------------------------|
| Time to first token | ~500ms (full generation) | ~50ms (immediate) |
| User perception | "Waiting..." | "Active generation" |
| Cancelable | No | Yes (close connection) |
| Memory overhead | Low | Slightly higher |

### Error Handling

Errors are sent as SSE `error` events:

```javascript
eventSource.addEventListener('error', (event) => {
  console.error('Error:', event.data);
  eventSource.close();
});
```

### Best Practices

1. **Always set `Accept: text/event-stream` header**
2. **Handle connection drops** - Implement reconnection logic
3. **Close connections** - Always close EventSource when done
4. **Buffer tokens** - Batch UI updates for better performance
5. **Set timeouts** - Detect stalled connections

---

## Testing with Different Tools

### curl

```bash
# Health check
curl http://127.0.0.1:8080/health

# Tokenize
curl -X POST http://127.0.0.1:8080/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello"}'

# Generate
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 10}'
```

### httpie

```bash
# Health check
http GET http://127.0.0.1:8080/health

# Tokenize
http POST http://127.0.0.1:8080/tokenize text="Hello"

# Generate
http POST http://127.0.0.1:8080/generate prompt="Hello" max_tokens:=10
```

### Python (requests)

```python
import requests

base_url = "http://127.0.0.1:8080"

# Health check
response = requests.get(f"{base_url}/health")
print(response.json())

# Tokenize
response = requests.post(
    f"{base_url}/tokenize",
    json={"text": "Hello world"}
)
print(response.json())

# Generate
response = requests.post(
    f"{base_url}/generate",
    json={
        "prompt": "Hello",
        "max_tokens": 20,
        "strategy": "greedy"
    }
)
print(response.json())
```

### JavaScript (fetch)

```javascript
const baseUrl = "http://127.0.0.1:8080";

// Health check
fetch(`${baseUrl}/health`)
  .then(r => r.json())
  .then(console.log);

// Tokenize
fetch(`${baseUrl}/tokenize`, {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({text: "Hello world"})
})
  .then(r => r.json())
  .then(console.log);

// Generate
fetch(`${baseUrl}/generate`, {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({
    prompt: "Hello",
    max_tokens: 20,
    strategy: "greedy"
  })
})
  .then(r => r.json())
  .then(console.log);
```

## Performance Considerations

### Latency

Typical response times (demo mode, CPU):
- **Health check**: < 1ms
- **Tokenize** (10 words): ~100µs
- **Generate** (10 tokens): ~1-2ms

### Throughput

The server handles requests serially in demo mode. For production workloads, consider:
- Request batching (Phase 2)
- Multiple instances with load balancer
- GPU acceleration for faster inference

## Next Steps

- [Generate Endpoint Details](./generate.md) - Complete generation parameters
- [Tokenize Endpoint Details](./tokenize.md) - Tokenization specifics
- [Error Handling](./error-handling.md) - Error responses and codes
- [Testing HTTP Endpoints](./testing.md) - Integration testing

## See Also

- [Serve Command](../cli/serve.md) - Starting the server
- [Generation Parameters](../generation/parameters.md) - Tuning generation
- [Sampling Strategies](../generation/sampling.md) - Strategy comparison
