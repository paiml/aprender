## End-to-End Demo: Deploying and Interacting with Qwen2.5-Coder-Instruct using `aprender`

This document provides a step-by-step guide to demonstrate the full lifecycle of a Hugging Face model within the `aprender` ecosystem, from initial import and optimization to serving via an API and embedding in a custom CLI application.

---

### Introduction

The `aprender` project offers a robust framework for managing, optimizing, and deploying machine learning models, with a focus on efficiency and compatibility across various environments. This demo showcases these capabilities using the `Qwen2.5-Coder-1.5B-Instruct` model from Hugging Face.

You will learn how to:
*   Import a Hugging Face model and convert it to the `aprender`'s native `.apr` format.
*   Quantize the `.apr` model to improve performance and reduce size.
*   Interact with the model via a built-in CLI chat interface.
*   Serve the model through an OpenAI-compatible REST API.
*   Build a custom, self-contained CLI application that embeds the model for dedicated use.

---

### Prerequisites

Before you begin, ensure you have:
*   **Rust and Cargo:** Installed on your system.
*   **`aprender` project:** Cloned to your local machine. Navigate into the project root directory where `Cargo.toml` is located.
*   **`qwen_tokenizer.json`:** The tokenizer file for the `Qwen/Qwen2.5-Coder-1.5B-Instruct` model. This is required for both the `apr chat` command and our custom CLI. Download it to your project root directory (e.g., `/home/alfredo/code/aprender/`) using the following command:

    ```bash
    curl -L -o qwen_tokenizer.json "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/resolve/main/tokenizer.json"
    ```

---

### Overall Workflow

The demo follows a logical progression:

1.  **Import:** Hugging Face Model (`.safetensors`) ➡️ Native APR Format (`.apr` - FP16)
2.  **Quantize:** APR Format (FP16) ➡️ Quantized APR Format (`.apr` - INT4)
3.  **Chat (Interactive):** Compare FP16 vs INT4 models using `apr chat`.
4.  **Serve:** Quantized APR Model (INT4) ➡️ OpenAI-Compatible REST API.
5.  **Custom CLI:** Quantized APR Model (INT4) ➡️ Embedded in a standalone Rust application.

---

### Step 1: Import the Model (Full Precision APR)

The `apr import` command is not just a download; it performs a crucial **conversion**. It downloads the model from Hugging Face and transforms its internal structure and data into `aprender`'s native, optimized `.apr` format. The `--arch qwen2` flag is vital as it tells the importer how to correctly map the Qwen2.5-Coder model's tensors into the APR canonical format.

Run the following command in your project's root directory:

```bash
cargo run -p apr-cli --release -- import hf://Qwen/Qwen2.5-Coder-1.5B-Instruct -o qwen2.5-coder-1.5b-fp16.apr --arch qwen2
```

This command will:
*   Resolve and download the `Qwen/Qwen2.5-Coder-1.5B-Instruct` model from Hugging Face.
*   Convert the model's structure into the `aprender` APR format.
*   Save the resulting full-precision (FP16) `.apr` file as `qwen2.5-coder-1.5b-fp16.apr` in your project root.

---

### Step 2: Quantize the Model (INT4 APR)

To optimize the model for faster inference and reduced memory footprint, we will quantize it to 4-bit integer precision (INT4). The `apr convert` command takes an existing `.apr` file and applies transformations like quantization, producing a new `.apr` file.

Run the following command:

```bash
cargo run -p apr-cli --release -- convert qwen2.5-coder-1.5b-fp16.apr --quantize int4 -o qwen2.5-coder-1.5b-int4.apr
```

This command will:
*   Load the `qwen2.5-coder-1.5b-fp16.apr` model.
*   Apply INT4 quantization to its weights.
*   Save the optimized, smaller model as `qwen2.5-coder-1.5b-int4.apr` in your project root.

---

### Step 3: Chat with the Models (Interactive CLI)

Now, let's experience both models directly using the `apr chat` command. This allows you to compare the full-precision (FP16) model with its quantized (INT4) counterpart.

1.  **Chat with the Full-Precision (FP16) Model:**
    ```bash
    cargo run -p apr-cli --release -- chat qwen2.5-coder-1.5b-fp16.apr
    ```
    Type your questions and press `Enter`. To exit, type `/exit`.

2.  **Chat with the Quantized (INT4) Model:**
    ```bash
    cargo run -p apr-cli --release -- chat qwen2.5-coder-1.5b-int4.apr
    ```
    You should observe that the INT4 model loads faster and consumes less memory. For most tasks, the response quality should remain very high, demonstrating the effectiveness of the quantization process.

---

### Step 4: Serve the Quantized Model (OpenAI-compatible REST API)

The `apr serve` command allows you to deploy your quantized model as a local web service, exposing an API that is compatible with OpenAI's `chat/completions` endpoint. This enables seamless integration with various client applications and tools.

Run the following command in a **new terminal window** (or as a background process):

```bash
cargo run -p apr-cli --release -- serve qwen2.5-coder-1.5b-int4.apr --port 8080
```

Once the server indicates it's ready, you can test it by sending a request using `curl` from another terminal:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d 
  {
    "model": "default",
    "messages": [{"role": "user", "content": "Write a short Rust function that returns the sum of two i32 integers."}],
    "max_tokens": 100
  }
```
You should receive a JSON response containing the model's generated code.
**Remember to stop the `apr serve` process (e.g., by pressing `Ctrl+C` in its terminal) when you are finished.**

---

### Step 5: Build a Custom CLI Tool with Embedded Model

Finally, we'll demonstrate how to integrate the quantized model directly into a custom Rust application. This showcases the programmatic use of `aprender`'s inference capabilities to create a self-contained chat CLI.

The code for this demo is already provided in the file `examples/end_to_end_demo.rs` (which was created in the previous step).

To compile and run your custom CLI:

```bash
cargo run --release --example end_to_end_demo
```

This will launch a dedicated interactive chat session using the `qwen2.5-coder-1.5b-int4.apr` model. Type your messages and press `Enter`. To exit, type `/exit`.

---

### Conclusion

You have successfully completed an end-to-end demonstration of the `Qwen2.5-Coder-1.5B-Instruct` model using `aprender`. This workflow highlighted:
*   The essential **conversion** performed by `apr import` into the native `.apr` format.
*   The significant **optimization** achieved through quantization with `apr convert`.
*   The ease of **interactive engagement** via `apr chat`.
*   The power of **API exposure** using `apr serve` for broader integration.
*   The flexibility of **programmatic embedding** in custom applications.

This comprehensive demo underscores `aprender`'s role in streamlining the deployment and utilization of modern large language models.