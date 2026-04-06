#!/usr/bin/env python3
"""
PyTorch Inference Migration Example - Input

Real-world inference scenario:
- Load pre-trained sentiment analysis model
- Tokenize input text
- Run inference
- Decode predictions
- Batch processing

Demonstrates common PyTorch inference patterns that Batuta can transpile to Rust/Realizar.

Note: This example uses a simplified model. In practice, you'd use models from
Hugging Face Transformers or similar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class SentimentClassifier(nn.Module):
    """Simple LSTM-based sentiment classifier."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)

        # Embed tokens
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Use last hidden state
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)

        # Classifier
        logits = self.fc(last_hidden)  # (batch_size, num_classes)

        return logits


class SimpleTokenizer:
    """Simple word-level tokenizer."""

    def __init__(self):
        # Simplified vocabulary (in practice, load from file)
        self.word_to_idx = {
            "<PAD>": 0,
            "<UNK>": 1,
            "great": 2,
            "good": 3,
            "excellent": 4,
            "bad": 5,
            "terrible": 6,
            "awful": 7,
            "movie": 8,
            "film": 9,
            "the": 10,
            "a": 11,
            "is": 12,
            "was": 13,
            "very": 14,
            "really": 15,
            "not": 16,
            "love": 17,
            "hate": 18,
            "boring": 19,
            "amazing": 20,
        }
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        self.unk_idx = 1
        self.pad_idx = 0

    def encode(self, text: str, max_len: int = 20) -> List[int]:
        """Tokenize and convert text to token IDs."""
        # Lowercase and split
        words = text.lower().replace(".", "").replace(",", "").split()

        # Convert to indices
        indices = [self.word_to_idx.get(word, self.unk_idx) for word in words]

        # Pad or truncate to max_len
        if len(indices) < max_len:
            indices += [self.pad_idx] * (max_len - len(indices))
        else:
            indices = indices[:max_len]

        return indices

    def decode(self, indices: List[int]) -> str:
        """Convert token IDs back to text."""
        words = [self.idx_to_word.get(idx, "<UNK>") for idx in indices if idx != self.pad_idx]
        return " ".join(words)


def load_model(checkpoint_path: str = None) -> Tuple[SentimentClassifier, SimpleTokenizer]:
    """Load pre-trained model and tokenizer."""
    # Model hyperparameters
    vocab_size = 1000
    embedding_dim = 128
    hidden_dim = 256
    num_classes = 3  # negative, neutral, positive

    # Create model
    model = SentimentClassifier(vocab_size, embedding_dim, hidden_dim, num_classes)

    # Load checkpoint if provided (in this demo, use random weights)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Set to evaluation mode
    model.eval()

    # Create tokenizer
    tokenizer = SimpleTokenizer()

    return model, tokenizer


def predict_single(model: SentimentClassifier, tokenizer: SimpleTokenizer, text: str) -> Tuple[int, torch.Tensor]:
    """Run inference on a single text sample."""
    # Tokenize
    token_ids = tokenizer.encode(text)

    # Convert to tensor
    input_tensor = torch.tensor([token_ids], dtype=torch.long)

    # Run inference (no gradients needed)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)

    # Get prediction
    predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class, probabilities[0]


def predict_batch(model: SentimentClassifier, tokenizer: SimpleTokenizer, texts: List[str]) -> Tuple[List[int], torch.Tensor]:
    """Run inference on a batch of texts."""
    # Tokenize all texts
    token_ids_list = [tokenizer.encode(text) for text in texts]

    # Convert to tensor (batch_size, max_len)
    input_tensor = torch.tensor(token_ids_list, dtype=torch.long)

    # Run inference
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)

    # Get predictions
    predicted_classes = torch.argmax(probabilities, dim=1).tolist()

    return predicted_classes, probabilities


def main():
    """Main inference pipeline."""
    print("PyTorch Sentiment Analysis Inference")
    print("=" * 50)

    # 1. Load model and tokenizer
    print("\n1. Loading model and tokenizer...")
    model, tokenizer = load_model()
    print("   Model loaded (eval mode)")
    print(f"   Vocabulary size: {len(tokenizer.word_to_idx)}")

    # Class labels
    class_labels = ["Negative", "Neutral", "Positive"]

    # 2. Single inference
    print("\n2. Single text inference...")
    sample_text = "This movie was really great and amazing"
    print(f"   Input: \"{sample_text}\"")

    pred_class, probs = predict_single(model, tokenizer, sample_text)

    print(f"   Predicted: {class_labels[pred_class]}")
    print("   Probabilities:")
    for i, label in enumerate(class_labels):
        print(f"      {label}: {probs[i]:.4f}")

    # 3. Batch inference
    print("\n3. Batch inference...")
    batch_texts = [
        "The film was excellent",
        "This was terrible and boring",
        "An average movie, nothing special",
        "I love this, it was amazing",
    ]

    print(f"   Batch size: {len(batch_texts)}")

    pred_classes, batch_probs = predict_batch(model, tokenizer, batch_texts)

    print("   Results:")
    for i, text in enumerate(batch_texts):
        print(f"\n   Text: \"{text}\"")
        print(f"   Prediction: {class_labels[pred_classes[i]]}")
        print(f"   Probabilities: {batch_probs[i].tolist()}")

    # 4. Throughput test
    print("\n4. Throughput test...")
    test_texts = ["This is a test sentence"] * 100

    import time
    start = time.time()
    _, _ = predict_batch(model, tokenizer, test_texts)
    elapsed = time.time() - start

    throughput = len(test_texts) / elapsed
    latency = elapsed / len(test_texts) * 1000  # ms per sample

    print(f"   Processed {len(test_texts)} samples")
    print(f"   Time: {elapsed:.3f}s")
    print(f"   Throughput: {throughput:.1f} samples/sec")
    print(f"   Latency: {latency:.2f} ms/sample")

    print("\n✅ Inference pipeline complete!")

    return model, tokenizer


if __name__ == "__main__":
    main()
