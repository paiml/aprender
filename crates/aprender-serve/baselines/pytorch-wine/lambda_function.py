"""
PyTorch Wine Quality Lambda - Baseline for comparison with .apr format.

This is a REAL PyTorch implementation for scientific comparison.
Same model architecture as the .apr version: Linear(11, 1)
"""

import json
import time
import torch
import torch.nn as nn

# Global model for warm start optimization
MODEL = None
LOAD_TIME_US = None


class WineQualityModel(nn.Module):
    """Simple linear model matching the .apr version exactly."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(11, 1)

    def forward(self, x):
        return self.linear(x)


def load_model():
    """Load model weights - same as embedded in .apr binary."""
    global MODEL, LOAD_TIME_US

    if MODEL is not None:
        return MODEL

    start = time.perf_counter_ns()

    MODEL = WineQualityModel()

    # Same weights as in the .apr model (from wine_lambda.rs)
    weights = torch.tensor([[
        0.05,   # fixed_acidity
        -0.15,  # volatile_acidity (negative = bad)
        0.08,   # citric_acid
        0.02,   # residual_sugar
        -0.10,  # chlorides (negative = bad)
        0.03,   # free_sulfur_dioxide
        -0.02,  # total_sulfur_dioxide
        -0.05,  # density
        0.04,   # pH
        0.12,   # sulphates
        0.25,   # alcohol (highest weight)
    ]])
    bias = torch.tensor([5.0])

    MODEL.linear.weight.data = weights
    MODEL.linear.bias.data = bias
    MODEL.eval()

    LOAD_TIME_US = (time.perf_counter_ns() - start) // 1000

    return MODEL


def predict_quality(features: list[float]) -> dict:
    """Predict wine quality from 11 features."""
    model = load_model()

    start = time.perf_counter_ns()

    with torch.no_grad():
        x = torch.tensor([features], dtype=torch.float32)
        quality = model(x).item()

    inference_us = (time.perf_counter_ns() - start) // 1000

    # Clamp to valid range
    quality = max(0.0, min(10.0, quality))

    # Categorize
    if quality < 5.0:
        category = "Poor"
    elif quality < 6.0:
        category = "Below Average"
    elif quality < 7.0:
        category = "Average"
    elif quality < 8.0:
        category = "Good"
    else:
        category = "Excellent"

    # Top factors (same logic as .apr)
    feature_names = [
        "fixed_acidity", "volatile_acidity", "citric_acid",
        "residual_sugar", "chlorides", "free_sulfur_dioxide",
        "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"
    ]
    weights = model.linear.weight.data[0].tolist()
    contributions = [(abs(w * f), name) for w, f, name in zip(weights, features, feature_names)]
    contributions.sort(reverse=True)
    top_factors = [name for _, name in contributions[:3]]

    return {
        "quality": round(quality, 7),
        "category": category,
        "confidence": 1.0,
        "top_factors": top_factors,
        "inference_us": inference_us,
        "model_load_us": LOAD_TIME_US,
        "runtime": "pytorch"
    }


def lambda_handler(event, context):
    """AWS Lambda handler."""
    try:
        # Parse input
        if isinstance(event, str):
            event = json.loads(event)

        # Extract features
        features = [
            event.get("fixed_acidity", 7.0),
            event.get("volatile_acidity", 0.3),
            event.get("citric_acid", 0.3),
            event.get("residual_sugar", 2.0),
            event.get("chlorides", 0.05),
            event.get("free_sulfur_dioxide", 30.0),
            event.get("total_sulfur_dioxide", 100.0),
            event.get("density", 0.995),
            event.get("pH", 3.3),
            event.get("sulphates", 0.6),
            event.get("alcohol", 10.5),
        ]

        result = predict_quality(features)

        return {
            "statusCode": 200,
            "body": json.dumps(result)
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }


# For local testing
if __name__ == "__main__":
    test_event = {
        "fixed_acidity": 7.0,
        "volatile_acidity": 0.3,
        "citric_acid": 0.3,
        "residual_sugar": 2.0,
        "chlorides": 0.05,
        "free_sulfur_dioxide": 30.0,
        "total_sulfur_dioxide": 100.0,
        "density": 0.995,
        "pH": 3.3,
        "sulphates": 0.6,
        "alcohol": 10.5,
    }

    print("Testing PyTorch Wine Quality Lambda...")
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))
