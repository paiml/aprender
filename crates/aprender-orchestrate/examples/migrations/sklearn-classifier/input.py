#!/usr/bin/env python3
"""
sklearn Classifier Migration Example - Input

Real-world ML classification scenario:
- Load iris dataset
- Train/test split
- Feature scaling
- Logistic regression training
- Model evaluation
- Prediction on new data

Demonstrates common sklearn patterns that Batuta can transpile to Rust/Aprender.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data():
    """Load the iris dataset."""
    iris = load_iris()
    X = iris.data  # Features: sepal/petal length/width
    y = iris.target  # Labels: 0=setosa, 1=versicolor, 2=virginica
    feature_names = iris.feature_names
    target_names = iris.target_names

    return X, y, feature_names, target_names


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Standardize features by removing mean and scaling to unit variance."""
    scaler = StandardScaler()

    # Fit on training data only
    scaler.fit(X_train)

    # Transform both train and test
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def train_model(X_train, y_train):
    """Train logistic regression classifier."""
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        multi_class='multinomial',  # For 3+ classes
        solver='lbfgs'  # L-BFGS solver
    )

    # Fit the model
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test, target_names):
    """Evaluate model performance."""
    # Make predictions
    y_pred = model.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)

    return accuracy, conf_matrix, report, y_pred


def predict_sample(model, scaler, sample):
    """Predict class for a new sample."""
    # Scale the sample using trained scaler
    sample_scaled = scaler.transform([sample])

    # Predict
    prediction = model.predict(sample_scaled)[0]
    probabilities = model.predict_proba(sample_scaled)[0]

    return prediction, probabilities


def main():
    """Main ML classification pipeline."""
    print("sklearn Classification Pipeline - Iris Dataset")
    print("=" * 50)

    # 1. Load data
    print("\n1. Loading data...")
    X, y, feature_names, target_names = load_data()
    print(f"   Dataset shape: {X.shape}")
    print(f"   Classes: {target_names}")
    print(f"   Features: {feature_names}")

    # 2. Split data
    print("\n2. Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # 3. Feature scaling
    print("\n3. Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    print(f"   Scaler mean: {scaler.mean_}")
    print(f"   Scaler std: {scaler.scale_}")

    # 4. Train model
    print("\n4. Training logistic regression...")
    model = train_model(X_train_scaled, y_train)
    print(f"   Model classes: {model.classes_}")
    print(f"   Model coefficients shape: {model.coef_.shape}")

    # 5. Evaluate
    print("\n5. Evaluating model...")
    accuracy, conf_matrix, report, y_pred = evaluate_model(
        model, X_test_scaled, y_test, target_names
    )
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"\n   Confusion Matrix:")
    print(conf_matrix)
    print(f"\n   Classification Report:")
    print(report)

    # 6. Predict on new samples
    print("\n6. Predicting on new samples...")

    # Example: A flower with specific measurements
    new_sample = [5.1, 3.5, 1.4, 0.2]  # Likely setosa
    pred_class, pred_proba = predict_sample(model, scaler, new_sample)

    print(f"   Sample: {new_sample}")
    print(f"   Predicted class: {target_names[pred_class]}")
    print(f"   Probabilities:")
    for i, class_name in enumerate(target_names):
        print(f"      {class_name}: {pred_proba[i]:.4f}")

    print("\n✅ Pipeline complete!")

    return model, scaler, accuracy


if __name__ == "__main__":
    main()
