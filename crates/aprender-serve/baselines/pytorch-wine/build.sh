#!/bin/bash
# Build PyTorch Wine Lambda package using Docker for Lambda compatibility
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building PyTorch Wine Lambda package..."

# Create a temporary directory for the package
rm -rf package lambda-pytorch-wine.zip
mkdir -p package

# Use Docker to install dependencies in Lambda-compatible environment
docker run --rm -v "$SCRIPT_DIR:/var/task" \
  public.ecr.aws/sam/build-python3.11:latest \
  /bin/bash -c "pip install torch==2.1.0 --target /var/task/package --platform manylinux2014_x86_64 --only-binary=:all: --no-cache-dir 2>/dev/null || pip install torch==2.1.0 --target /var/task/package --no-cache-dir"

# Copy Lambda function
cp lambda_function.py package/

# Create zip
cd package
zip -r9 ../lambda-pytorch-wine.zip . -x "*.pyc" -x "__pycache__/*" -x "*.dist-info/*"
cd ..

# Show size
ZIPSIZE=$(du -h lambda-pytorch-wine.zip | cut -f1)
echo "Package size: $ZIPSIZE"
echo "Built: lambda-pytorch-wine.zip"
