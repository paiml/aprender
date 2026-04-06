#!/bin/bash
# Deploy PyTorch Wine Lambda via container image
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FUNCTION_NAME="baseline-pytorch-wine"
REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/pytorch-wine-lambda"

echo "=== PyTorch Wine Lambda Deployment ==="
echo "Account: $ACCOUNT_ID"
echo "Region: $REGION"
echo ""

# Create ECR repository if needed
echo "1. Creating ECR repository..."
aws ecr create-repository --repository-name pytorch-wine-lambda --region $REGION 2>/dev/null || echo "   Repository exists"

# Login to ECR
echo "2. Logging into ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REPO

# Build container
echo "3. Building Docker image..."
docker build -t pytorch-wine-lambda .

# Tag and push
echo "4. Pushing to ECR..."
docker tag pytorch-wine-lambda:latest $ECR_REPO:latest
docker push $ECR_REPO:latest

# Get image digest
IMAGE_URI="$ECR_REPO:latest"

# Check if function exists
echo "5. Deploying Lambda function..."
if aws lambda get-function --function-name $FUNCTION_NAME --region $REGION 2>/dev/null; then
    echo "   Updating existing function..."
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --image-uri $IMAGE_URI \
        --region $REGION
else
    echo "   Creating new function..."
    # Get role ARN (use existing role from wine-apr)
    ROLE_ARN=$(aws lambda get-function --function-name realizar-wine-apr --query 'Configuration.Role' --output text 2>/dev/null || echo "")

    if [ -z "$ROLE_ARN" ]; then
        echo "ERROR: Could not find Lambda execution role"
        exit 1
    fi

    aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --package-type Image \
        --code ImageUri=$IMAGE_URI \
        --role $ROLE_ARN \
        --memory-size 512 \
        --timeout 30 \
        --region $REGION
fi

# Wait for function to be ready
echo "6. Waiting for function to be ready..."
aws lambda wait function-active --function-name $FUNCTION_NAME --region $REGION

# Get function info
echo ""
echo "=== Deployment Complete ==="
aws lambda get-function --function-name $FUNCTION_NAME --region $REGION \
    --query '{FunctionName:Configuration.FunctionName,Runtime:Configuration.PackageType,CodeSize:Configuration.CodeSize,MemorySize:Configuration.MemorySize}' \
    --output table

echo ""
echo "Test with:"
echo "  aws lambda invoke --function-name $FUNCTION_NAME --payload '{\"fixed_acidity\":7.0,\"volatile_acidity\":0.3,\"citric_acid\":0.3,\"residual_sugar\":2.0,\"chlorides\":0.05,\"free_sulfur_dioxide\":30.0,\"total_sulfur_dioxide\":100.0,\"density\":0.995,\"pH\":3.3,\"sulphates\":0.6,\"alcohol\":10.5}' --cli-binary-format raw-in-base64-out response.json && cat response.json"
