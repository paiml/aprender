#!/usr/bin/env bash
# Docker build script for Batuta
#
# Builds Docker images for different environments
#
# Usage:
#   ./scripts/docker-build.sh [prod|dev|all]
#
# Examples:
#   ./scripts/docker-build.sh prod    # Build production image only
#   ./scripts/docker-build.sh dev     # Build development image only
#   ./scripts/docker-build.sh all     # Build all images

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="batuta"
VERSION=$(grep '^version' Cargo.toml | head -1 | cut -d'"' -f2)
BUILD_TYPE="${1:-all}"

echo -e "${BLUE}🐳 Batuta Docker Build${NC}"
echo "======================"
echo "Version: $VERSION"
echo "Build type: $BUILD_TYPE"
echo

# Function to build an image
build_image() {
    local dockerfile=$1
    local tag=$2
    local description=$3

    echo -e "${BLUE}📦 Building $description...${NC}"
    echo "Dockerfile: $dockerfile"
    echo "Tag: $IMAGE_NAME:$tag"
    echo

    if docker build -f "$dockerfile" -t "$IMAGE_NAME:$tag" -t "$IMAGE_NAME:$VERSION-$tag" .; then
        echo -e "${GREEN}✓ $description built successfully${NC}"

        # Show image size
        local size=$(docker images "$IMAGE_NAME:$tag" --format "{{.Size}}")
        echo "Image size: $size"
        echo
        return 0
    else
        echo -e "${RED}✗ Failed to build $description${NC}"
        return 1
    fi
}

# Build production image
if [[ "$BUILD_TYPE" == "prod" ]] || [[ "$BUILD_TYPE" == "all" ]]; then
    if ! build_image "Dockerfile" "latest" "Production image"; then
        exit 1
    fi

    # Also tag as version
    docker tag "$IMAGE_NAME:latest" "$IMAGE_NAME:$VERSION"
    echo -e "${GREEN}✓ Tagged as $IMAGE_NAME:$VERSION${NC}"
    echo
fi

# Build development image
if [[ "$BUILD_TYPE" == "dev" ]] || [[ "$BUILD_TYPE" == "all" ]]; then
    if ! build_image "Dockerfile.dev" "dev" "Development image"; then
        exit 1
    fi
fi

# Summary
echo -e "${GREEN}✅ Build complete!${NC}"
echo
echo "Available images:"
docker images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"
echo

# Usage examples
echo -e "${BLUE}Usage Examples:${NC}"
echo "━━━━━━━━━━━━━━"
echo
echo "1. Run Batuta CLI:"
echo "   docker run -v \$(pwd):/workspace $IMAGE_NAME:latest analyze /workspace"
echo
echo "2. Start development environment:"
echo "   docker-compose up dev"
echo
echo "3. Run tests in CI:"
echo "   docker-compose up ci"
echo
echo "4. Interactive shell:"
echo "   docker run -it -v \$(pwd):/workspace $IMAGE_NAME:latest /bin/bash"
echo
echo "5. Build WASM:"
echo "   docker-compose up wasm"
echo

# Test the image
if [[ "$BUILD_TYPE" == "prod" ]] || [[ "$BUILD_TYPE" == "all" ]]; then
    echo -e "${BLUE}🧪 Testing image...${NC}"
    if docker run --rm "$IMAGE_NAME:latest" batuta --version; then
        echo -e "${GREEN}✓ Image test passed${NC}"
    else
        echo -e "${RED}✗ Image test failed${NC}"
        exit 1
    fi
fi

echo
echo -e "${GREEN}🎉 All done!${NC}"
