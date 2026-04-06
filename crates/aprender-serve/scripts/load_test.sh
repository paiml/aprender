#!/usr/bin/env bash
# Load testing script for Realizar HTTP API
#
# Usage:
#   ./scripts/load_test.sh          # Run with default server
#   ./scripts/load_test.sh --no-server  # Run against existing server
#   ./scripts/load_test.sh --help   # Show help

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVER_PORT=${SERVER_PORT:-8080}
SERVER_URL="http://127.0.0.1:${SERVER_PORT}"
START_SERVER=true
SERVER_PID=""

# Print colored message
print_msg() {
    local color=$1
    shift
    echo -e "${color}$*${NC}"
}

# Show usage
show_help() {
    cat <<EOF
Load Testing Script for Realizar

Usage: $0 [OPTIONS]

Options:
    --no-server         Don't start server (use existing server at $SERVER_URL)
    --port PORT         Server port (default: 8080)
    --help              Show this help message

Environment Variables:
    SERVER_PORT         Port to run server on (default: 8080)

Examples:
    # Start server and run load tests
    ./scripts/load_test.sh

    # Run against existing server
    ./scripts/load_test.sh --no-server

    # Use custom port
    ./scripts/load_test.sh --port 9000

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-server)
            START_SERVER=false
            shift
            ;;
        --port)
            SERVER_PORT="$2"
            SERVER_URL="http://127.0.0.1:${SERVER_PORT}"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_msg "$RED" "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Cleanup function
cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        print_msg "$YELLOW" "Stopping server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Check if server is already running
check_server() {
    if curl -s -f "$SERVER_URL/health" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Start server if needed
if [[ "$START_SERVER" == true ]]; then
    print_msg "$BLUE" "Starting Realizar server on port $SERVER_PORT..."

    # Build the binary first
    print_msg "$BLUE" "Building realizar..."
    cargo build --release --features cli

    # Start server in background
    ./target/release/realizar serve --demo --port "$SERVER_PORT" &
    SERVER_PID=$!

    # Wait for server to be ready
    print_msg "$BLUE" "Waiting for server to be ready..."
    for i in {1..30}; do
        if check_server; then
            print_msg "$GREEN" "Server is ready!"
            break
        fi
        if [[ $i -eq 30 ]]; then
            print_msg "$RED" "Server failed to start within 30 seconds"
            exit 1
        fi
        sleep 1
    done
else
    print_msg "$BLUE" "Checking for existing server at $SERVER_URL..."
    if ! check_server; then
        print_msg "$RED" "No server found at $SERVER_URL"
        print_msg "$YELLOW" "Start a server or run without --no-server flag"
        exit 1
    fi
    print_msg "$GREEN" "Server found at $SERVER_URL"
fi

# Run load tests
print_msg "$BLUE" "\n========================================="
print_msg "$BLUE" "Running Load Tests"
print_msg "$BLUE" "=========================================\n"

# Run the actual load tests
print_msg "$YELLOW" "Running load tests with cargo test..."
cargo test --test load_test --features load-test-enabled -- --nocapture --test-threads=1

# Print summary
print_msg "$GREEN" "\n========================================="
print_msg "$GREEN" "Load Tests Complete!"
print_msg "$GREEN" "=========================================\n"

# If we started the server, show how to keep it running
if [[ "$START_SERVER" == true ]]; then
    print_msg "$YELLOW" "Server will be stopped when script exits."
    print_msg "$YELLOW" "To keep server running, use: ./target/release/realizar serve --demo --port $SERVER_PORT"
fi
