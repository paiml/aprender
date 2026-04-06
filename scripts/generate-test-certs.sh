#!/bin/sh
# Generate self-signed TLS certificates for testing
# ⚡ Purified by bashrs - POSIX-compliant with safety guarantees

set -eu

CERT_DIR="${1:-./certs}"
DAYS="${2:-365}"

echo "🔐 Generating self-signed TLS certificates for testing"
echo "   Directory: ${CERT_DIR}"
echo "   Validity: ${DAYS} days"
echo ""

# Create certificate directory
mkdir -p "${CERT_DIR}"

# Generate CA certificate
echo "  [1/3] Generating CA certificate..."
openssl req -x509 \
    -newkey rsa:4096 \
    -keyout "${CERT_DIR}/ca.key" \
    -out "${CERT_DIR}/ca.pem" \
    -days "${DAYS}" \
    -nodes \
    -subj "/C=US/ST=State/L=City/O=Repartir Test CA/CN=repartir-ca" \
    >/dev/null 2>&1

# Generate server certificate
echo "  [2/3] Generating server certificate..."
openssl req -newkey rsa:4096 \
    -keyout "${CERT_DIR}/server.key" \
    -out "${CERT_DIR}/server.csr" \
    -nodes \
    -subj "/C=US/ST=State/L=City/O=Repartir/CN=localhost" \
    >/dev/null 2>&1

# Create SAN extension file (POSIX-compliant)
printf "subjectAltName=DNS:localhost,IP:127.0.0.1" > "${CERT_DIR}/san.ext"

openssl x509 -req \
    -in "${CERT_DIR}/server.csr" \
    -CA "${CERT_DIR}/ca.pem" \
    -CAkey "${CERT_DIR}/ca.key" \
    -CAcreateserial \
    -out "${CERT_DIR}/server.pem" \
    -days "${DAYS}" \
    -extfile "${CERT_DIR}/san.ext" \
    >/dev/null 2>&1

# Generate client certificate
echo "  [3/3] Generating client certificate..."
openssl req -newkey rsa:4096 \
    -keyout "${CERT_DIR}/client.key" \
    -out "${CERT_DIR}/client.csr" \
    -nodes \
    -subj "/C=US/ST=State/L=City/O=Repartir/CN=client" \
    >/dev/null 2>&1

openssl x509 -req \
    -in "${CERT_DIR}/client.csr" \
    -CA "${CERT_DIR}/ca.pem" \
    -CAkey "${CERT_DIR}/ca.key" \
    -CAcreateserial \
    -out "${CERT_DIR}/client.pem" \
    -days "${DAYS}" \
    >/dev/null 2>&1

# Clean up CSR and temp files
rm -f "${CERT_DIR}"/*.csr "${CERT_DIR}/ca.srl" "${CERT_DIR}/san.ext"

echo ""
echo "✅ Certificates generated successfully!"
echo ""
echo "Files created:"
echo "  CA:     ${CERT_DIR}/ca.pem (+ ca.key)"
echo "  Server: ${CERT_DIR}/server.pem (+ server.key)"
echo "  Client: ${CERT_DIR}/client.pem (+ client.key)"
echo ""
echo "⚠️  WARNING: These are self-signed certificates for TESTING ONLY"
echo "   DO NOT use in production!"
