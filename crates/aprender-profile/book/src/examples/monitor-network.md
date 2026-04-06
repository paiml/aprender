# Example: Monitor Network Operations

This example shows how to use Renacer to debug network operations, analyze protocols, and troubleshoot connectivity issues.

## Scenario: Debug HTTP Request

Your HTTP client can't reach the server. Let's trace the network calls.

### Step 1: Basic Network Tracing

```bash
$ renacer -e 'trace=network' -- curl https://example.com
```

**Output:**

```text
socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) = 3
connect(3, {sa_family=AF_INET, sin_port=htons(443), sin_addr=inet_addr("93.184.216.34")}, 16) = 0
sendto(3, "\x16\x03\x01\x02\x00...", 517, MSG_NOSIGNAL, NULL, 0) = 517
recvfrom(3, "\x16\x03\x03\x00\x59...", 16384, 0, NULL, NULL) = 1234
recvfrom(3, "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n...", 16384, 0, NULL, NULL) = 2048
close(3) = 0
```

**Analysis:**
- Socket created successfully (FD 3)
- Connection to 93.184.216.34:443 succeeded
- TLS handshake completed (0x16 0x03 = TLS)
- HTTP response received (200 OK)

### Step 2: Find Connection Failures

```bash
$ renacer -e 'trace=connect' -- curl http://localhost:9999
```

**Output:**

```text
socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) = 3
connect(3, {sa_family=AF_INET, sin_port=htons(9999), sin_addr=inet_addr("127.0.0.1")}, 16) = -ECONNREFUSED
close(3) = 0
```

**Problem Found:** Connection refused - server not listening on port 9999.

### Step 3: Trace with Source Correlation

```bash
$ renacer --source -e 'trace=network' -- ./http-client
```

**Output:**

```text
socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) = 3   [src/client.rs:45 in connect_to_server]
connect(3, {...}, 16) = -ETIMEDOUT   [src/client.rs:52 in connect_to_server]
```

**Insight:** Connection timeout at `src/client.rs:52` - network unreachable or firewall blocking.

## Scenario: Analyze API Response Times

Your API client is slow. Is it network latency or server processing?

### Step 1: Measure Network Call Duration

```bash
$ renacer -c -e 'trace=network' -- curl https://api.example.com/data
```

**Output:**

```text
System Call Summary:
====================
Syscall          Calls    Total Time    Avg Time    p50      p90      p99
socket           1        0.15ms        0.150ms     -        -        -
connect          1        245.67ms      245.670ms   -        -        -
sendto           3        1.23ms        0.410ms     0.3ms    0.5ms    0.6ms
recvfrom         12       3456.78ms     288.065ms   12.3ms   567.8ms  1234.5ms
shutdown         1        0.08ms        0.080ms     -        -        -
close            1        0.05ms        0.050ms     -        -        -
```

**Analysis:**
- Connect: 245ms (DNS + TCP handshake)
- Receive: 3.4s total, p99 is 1.2s (server latency)
- Network latency dominates (97% of time)

### Step 2: Compare Multiple Requests

```bash
# First request (cold cache)
$ renacer -c -e 'trace=recvfrom' -- curl https://api.example.com/data

# Second request (warm cache)
$ renacer -c -e 'trace=recvfrom' -- curl https://api.example.com/data
```

**First Request:**
```text
Syscall          Calls    Total Time
recvfrom         12       3456.78ms
```

**Second Request:**
```text
Syscall          Calls    Total Time
recvfrom         12       234.56ms
```

**Insight:** 15x faster on second request - server caching works, cold start is slow.

## Scenario: Debug WebSocket Connection

WebSocket connection drops unexpectedly. Let's trace the lifecycle.

### Step 1: Trace Socket Lifecycle

```bash
$ renacer -e 'trace=socket,connect,send,recv,close' -- ./websocket-client
```

**Output:**

```text
socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) = 3
connect(3, {sin_addr=inet_addr("192.168.1.100"), sin_port=htons(8080)}, 16) = 0
sendto(3, "GET /ws HTTP/1.1\r\nUpgrade: websocket\r\n...", 234, MSG_NOSIGNAL, NULL, 0) = 234
recvfrom(3, "HTTP/1.1 101 Switching Protocols\r\n...", 4096, 0, NULL, NULL) = 156
# WebSocket frames exchanged
sendto(3, "\x81\x85...", 7, MSG_NOSIGNAL, NULL, 0) = 7
recvfrom(3, "\x81\x05hello", 4096, 0, NULL, NULL) = 7
# ... 500 more messages ...
recvfrom(3, "", 4096, 0, NULL, NULL) = 0
close(3) = 0
```

**Analysis:**
- HTTP upgrade to WebSocket succeeded (101 response)
- 500 messages exchanged successfully
- Server closed connection gracefully (recvfrom returns 0)

### Step 2: Find Abnormal Closures

```bash
$ renacer -e 'trace=network' -- ./websocket-client 2>&1 | grep -E "close|shutdown|recv.*= 0"
```

**Output:**

```text
recvfrom(3, "", 4096, 0, NULL, NULL) = 0
close(3) = 0
```

**Normal:** `recvfrom` returns 0 (EOF), then close - clean shutdown.

**Abnormal example:**

```text
recvfrom(3, ..., 4096, 0, NULL, NULL) = -ECONNRESET
close(3) = 0
```

**Problem:** `ECONNRESET` indicates server crashed or network issue.

## Scenario: Monitor DNS Resolution

Your application has slow startup due to DNS lookups.

### Step 1: Trace DNS Syscalls

DNS happens via socket syscalls (not dedicated DNS syscalls in Linux).

```bash
$ renacer -e 'trace=socket,connect,send,recv' -- host example.com
```

**Output:**

```text
socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP) = 3
connect(3, {sa_family=AF_INET, sin_port=htons(53), sin_addr=inet_addr("8.8.8.8")}, 16) = 0
sendto(3, "\x12\x34\x01\x00\x00\x01...", 32, MSG_NOSIGNAL, NULL, 0) = 32
recvfrom(3, "\x12\x34\x81\x80...", 1024, 0, NULL, NULL) = 48
close(3) = 0
```

**Analysis:**
- UDP socket to 8.8.8.8:53 (Google DNS)
- DNS query sent (32 bytes)
- Response received (48 bytes)

### Step 2: Measure DNS Latency

```bash
$ renacer -c -e 'trace=network' -- getent hosts api.example.com
```

**Output:**

```text
System Call Summary:
====================
Syscall          Calls    Total Time    Avg Time
socket           2        0.25ms        0.125ms
connect          2        0.15ms        0.075ms
sendto           2        0.12ms        0.060ms
recvfrom         2        456.78ms      228.390ms
close            2        0.08ms        0.040ms
```

**Problem:** `recvfrom` taking 456ms - DNS server latency or network issue.

### Step 3: Identify Slow DNS Servers

```bash
$ renacer -e 'trace=connect' -- host slow-domain.example.com
```

**Output:**

```text
socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP) = 3
connect(3, {sin_addr=inet_addr("192.168.1.1"), sin_port=htons(53)}, 16) = 0
# ... long wait ...
recvfrom(3, ..., 1024, 0, NULL, NULL) = -ETIMEDOUT
socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP) = 4
connect(4, {sin_addr=inet_addr("8.8.8.8"), sin_port=htons(53)}, 16) = 0
recvfrom(4, ..., 1024, 0, NULL, NULL) = 48
```

**Analysis:** Primary DNS (192.168.1.1) times out, fallback to 8.8.8.8 succeeds.

## Scenario: Analyze TCP Connection Parameters

Debugging TCP connection establishment and socket options.

### Step 1: Trace Socket Options

```bash
$ renacer -e 'trace=setsockopt,getsockopt' -- curl https://example.com
```

**Output:**

```text
setsockopt(3, SOL_SOCKET, SO_KEEPALIVE, [1], 4) = 0
setsockopt(3, SOL_TCP, TCP_NODELAY, [1], 4) = 0
setsockopt(3, SOL_SOCKET, SO_RCVTIMEO, {tv_sec=30, tv_usec=0}, 16) = 0
setsockopt(3, SOL_SOCKET, SO_SNDTIMEO, {tv_sec=30, tv_usec=0}, 16) = 0
```

**Analysis:**
- Keepalive enabled (detects dead connections)
- Nagle disabled (TCP_NODELAY for low latency)
- 30s receive/send timeouts

### Step 2: Debug Timeout Issues

```bash
$ renacer --source -e 'trace=setsockopt,recvfrom' -- ./slow-client
```

**Output:**

```text
setsockopt(3, SOL_SOCKET, SO_RCVTIMEO, {tv_sec=5, tv_usec=0}, 16) = 0   [src/client.rs:78]
# ... 5 seconds later ...
recvfrom(3, ..., 4096, 0, NULL, NULL) = -EAGAIN   [src/client.rs:92]
```

**Problem:** 5-second timeout is too short, causing `EAGAIN` errors.

## Scenario: Monitor Protocol-Level Behavior

Analyzing HTTP/2, gRPC, or custom protocols.

### Step 1: Count Request/Response Pairs

```bash
$ renacer -c -e 'trace=sendto,recvfrom' -- curl http://example.com
```

**Output:**

```text
System Call Summary:
====================
Syscall          Calls    Total Time
sendto           3        1.23ms
recvfrom         12       234.56ms
```

**Analysis:**
- 3 sends (HTTP request + headers)
- 12 receives (HTTP response in chunks)

### Step 2: Detect Protocol Errors

```bash
$ renacer -e 'trace=send,recv' -- ./http-client 2>&1 | grep "= -"
```

**Output:**

```text
sendto(3, "GET /api/data HTTP/1.1\r\n...", 145, MSG_NOSIGNAL, NULL, 0) = 145
recvfrom(3, "HTTP/1.1 400 Bad Request\r\n...", 4096, 0, NULL, NULL) = 123
```

**Error Found:** Server returns 400 Bad Request - malformed HTTP request.

### Step 3: Analyze Message Sizes

```bash
$ renacer --format json -e 'trace=sendto,recvfrom' -- ./grpc-client > network.json
$ jq '.syscalls[] | select(.name == "sendto") | .return.value' network.json
```

**Output:**

```text
145
234
67
512
...
```

**Insight:** Message sizes vary widely (67 to 512 bytes) - analyze protocol framing.

## Scenario: Debug TLS/SSL Handshake

TLS connection fails or is slow. Let's trace the handshake.

### Step 1: Trace TLS Handshake

```bash
$ renacer -e 'trace=send,recv' -- openssl s_client -connect example.com:443
```

**Output:**

```text
socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) = 3
connect(3, {sin_addr=inet_addr("93.184.216.34"), sin_port=htons(443)}, 16) = 0
# Client Hello
sendto(3, "\x16\x03\x01\x02\x00\x01\x00\x01\xfc\x03\x03...", 517, MSG_NOSIGNAL, NULL, 0) = 517
# Server Hello, Certificate, ServerHelloDone
recvfrom(3, "\x16\x03\x03\x00\x59\x02\x00\x00\x55...", 16384, 0, NULL, NULL) = 1234
# Client Key Exchange, ChangeCipherSpec, Finished
sendto(3, "\x16\x03\x03\x00\x46\x10\x00\x00\x42...", 137, MSG_NOSIGNAL, NULL, 0) = 137
# Server ChangeCipherSpec, Finished
recvfrom(3, "\x14\x03\x03\x00\x01\x01\x16\x03\x03...", 16384, 0, NULL, NULL) = 51
```

**Analysis:**
- TLS 1.2 handshake (0x03 0x03)
- Client Hello: 517 bytes
- Server response: 1234 bytes (certificate chain)
- Key exchange completed

### Step 2: Measure TLS Handshake Time

```bash
$ renacer -c -e 'trace=network' -- curl https://example.com
```

**Output:**

```text
System Call Summary:
====================
Syscall          Calls    Total Time    Avg Time
connect          1        45.67ms       45.670ms
sendto           5        2.34ms        0.468ms
recvfrom         8        456.78ms      57.098ms
```

**Analysis:**
- Connect: 45ms (TCP handshake)
- TLS handshake: ~460ms (sendto + recvfrom)
- Total connection setup: 505ms

### Step 3: Compare HTTP vs HTTPS

```bash
# HTTP
$ renacer -c -e 'trace=network' -- curl http://example.com > http.txt

# HTTPS
$ renacer -c -e 'trace=network' -- curl https://example.com > https.txt

$ diff http.txt https.txt
```

**Difference:**
```text
HTTP:  connect=45ms, total=234ms
HTTPS: connect=45ms, total=705ms (+470ms for TLS)
```

## Scenario: Monitor Streaming Data

Analyze real-time streaming protocols (video, audio).

### Step 1: Count Receive Rate

```bash
$ renacer -c -e 'trace=recvfrom' -- vlc http://stream.example.com/live.mp4
```

**Output:**

```text
System Call Summary:
====================
Syscall          Calls    Total Time    Avg Time    p50      p90      p99
recvfrom         12456    34567.89ms    2.777ms     1.2ms    5.6ms    23.4ms
```

**Analysis:**
- 12,456 receives in 34 seconds = 366 recv/sec
- Average 2.8ms per receive
- p99 is 23ms (occasional latency spikes)

### Step 2: Detect Buffer Starvation

```bash
$ renacer -e 'trace=recv' -- ./video-player stream.m3u8 2>&1 | grep "= 0"
```

**Output:**

```text
recvfrom(3, "...", 65536, 0, NULL, NULL) = 8192
recvfrom(3, "...", 65536, 0, NULL, NULL) = 8192
recvfrom(3, "", 65536, 0, NULL, NULL) = 0
# ... playback stutters ...
socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) = 4
```

**Problem:** Connection closed (recvfrom = 0), new connection established - rebuffering.

## Common Network Patterns

### Pattern 1: Connection Refused

**Symptom:**

```text
connect(3, {...}, 16) = -ECONNREFUSED
```

**Causes:**
- Server not running
- Firewall blocking port
- Wrong IP/port

**Fix:** Verify server is listening (`netstat -tln | grep <port>`).

### Pattern 2: Connection Timeout

**Symptom:**

```text
connect(3, {...}, 16) = -ETIMEDOUT
```

**Causes:**
- Network unreachable
- Firewall dropping packets
- Server overloaded

**Fix:** Check routing, firewall rules, server health.

### Pattern 3: Connection Reset

**Symptom:**

```text
recvfrom(3, ..., 4096, 0, NULL, NULL) = -ECONNRESET
```

**Causes:**
- Server crashed
- Proxy/LB killed connection
- TCP RST sent

**Fix:** Check server logs, proxy settings.

### Pattern 4: Broken Pipe

**Symptom:**

```text
sendto(3, ..., 1024, MSG_NOSIGNAL, NULL, 0) = -EPIPE
```

**Causes:**
- Client closed connection before write
- Server terminated unexpectedly

**Fix:** Handle SIGPIPE, check connection state before writing.

## Network Debugging Workflow

### Step 1: Verify Connection Establishment

```bash
$ renacer -e 'trace=socket,connect' -- ./app
```

**Check for:**
- Successful socket creation (FD > 0)
- Successful connect (= 0)
- Connection errors (ECONNREFUSED, ETIMEDOUT, etc.)

### Step 2: Analyze Send/Receive Patterns

```bash
$ renacer -c -e 'trace=sendto,recvfrom' -- ./app
```

**Look for:**
- Balanced send/recv counts (request/response pairs)
- Large time differences (network latency)
- Errors (EAGAIN, EWOULDBLOCK for non-blocking sockets)

### Step 3: Identify Protocol Issues

```bash
$ renacer -e 'trace=send,recv' -- ./app 2>&1 | head -50
```

**Inspect:**
- First few bytes (protocol headers)
- Message framing
- Response codes (HTTP status, etc.)

### Step 4: Measure Performance

```bash
$ renacer -c -e 'trace=network' -- ./app
```

**Analyze:**
- Total time per syscall
- p99 latency for reliability
- Call frequency for throughput

### Step 5: Export for Analysis

```bash
$ renacer --format json -e 'trace=network' -- ./app > network-trace.json
$ jq '.syscalls[] | select(.name == "recvfrom" and .return.value == -1)' network-trace.json
```

**Use Case:** Find all failed receives.

## Best Practices

### 1. Use the Network Class

```bash
$ renacer -e 'trace=network' -- ./app
```

**Why:** Automatically includes all network syscalls (socket, connect, send, recv, etc.).

### 2. Filter to Specific Operations

```bash
# Only trace receive operations
$ renacer -e 'trace=recv,recvfrom,recvmsg' -- ./app
```

**Why:** Focus on specific protocol direction (sending vs. receiving).

### 3. Combine with Statistics

```bash
$ renacer -c -e 'trace=network' -- ./app
```

**Why:** Get aggregate view of network performance.

### 4. Use Source Correlation

```bash
$ renacer --source -e 'trace=connect' -- ./app
```

**Why:** Identify which code is making connections.

### 5. Export for Protocol Analysis

```bash
$ renacer --format json -e 'trace=network' -- ./app > trace.json
# Analyze with wireshark-style tools
```

**Why:** Deep protocol analysis requires structured data.

### 6. Compare Multiple Runs

```bash
# Before optimization
$ renacer -c -e 'trace=network' -- ./app-v1 > v1.txt

# After optimization
$ renacer -c -e 'trace=network' -- ./app-v2 > v2.txt

$ diff v1.txt v2.txt
```

**Why:** Quantify network performance improvements.

## Troubleshooting

### Issue: No Network Calls Visible

**Symptoms:**

```bash
$ renacer -e 'trace=network' -- ./app
# No output
```

**Causes:**
- Application uses async I/O (io_uring, not standard syscalls)
- Network library uses different syscalls
- Application doesn't make network calls

**Solution:**

```bash
# Trace all syscalls to see what's happening
$ renacer -- ./app | head -100
```

### Issue: TLS Data is Encrypted

**Symptoms:**

```bash
recvfrom(3, "\x17\x03\x03\x04\x56...", 16384, 0, NULL, NULL) = 1110
```

**Explanation:** Renacer sees encrypted TLS data (0x17 = Application Data).

**Solution:** Use `SSLKEYLOGFILE` for TLS decryption with Wireshark, or trace before encryption.

### Issue: High recv Call Count

**Symptoms:**

```text
Syscall          Calls    Total Time
recvfrom         50000    12345.67ms
```

**Diagnosis:** Receiving in small chunks (inefficient buffering).

**Fix:** Increase receive buffer size, use `setsockopt(SO_RCVBUF)`.

## Summary

**Network debugging workflow:**
1. **Establish** - Trace socket, connect for connection issues
2. **Communicate** - Trace send/recv for data transfer
3. **Analyze** - Use statistics to find performance issues
4. **Optimize** - Compare before/after changes

**Key syscalls:**
- **socket** - Create endpoint
- **connect** - Establish connection
- **send/sendto** - Send data
- **recv/recvfrom** - Receive data
- **setsockopt** - Configure socket behavior
- **close/shutdown** - Clean up

**Common issues:**
- ECONNREFUSED - Server not listening
- ETIMEDOUT - Network unreachable
- ECONNRESET - Connection terminated
- EPIPE - Broken pipe (write to closed socket)

## Next Steps

- [Debug Performance Issues](./debug-performance.md) - Profile I/O performance
- [Attach to Running Process](./attach-process.md) - Debug production apps
- [Export to JSON/CSV](./export-data.md) - Automated analysis
