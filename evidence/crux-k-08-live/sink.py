import http.server, pathlib, sys
OUT = pathlib.Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
n = [0]
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        n[0] += 1
        (OUT / f"body-{n[0]:03d}.json").write_bytes(body)
        with open(OUT / "requests.log", "a") as f:
            f.write(f"{self.path} {self.headers.get('Content-Type')} {len(body)}\n")
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.end_headers(); self.wfile.write(b"{}")
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
