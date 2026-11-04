"""本地预览：托管 index.html + 假后端（不跑模型，只为了看界面）。"""
import json, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HTML = Path(__file__).with_name("index.html").read_bytes()

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path == "/api/health":
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(HTML)))
            self.end_headers()
            self.wfile.write(HTML)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/chat":
            return self.send_error(404)
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        last = (body.get("messages") or [{}])[-1].get("content", "")
        reply = f"这是预览模式，后端是假的。我收到了你说的话：「{last}」。真正接上模型后，这里会是 ATRI 的回复。"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for ch in reply:
            self.wfile.write(f"data: {json.dumps({'text': ch}, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.018)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", 8765), H)
    print("preview: http://127.0.0.1:8765")
    srv.serve_forever()
