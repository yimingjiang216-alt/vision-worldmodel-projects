# -*- coding: utf-8 -*-
"""带正确 MIME 的本地服务器 (.mjs 必须是 text/javascript)."""
import http.server, socketserver, functools, sys, os

class H(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".mjs": "text/javascript",
                      ".wasm": "application/wasm",
                      ".json": "application/json"}

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8791
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with socketserver.TCPServer(("127.0.0.1", port), functools.partial(H, directory=".")) as httpd:
        print("serving on", port, flush=True)
        httpd.serve_forever()
