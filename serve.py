#!/usr/bin/env python3
"""Serve results.html on http://localhost:8000 - open in browser to view results."""
import http.server
import webbrowser
import threading
import time
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PORT = 8000

def open_browser():
    time.sleep(1)
    webbrowser.open(f"http://localhost:{PORT}/results.html")

if not os.path.exists("results.html"):
    print("Run 'python run_all.py' first to generate results.html")
    exit(1)

threading.Thread(target=open_browser, daemon=True).start()
print(f"Serving at http://localhost:{PORT}/results.html")
print("Press Ctrl+C to stop")
http.server.HTTPServer(("", PORT), http.server.SimpleHTTPRequestHandler).serve_forever()
