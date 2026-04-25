# space_entrypoint.py
# Runs training in a background thread and serves live logs on port 7860.
# HuggingFace Spaces requires an HTTP server on 7860 to stay alive.

import threading
import subprocess
import sys
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

log_lines = ["[SafeSignal] Starting GRPO training..."]
training_done = False
training_exit_code = None


def run_training():
    global training_done, training_exit_code
    proc = subprocess.Popen(
        [sys.executable, "training/train_grpo.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        line = line.rstrip()
        print(line, flush=True)
        log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] {line}")
        if len(log_lines) > 500:
            log_lines.pop(0)
    proc.wait()
    training_exit_code = proc.returncode
    training_done = True
    status = "COMPLETE" if proc.returncode == 0 else f"FAILED (code {proc.returncode})"
    log_lines.append(f"\n=== Training {status} ===")
    print(f"Training {status}", flush=True)


class LogHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = "\n".join(log_lines)
        status = "done" if training_done else "running"
        html = f"""<!DOCTYPE html>
<html>
<head>
  <title>SafeSignal Training</title>
  <meta http-equiv="refresh" content="5">
  <style>
    body {{ background:#111; color:#0f0; font-family:monospace; padding:20px; }}
    h2   {{ color:#fff; }}
    pre  {{ white-space:pre-wrap; font-size:13px; }}
    .badge {{ display:inline-block; padding:4px 12px; border-radius:4px;
              background:{'#2ecc71' if training_done else '#e67e22'}; color:#fff; }}
  </style>
</head>
<body>
  <h2>SafeSignal GRPO Training</h2>
  <span class="badge">{'Done' if training_done else 'Training...'}</span>
  <p style="color:#aaa">Page auto-refreshes every 5 s &nbsp;|&nbsp; status: {status}</p>
  <pre>{body}</pre>
</body>
</html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())

    def log_message(self, *args):
        pass  # suppress access logs


# Start training thread
t = threading.Thread(target=run_training, daemon=True)
t.start()

# Serve log UI on 7860
print("Log server: http://0.0.0.0:7860", flush=True)
HTTPServer(("0.0.0.0", 7860), LogHandler).serve_forever()
