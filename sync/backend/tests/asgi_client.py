"""Minimal ASGI caller so API tests need no extra packages (no httpx / TestClient)."""
import asyncio


def request(app, method="GET", path="/", headers=None, host="127.0.0.1"):
    """Send one request straight to an ASGI app. Returns {"status", "headers", "body"}."""
    hdrs = [(b"host", host.encode())] + [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "path": path,
             "raw_path": path.encode(), "query_string": b"", "headers": hdrs, "server": ("127.0.0.1", 8766),
             "client": ("127.0.0.1", 50000), "scheme": "http", "root_path": ""}
    out = {"status": None, "headers": {}, "body": b""}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        if msg["type"] == "http.response.start":
            out["status"] = msg["status"]
            out["headers"] = {k.decode().lower(): v.decode() for k, v in msg["headers"]}
        elif msg["type"] == "http.response.body":
            out["body"] += msg.get("body", b"")

    asyncio.run(app(scope, receive, send))
    return out
