"""Local-API protections: origin allow-list, loopback Host only, and path containment on project file download."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import main  # noqa: E402
from asgi_client import request  # noqa: E402
from utils import app_config  # noqa: E402

GOOD, EVIL = "http://localhost:5173", "https://evil.example"


def test_allowed_origin_gets_cors_header():
    r = request(main.app, path="/openapi.json", headers={"Origin": GOOD})
    assert r["status"] == 200 and r["headers"]["access-control-allow-origin"] == GOOD


def test_foreign_origin_is_refused():
    assert request(main.app, path="/openapi.json", headers={"Origin": EVIL})["status"] == 403
    assert request(main.app, method="POST", path="/api/settings", headers={"Origin": EVIL})["status"] == 403


def test_request_without_origin_is_allowed():
    assert request(main.app, path="/openapi.json")["status"] == 200


def test_non_loopback_host_is_refused():
    assert request(main.app, path="/openapi.json", host="evil.example")["status"] == 400


def _with_exports(fn):
    original = app_config.get_exports_dir
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "SUB" / "run").mkdir(parents=True)
        (root / "SUB" / "run" / "ok.txt").write_text("ok")
        (root / "secret.txt").write_text("TOP-SECRET-CONTENT")
        app_config.get_exports_dir = lambda: root
        try:
            fn()
        finally:
            app_config.get_exports_dir = original


def test_file_download_inside_project_works():
    def check():
        r = request(main.app, path="/api/projects/SUB/run/file/ok.txt")
        assert r["status"] == 200 and r["body"] == b"ok"
    _with_exports(check)


def test_file_download_cannot_leave_project_folder():
    def check():
        for name in ("..\\..\\secret.txt", "..\\..\\..\\secret.txt", "..", "."):
            r = request(main.app, path=f"/api/projects/SUB/run/file/{name}")
            assert r["status"] in (400, 404), (name, r["status"])
            assert b"TOP-SECRET-CONTENT" not in r["body"], name
    _with_exports(check)
