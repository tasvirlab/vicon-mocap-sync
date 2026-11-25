"""Local-API protections of the sync backend: origin allow-list, loopback Host only, path containment, and the delete guard."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import main  # noqa: E402
from asgi_client import request  # noqa: E402
from utils import app_config  # noqa: E402

GOOD, EVIL = "http://127.0.0.1:5174", "https://evil.example"
TRIAL = "SUB01_lifting_T01"


def test_allowed_origin_gets_cors_header():
    r = request(main.app, path="/openapi.json", headers={"Origin": GOOD})
    assert r["status"] == 200 and r["headers"]["access-control-allow-origin"] == GOOD


def test_foreign_origin_is_refused():
    assert request(main.app, path="/openapi.json", headers={"Origin": EVIL})["status"] == 403
    assert request(main.app, method="DELETE", path=f"/api/projects/{TRIAL}", headers={"Origin": EVIL})["status"] == 403


def test_non_loopback_host_is_refused():
    assert request(main.app, path="/openapi.json", host="evil.example")["status"] == 400


def _with_outputs(fn):
    original = app_config.get_outputs_dir
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        outputs = root / "outputs"
        trial = outputs / "SUB01" / "01_lifting" / TRIAL
        (trial / "qa").mkdir(parents=True)
        (trial / "qa" / "note.txt").write_text("ok")
        (root / "secret.txt").write_text("TOP-SECRET-CONTENT")
        app_config.get_outputs_dir = lambda: outputs
        try:
            fn(root, trial)
        finally:
            app_config.get_outputs_dir = original


def test_file_download_inside_trial_works():
    def check(root, trial):
        r = request(main.app, path=f"/api/projects/{TRIAL}/file/qa/note.txt")
        assert r["status"] == 200 and r["body"] == b"ok"
    _with_outputs(check)


def test_file_download_cannot_leave_trial_folder():
    def check(root, trial):
        for name in ("../../../../../secret.txt", "..\\..\\..\\..\\..\\secret.txt", "../note.txt"):
            r = request(main.app, path=f"/api/projects/{TRIAL}/file/{name}")
            assert r["status"] in (400, 404), (name, r["status"])
            assert b"TOP-SECRET-CONTENT" not in r["body"], name
    _with_outputs(check)


def test_delete_cannot_leave_outputs_folder():
    def check(root, trial):
        for name in ("..", "..\\..", ".."):
            request(main.app, method="DELETE", path=f"/api/projects/{name}")
        assert (root / "secret.txt").exists() and (root / "outputs").exists() and trial.exists()
    _with_outputs(check)


def test_delete_removes_only_the_requested_trial():
    def check(root, trial):
        r = request(main.app, method="DELETE", path=f"/api/projects/{TRIAL}")
        assert r["status"] == 200 and not trial.exists()
        assert (root / "secret.txt").exists() and (root / "outputs").exists()
    _with_outputs(check)
