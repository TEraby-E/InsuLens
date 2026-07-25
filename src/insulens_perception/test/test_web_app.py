from pathlib import Path

from fastapi.testclient import TestClient

from insulens_perception import web_app


class StubRunner:
    def __init__(self) -> None:
        self.detector = type("Detector", (), {
            "backend": "test",
            "load_error": None,
            "compatibility_error": None,
            "model_loaded": True,
            "ready_for_inspection": True,
            "category_schema": lambda self: [
                {"id": 0, "name": "tower", "source_name": "tower", "display_name": "杆塔", "color": "#38D59A"},
                {"id": 1, "name": "vehicle", "source_name": "vehicle", "display_name": "车辆", "color": "#EF6757"},
            ],
            "empty_category_counts": lambda self: {"tower": 0, "vehicle": 0},
            "model_status": lambda self: {
                "backend": self.backend,
                "model_loaded": self.model_loaded,
                "ready_for_inspection": self.ready_for_inspection,
                "class_names": ["tower", "vehicle"],
                "class_schema": self.category_schema(),
                "load_error": self.load_error,
                "compatibility_error": self.compatibility_error,
            },
        })()
        self.video_path: Path | None = None

    def inspect_video(self, source: Path, job_id: str | None = None, on_progress=None) -> dict:
        self.video_path = source
        if on_progress:
            on_progress({
                "status": "processing",
                "frames_processed": 1,
                "frames_total": 1,
                "category_counts": {"tower": 1, "vehicle": 0},
                "class_schema": self.detector.category_schema(),
            })
        return {
            "job_id": job_id or "video-job",
            "detection_total": 1,
            "fps": 12.0,
            "average_confidence": 0.0,
            "frames": 1,
            "category_counts": {"tower": 1, "vehicle": 0},
            "class_schema": self.detector.category_schema(),
            "output_media": "annotated.mp4",
            "media_type": "video",
            "report_json": "report.json",
            "report_csv": "report.csv",
            "backend": "test",
            "processing_seconds": 0.01,
        }


def test_video_upload_post_reaches_inspection_route(monkeypatch, tmp_path: Path) -> None:
    runner = StubRunner()
    monkeypatch.setattr(web_app, "runner", runner)
    monkeypatch.setattr(web_app, "RESULT_ROOT", tmp_path)

    with TestClient(web_app.app) as client:
        response = client.post(
            "/api/inspect",
            files={"upload": ("inspection.mp4", b"test video bytes", "video/mp4")},
        )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert response.json()["status"] == "queued"
    with TestClient(web_app.app) as client:
        status = client.get(f"/api/jobs/{job_id}/status")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert status.json()["report"]["detection_total"] == 1
    assert runner.video_path is not None
    assert not runner.video_path.exists()


def test_inspection_rejects_request_when_model_is_not_loaded(monkeypatch, tmp_path: Path) -> None:
    runner = StubRunner()
    runner.detector.ready_for_inspection = False
    runner.detector.load_error = "weight load failed"
    runner.detector.compatibility_error = "weight load failed"
    monkeypatch.setattr(web_app, "runner", runner)
    monkeypatch.setattr(web_app, "RESULT_ROOT", tmp_path)

    with TestClient(web_app.app) as client:
        response = client.post(
            "/api/inspect",
            files={"upload": ("inspection.mp4", b"test video bytes", "video/mp4")},
        )

    assert response.status_code == 503
    assert "weight load failed" in response.json()["detail"]


def test_health_exposes_dynamic_model_schema(monkeypatch) -> None:
    runner = StubRunner()
    monkeypatch.setattr(web_app, "runner", runner)

    with TestClient(web_app.app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["class_names"] == ["tower", "vehicle"]
    assert [item["display_name"] for item in response.json()["class_schema"]] == [
        "杆塔", "车辆"
    ]


def test_api_routes_are_registered_before_root_static_mount() -> None:
    routes = list(web_app.app.router.routes)
    inspect_index = next(index for index, route in enumerate(routes) if getattr(route, "path", None) == "/api/inspect")
    static_index = next(index for index, route in enumerate(routes) if getattr(route, "path", None) == "")

    assert inspect_index < static_index


def test_video_artifact_is_served_inline_with_byte_ranges(monkeypatch, tmp_path: Path) -> None:
    job = tmp_path / "inspection-video"
    job.mkdir()
    payload = b"synthetic-mp4-payload"
    (job / "annotated.mp4").write_bytes(payload)
    monkeypatch.setattr(web_app, "RESULT_ROOT", tmp_path)

    with TestClient(web_app.app) as client:
        response = client.get(
            "/api/jobs/inspection-video/annotated.mp4",
            headers={"Range": "bytes=0-8"},
        )

    assert response.status_code == 206
    assert response.content == payload[:9]
    assert response.headers["content-type"].startswith("video/mp4")
    assert response.headers["content-disposition"].startswith("inline;")
    assert response.headers["accept-ranges"] == "bytes"


def test_portal_previews_selected_video_before_submitting() -> None:
    portal = Path(web_app.STATIC_DIR)
    html = (portal / "index.html").read_text(encoding="utf-8")
    javascript = (portal / "app.js").read_text(encoding="utf-8")

    assert 'id="source-preview"' in html
    assert 'id="source-media"' in html
    assert "URL.createObjectURL(file)" in javascript
    assert 'document.createElement(video ? "video" : "img")' in javascript
    assert "element.controls = true" in javascript
    assert "element.playsInline = true" in javascript
