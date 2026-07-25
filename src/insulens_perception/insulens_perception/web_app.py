"""FastAPI entrypoint for upload, inspection, live polling and report export."""

from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
import shutil
from threading import Lock, Thread
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .inspection_service import InspectionRunner, InsulatorDetector


ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = ROOT / "src" / "insulens_perception" / "web" / "inspection_portal"
RESULT_ROOT = Path(os.getenv("INSULENS_RESULT_DIR", ROOT / "inspection_results" / "web"))


def _json_mapping(environment_name: str) -> dict[str, str]:
    raw_value = os.getenv(environment_name, "").strip()
    if not raw_value:
        return {}
    value = json.loads(raw_value)
    if not isinstance(value, dict):
        raise ValueError(f"{environment_name} 必须是 JSON 对象。")
    return {str(key): str(item) for key, item in value.items()}


def _optional_float(environment_name: str) -> float | None:
    value = os.getenv(environment_name, "").strip()
    return float(value) if value else None


WEIGHTS = os.getenv(
    "INSULENS_WEB_MODEL",
    str(ROOT / "models" / "insulator_six_class_yolov10s.pt"),
)
runner = InspectionRunner(
    InsulatorDetector(
        WEIGHTS,
        class_aliases=_json_mapping("INSULENS_CLASS_ALIASES"),
        class_labels=_json_mapping("INSULENS_CLASS_LABELS"),
        inferred_class=os.getenv("INSULENS_INFERRED_CLASS"),
        inferred_class_threshold=_optional_float("INSULENS_INFERRED_CLASS_THRESHOLD"),
        inference_candidate_confidence=_optional_float(
            "INSULENS_INFERENCE_CANDIDATE_CONFIDENCE"
        ),
    ),
    RESULT_ROOT,
)
video_jobs: dict[str, dict] = {}
video_jobs_lock = Lock()

app = FastAPI(title="InsuLens Inspection API", version="1.0.0")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", **runner.detector.model_status()}


def _require_loaded_model() -> None:
    if not runner.detector.ready_for_inspection:
        detail = runner.detector.compatibility_error or "模型未加载或不兼容当前目标检测接口。"
        raise HTTPException(status_code=503, detail=f"无法启动巡检：{detail}")


def _update_video_job(job_id: str, update: dict) -> None:
    with video_jobs_lock:
        video_jobs[job_id].update(update)


def _inspect_video_in_background(job_id: str, source: Path) -> None:
    try:
        report = runner.inspect_video(source, job_id=job_id, on_progress=lambda update: _update_video_job(job_id, update))
        report["download_base"] = f"/api/jobs/{job_id}"
        _update_video_job(job_id, {"status": "completed", "report": report})
    except (OSError, RuntimeError, ValueError) as error:
        _update_video_job(job_id, {"status": "failed", "detail": str(error)})
    finally:
        source.unlink(missing_ok=True)


@app.post("/api/inspect")
def inspect(upload: Annotated[UploadFile, File(...)]) -> dict:
    suffix = Path(upload.filename or "upload").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(status_code=415, detail="仅支持 jpg/png/bmp 图片和 mp4/avi/mov/mkv 视频。")
    _require_loaded_model()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    staging = RESULT_ROOT / f"upload_{os.urandom(6).hex()}{suffix}"
    is_image = suffix in {".jpg", ".jpeg", ".png", ".bmp"}
    try:
        with staging.open("wb") as target:
            shutil.copyfileobj(upload.file, target)
        if not is_image:
            job_id = f"inspection_{uuid4().hex}"
            with video_jobs_lock:
                video_jobs[job_id] = {
                    "job_id": job_id,
                    "status": "queued",
                    "frames_processed": 0,
                    "frames_total": 0,
                    "category_counts": runner.detector.empty_category_counts(),
                    "class_schema": runner.detector.category_schema(),
                }
            Thread(target=_inspect_video_in_background, args=(job_id, staging), daemon=True).start()
            return {
                "job_id": job_id,
                "status": "queued",
                "class_schema": runner.detector.category_schema(),
            }
        report = runner.inspect_image(staging)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        if is_image:
            staging.unlink(missing_ok=True)
    report["download_base"] = f"/api/jobs/{report['job_id']}"
    return report


@app.get("/api/jobs/{job_id}/status")
def job_status(job_id: str) -> dict:
    with video_jobs_lock:
        state = video_jobs.get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="任务不存在或已被清理。")
        return dict(state)


@app.get("/api/jobs/{job_id}/{artifact}")
def download(job_id: str, artifact: str) -> FileResponse:
    if Path(job_id).name != job_id or Path(artifact).name != artifact:
        raise HTTPException(status_code=400, detail="非法文件路径。")
    target = RESULT_ROOT / job_id / artifact
    if not target.is_file():
        raise HTTPException(status_code=404, detail="任务或文件不存在。")
    media_type, _ = mimetypes.guess_type(target.name)
    inline = bool(media_type and (media_type.startswith("video/") or media_type.startswith("image/")))
    return FileResponse(
        target,
        filename=artifact,
        media_type=media_type or "application/octet-stream",
        content_disposition_type="inline" if inline else "attachment",
    )


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="portal")


def main() -> None:
    import uvicorn

    uvicorn.run("insulens_perception.web_app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")), reload=False)


if __name__ == "__main__":
    main()
