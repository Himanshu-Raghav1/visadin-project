"""
main.py – Visadin AI Entry Point & FastAPI Dashboard Server
===========================================================
Responsibilities
----------------
* Bootstrap logging.
* Load the YOLO model.
* Start one CameraPipeline per configured RTSP URL.
* Serve the FastAPI dashboard:
    GET  /                       → HTML dashboard UI
    GET  /api/cameras            → list camera IDs & stats
    GET  /api/cameras/{id}/stats → per-camera stats JSON
    GET  /stream/{id}            → MJPEG live video feed
    POST /api/cameras/{id}/control → override FPS / quality / state
    GET  /api/alerts/recent      → list recent alert thumbnails
    GET  /api/alerts/{filename}  → serve thumbnail image
    POST /api/system/shutdown    → graceful shutdown
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# FastAPI + Starlette
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from alerter import Alerter
from config import cfg
from detector import ThreatDetector
from pipeline import CameraPipeline

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def _setup_logging() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if cfg.log.log_file:
        handlers.append(logging.FileHandler(cfg.log.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, cfg.log.level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
pipelines: Dict[str, CameraPipeline] = {}
_detector: Optional[ThreatDetector] = None
_alerter: Optional[Alerter] = None
_main_loop: Optional[asyncio.AbstractEventLoop] = None

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Visadin AI",
    description="Edge AI Security Camera Dashboard",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.api.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class CameraControl(BaseModel):
    fps: Optional[int] = None         # None = auto
    quality: Optional[int] = None     # 0–100, None = auto
    state: Optional[str] = None       # "idle" | "active" | None = auto


class AlarmTrigger(BaseModel):
    camera_id: str = "cam0"           # which camera to grab snapshot from
    note: str = ""                    # operator note
    include_snapshot: bool = True


class CameraAdd(BaseModel):
    # source can be an integer device index ("0", "1") or a full RTSP/HTTP URL
    source: str
    # Optional friendly name; auto-assigned if blank
    camera_id: str = ""


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup() -> None:
    global _detector, _alerter, _main_loop

    _setup_logging()
    logger.info("===================================")
    logger.info("  Visadin AI starting up ...")
    logger.info("===================================")

    _main_loop = asyncio.get_event_loop()
    _alerter = Alerter()
    _detector = ThreatDetector(
        on_threat=lambda r: _alerter.fire_and_forget(r, _main_loop)  # type: ignore[arg-type]
    )

    logger.info("Loading YOLO model ...")
    _detector.load()

    sources = cfg.stream.effective_sources()
    if not sources:
        logger.warning(
            "No camera sources in .env – add CAMERA_SOURCES=0  (USB) "
            "or CAMERA_SOURCES=rtsp://... and restart."
        )
    for i, src in enumerate(sources):
        cam_id = f"cam{i}"
        _start_pipeline(cam_id, src)

    logger.info("Dashboard: http://%s:%d", cfg.api.host, cfg.api.port)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("Shutting down …")
    for pipeline in pipelines.values():
        pipeline.stop()
    if _alerter:
        await _alerter.close()
    logger.info("Visadin AI stopped cleanly.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_pipeline(camera_id: str) -> CameraPipeline:
    if camera_id not in pipelines:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    return pipelines[camera_id]


def _encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, buf = cv2.imencode(".jpg", frame, params)
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def _start_pipeline(cam_id: str, source: str) -> CameraPipeline:
    """Create and start a pipeline; registers it in global ``pipelines``."""
    assert _detector is not None and _alerter is not None and _main_loop is not None
    pipeline = CameraPipeline(
        camera_id=cam_id,
        rtsp_url=source,   # CameraPipeline stores this as the source string
        detector=_detector,
        alerter=_alerter,
        loop=_main_loop,
    )
    pipeline.start()
    pipelines[cam_id] = pipeline
    logger.info("Pipeline started: %s  source=%s", cam_id, source)
    return pipeline


def _next_cam_id() -> str:
    """Return the next unused cam0, cam1, … identifier."""
    i = 0
    while f"cam{i}" in pipelines:
        i += 1
    return f"cam{i}"


# ---------------------------------------------------------------------------
# Routes – Dashboard HTML
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    html_path = Path(__file__).parent / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Dashboard not found – place dashboard.html next to main.py</h1>")


# ---------------------------------------------------------------------------
# Routes – Manual Alarm
# ---------------------------------------------------------------------------
@app.post("/api/alarm/trigger")
async def trigger_alarm(body: AlarmTrigger) -> JSONResponse:
    """Manually trigger a full alarm: SMS + email + webhook + snapshot."""
    if _alerter is None:
        raise HTTPException(status_code=503, detail="Alerter not initialised.")

    snapshot: Optional[np.ndarray] = None
    if body.include_snapshot and body.camera_id in pipelines:
        snapshot = pipelines[body.camera_id].get_latest_frame()

    result = await _alerter.manual_alarm(
        camera_id=body.camera_id,
        note=body.note,
        snapshot=snapshot,
    )
    logger.warning(
        "MANUAL ALARM triggered by dashboard – cam: %s, note: %r",
        body.camera_id, body.note,
    )
    return JSONResponse({"status": "alarm_triggered", "notifications": result})


# ---------------------------------------------------------------------------
# Routes – Camera API
# ---------------------------------------------------------------------------
@app.get("/api/cameras")
async def list_cameras() -> JSONResponse:
    return JSONResponse({
        "cameras": [
            {"camera_id": cid, "stats": p.get_stats()}
            for cid, p in pipelines.items()
        ]
    })


@app.get("/api/cameras/{camera_id}/stats")
async def camera_stats(camera_id: str) -> JSONResponse:
    p = _get_pipeline(camera_id)
    return JSONResponse(p.get_stats())


@app.post("/api/cameras/{camera_id}/control")
async def camera_control(camera_id: str, body: CameraControl) -> JSONResponse:
    p = _get_pipeline(camera_id)
    p.override_fps(body.fps)
    p.override_quality(body.quality)
    p.force_state(body.state)
    return JSONResponse({"status": "ok", "camera_id": camera_id, "applied": body.dict()})


@app.post("/api/cameras/add")
async def add_camera(body: CameraAdd) -> JSONResponse:
    """
    Dynamically add a new camera source **without restarting the server**.

    Body examples::

        {"source": "0"}                           # phone via DroidCam USB (index 0)
        {"source": "1"}                           # second USB camera
        {"source": "rtsp://192.168.1.10/stream"}  # IP camera
        {"source": "0", "camera_id": "phone"}     # custom friendly name
    """
    cam_id = body.camera_id.strip() or _next_cam_id()
    if cam_id in pipelines:
        raise HTTPException(
            status_code=409,
            detail=f"Camera '{cam_id}' already exists. Remove it first or choose a different name.",
        )
    try:
        pipeline = await asyncio.to_thread(_start_pipeline, cam_id, body.source)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start pipeline: {exc}")
    return JSONResponse({
        "status": "ok",
        "camera_id": cam_id,
        "source": body.source,
        "stream_url": f"/stream/{cam_id}",
    })


@app.delete("/api/cameras/{camera_id}")
async def remove_camera(camera_id: str) -> JSONResponse:
    """Stop and remove a camera pipeline (e.g. disconnect phone camera)."""
    if camera_id not in pipelines:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    pipeline = pipelines.pop(camera_id)
    await asyncio.to_thread(pipeline.stop)
    logger.info("Pipeline removed: %s", camera_id)
    return JSONResponse({"status": "removed", "camera_id": camera_id})


@app.get("/api/devices/list")
async def list_devices() -> JSONResponse:
    """
    Scan system for available webcam/USB camera device indices (0-9).
    Used by the dashboard to show which device indices have a live camera.
    """
    def _scan():
        found = []
        for idx in range(10):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                found.append({"index": idx, "label": f"Device {idx}  ({w}×{h})"})
                cap.release()
        return found
    devices = await asyncio.to_thread(_scan)
    return JSONResponse({"devices": devices})


# ---------------------------------------------------------------------------
# Routes – MJPEG Stream
# ---------------------------------------------------------------------------
@app.get("/stream/{camera_id}")
async def mjpeg_stream(camera_id: str):
    p = _get_pipeline(camera_id)
    quality = cfg.api.mjpeg_quality

    async def frame_generator():
        boundary = b"--visadinframe\r\n"
        while True:
            frame = p.get_latest_frame()
            if frame is None:
                await asyncio.sleep(0.05)
                continue
            try:
                jpg = _encode_jpeg(frame, quality)
            except Exception:
                await asyncio.sleep(0.05)
                continue
            header = (
                boundary
                + b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpg)}\r\n\r\n".encode()
            )
            yield header + jpg + b"\r\n"
            # Target ~25 FPS for browser display
            await asyncio.sleep(0.04)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=visadinframe",
    )


# ---------------------------------------------------------------------------
# Routes – Snapshot
# ---------------------------------------------------------------------------
@app.get("/snapshot/{camera_id}")
async def snapshot(camera_id: str) -> Response:
    p = _get_pipeline(camera_id)
    frame = p.get_latest_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available yet.")
    jpg = _encode_jpeg(frame, quality=95)
    return Response(content=jpg, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Routes – Alerts
# ---------------------------------------------------------------------------
@app.get("/api/alerts/recent")
async def recent_alerts(limit: int = 20) -> JSONResponse:
    thumb_dir = cfg.alert.thumbnail_dir
    if not thumb_dir.exists():
        return JSONResponse({"alerts": []})
    files = sorted(thumb_dir.glob("*.jpg"), key=lambda f: f.stat().st_mtime, reverse=True)
    alerts = [
        {
            "filename": f.name,
            "url": f"/api/alerts/{f.name}",
            "mtime": f.stat().st_mtime,
            "timestamp": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(f.stat().st_mtime)
            ),
        }
        for f in files[:limit]
    ]
    return JSONResponse({"alerts": alerts})


@app.get("/api/alerts/{filename}")
async def serve_alert(filename: str) -> Response:
    path = cfg.alert.thumbnail_dir / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Thumbnail not found.")
    return Response(content=path.read_bytes(), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Routes – Recordings
# ---------------------------------------------------------------------------
@app.get("/api/recordings")
async def list_recordings(limit: int = 50) -> JSONResponse:
    rec_dir = cfg.throttler.storage_dir
    if not rec_dir.exists():
        return JSONResponse({"recordings": []})
    files = sorted(rec_dir.glob("*.mp4"), key=lambda f: f.stat().st_mtime, reverse=True)
    recs = [
        {
            "filename": f.name,
            "size_mb": round(f.stat().st_size / 1_048_576, 2),
            "timestamp": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(f.stat().st_mtime)
            ),
        }
        for f in files[:limit]
    ]
    return JSONResponse({"recordings": recs})


# ---------------------------------------------------------------------------
# Routes – System
# ---------------------------------------------------------------------------
@app.post("/api/system/shutdown")
async def system_shutdown() -> JSONResponse:
    logger.warning("Shutdown requested via API.")
    os.kill(os.getpid(), signal.SIGTERM)
    return JSONResponse({"status": "shutting_down"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    _setup_logging()
    uvicorn.run(
        "main:app",
        host=cfg.api.host,
        port=cfg.api.port,
        log_level=cfg.log.level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
