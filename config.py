"""
config.py – Visadin AI Central Configuration
=============================================
All tuneable parameters are read from environment variables (or a .env file
loaded at startup).  Code throughout the project imports from here; nothing
else reads os.environ directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Try to load a .env file if python-dotenv is available.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # dotenv optional – use real env vars in production


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes")


def _int(key: str, default: int) -> int:
    return int(os.getenv(key, default))


def _float(key: str, default: float) -> float:
    return float(os.getenv(key, default))


def _str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _list(key: str, default: str) -> List[str]:
    raw = os.getenv(key, default)
    return [s.strip() for s in raw.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Stream / Camera
# ---------------------------------------------------------------------------
@dataclass
class StreamConfig:
    # ----------------------------------------------------------------
    # CAMERA_SOURCES – comma-separated list of RTSP URLs OR integer
    # device indices for USB/webcam cameras.
    # Examples:
    #   CAMERA_SOURCES=0                          (USB phone cam index 0)
    #   CAMERA_SOURCES=0,1                        (two USB cameras)
    #   CAMERA_SOURCES=rtsp://192.168.1.10/stream (IP camera)
    #   CAMERA_SOURCES=0,rtsp://192.168.1.10/live (mixed)
    # ----------------------------------------------------------------
    camera_sources: List[str] = field(
        default_factory=lambda: _list(
            "CAMERA_SOURCES",
            "0",   # default: first USB/phone camera on the system
        )
    )
    # Legacy alias kept for backward compatibility
    rtsp_urls: List[str] = field(
        default_factory=lambda: _list(
            "RTSP_URLS",
            "",
        )
    )
    # GStreamer pipeline template – {url} is substituted at runtime.
    # Falls back to plain cv2 if GStreamer is not available.
    gst_pipeline_template: str = field(
        default_factory=lambda: _str(
            "GST_PIPELINE",
            (
                "rtspsrc location={url} latency=80 protocols=tcp ! "
                "rtph264depay ! h264parse ! avdec_h264 ! "
                "videoconvert ! appsink max-buffers=1 drop=true"
            ),
        )
    )
    reconnect_delay_s: float = field(
        default_factory=lambda: _float("RECONNECT_DELAY_S", 3.0)
    )
    buffer_size: int = field(
        default_factory=lambda: _int("FRAME_BUFFER_SIZE", 30)
    )
    # Force OpenCV backend even when GStreamer is detected
    force_opencv_backend: bool = field(
        default_factory=lambda: _bool("FORCE_OPENCV_BACKEND", False)
    )

    def effective_sources(self) -> List[str]:
        """Return the final merged list of camera sources."""
        sources = list(self.camera_sources)
        # Also append any legacy RTSP_URLS entries not already present
        for url in self.rtsp_urls:
            if url and url not in sources:
                sources.append(url)
        return [s for s in sources if s]


# ---------------------------------------------------------------------------
# FPS Throttling / Storage
# ---------------------------------------------------------------------------
@dataclass
class ThrottlerConfig:
    # IDLE-state recording FPS
    idle_fps: int = field(default_factory=lambda: _int("IDLE_FPS", 5))
    # ACTIVE/THREAT-state recording FPS
    active_fps: int = field(default_factory=lambda: _int("ACTIVE_FPS", 30))

    # MOG2 motion threshold (number of foreground pixels to trigger activity)
    motion_pixel_threshold: int = field(
        default_factory=lambda: _int("MOTION_PIXEL_THRESHOLD", 1500)
    )
    # How long (seconds) without motion/threat before reverting to IDLE
    idle_timeout_s: float = field(
        default_factory=lambda: _float("IDLE_TIMEOUT_S", 10.0)
    )
    # Pre-event ring buffer duration (seconds)
    pre_event_buffer_s: float = field(
        default_factory=lambda: _float("PRE_EVENT_BUFFER_S", 5.0)
    )
    # Directory where video chunks are stored
    storage_dir: Path = field(
        default_factory=lambda: Path(_str("STORAGE_DIR", "recordings"))
    )
    # Video codec used for high-bitrate ACTIVE chunks
    active_codec: str = field(
        default_factory=lambda: _str("ACTIVE_CODEC", "mp4v")
    )
    idle_codec: str = field(
        default_factory=lambda: _str("IDLE_CODEC", "mp4v")
    )
    chunk_duration_s: int = field(
        default_factory=lambda: _int("CHUNK_DURATION_S", 60)
    )
    # CRF-equivalent quality: 0=lossless .. 51=worst (only for software encoders)
    active_quality: int = field(
        default_factory=lambda: _int("ACTIVE_QUALITY", 18)
    )
    idle_quality: int = field(
        default_factory=lambda: _int("IDLE_QUALITY", 28)
    )


# ---------------------------------------------------------------------------
# AI Detector
# ---------------------------------------------------------------------------
@dataclass
class DetectorConfig:
    model_path: str = field(
        default_factory=lambda: _str("YOLO_MODEL_PATH", "yolov8n.pt")
    )
    # "auto" | "cpu" | "cuda" | "mps" | "openvino" | "tensorrt"
    device: str = field(default_factory=lambda: _str("YOLO_DEVICE", "auto"))
    # Confidence threshold
    conf_threshold: float = field(
        default_factory=lambda: _float("YOLO_CONF", 0.45)
    )
    # IoU NMS threshold
    iou_threshold: float = field(
        default_factory=lambda: _float("YOLO_IOU", 0.5)
    )
    # Class names to treat as threats (must match your model's class names)
    threat_classes: List[str] = field(
        default_factory=lambda: _list(
            "THREAT_CLASSES",
            "knife,gun,pistol,rifle,weapon,person",
        )
    )
    # Run inference on every Nth frame during IDLE; run on every frame in ACTIVE
    idle_inference_interval: int = field(
        default_factory=lambda: _int("IDLE_INFERENCE_INTERVAL", 5)
    )
    # Image size fed to YOLO
    imgsz: int = field(default_factory=lambda: _int("YOLO_IMGSZ", 640))


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------
@dataclass
class AlertConfig:
    # HTTP webhook URL; leave blank to disable
    webhook_url: str = field(
        default_factory=lambda: _str("WEBHOOK_URL", "")
    )
    webhook_timeout_s: float = field(
        default_factory=lambda: _float("WEBHOOK_TIMEOUT_S", 5.0)
    )
    # Directory to store cropped threat thumbnails
    thumbnail_dir: Path = field(
        default_factory=lambda: Path(_str("THUMBNAIL_DIR", "alerts"))
    )
    # Minimum seconds between repeated alerts for the same class
    alert_cooldown_s: float = field(
        default_factory=lambda: _float("ALERT_COOLDOWN_S", 15.0)
    )

    # ── Twilio SMS (police / emergency notification) ──────────────────
    # Sign up free at https://www.twilio.com/try-twilio
    twilio_account_sid: str = field(
        default_factory=lambda: _str("TWILIO_ACCOUNT_SID", "")
    )
    twilio_auth_token: str = field(
        default_factory=lambda: _str("TWILIO_AUTH_TOKEN", "")
    )
    twilio_from_number: str = field(
        default_factory=lambda: _str("TWILIO_FROM_NUMBER", "")  # +1xxxxxxxxxx
    )
    # Comma-separated list of phone numbers to notify (police / security)
    emergency_numbers: List[str] = field(
        default_factory=lambda: _list("EMERGENCY_NUMBERS", "")
    )

    # ── Email alerts (optional) ───────────────────────────────────────
    smtp_host: str = field(default_factory=lambda: _str("SMTP_HOST", ""))
    smtp_port: int = field(default_factory=lambda: _int("SMTP_PORT", 587))
    smtp_user: str = field(default_factory=lambda: _str("SMTP_USER", ""))
    smtp_pass: str = field(default_factory=lambda: _str("SMTP_PASS", ""))
    alert_email_to: str = field(default_factory=lambda: _str("ALERT_EMAIL_TO", ""))

    # Location info embedded in SMS/email (e.g. "Shop 12, MG Road, Pune")
    location_name: str = field(
        default_factory=lambda: _str("LOCATION_NAME", "Visadin AI Protected Premises")
    )


# ---------------------------------------------------------------------------
# API / Dashboard
# ---------------------------------------------------------------------------
@dataclass
class APIConfig:
    host: str = field(default_factory=lambda: _str("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _int("API_PORT", 8000))
    mjpeg_quality: int = field(
        default_factory=lambda: _int("MJPEG_QUALITY", 80)
    )
    # CORS origins (comma-separated)
    cors_origins: List[str] = field(
        default_factory=lambda: _list("CORS_ORIGINS", "*")
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
@dataclass
class LogConfig:
    level: str = field(default_factory=lambda: _str("LOG_LEVEL", "INFO"))
    log_file: Optional[str] = field(
        default_factory=lambda: _str("LOG_FILE", "") or None
    )


# ---------------------------------------------------------------------------
# Root config object
# ---------------------------------------------------------------------------
@dataclass
class VisadinConfig:
    stream: StreamConfig = field(default_factory=StreamConfig)
    throttler: ThrottlerConfig = field(default_factory=ThrottlerConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    api: APIConfig = field(default_factory=APIConfig)
    log: LogConfig = field(default_factory=LogConfig)


# Singleton – import this from anywhere
cfg = VisadinConfig()
