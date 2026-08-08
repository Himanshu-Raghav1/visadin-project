"""
stream_ingest.py – RTSP Frame Ingestion Module
================================================
Responsibilities
----------------
* Open an RTSP camera feed using a GStreamer pipeline (with an OpenCV fallback).
* Maintain a thread-safe circular frame buffer so the latest frame is always
  ready for downstream consumers without blocking the capture loop.
* Reconnect automatically whenever the stream drops.

Thread model
------------
    CaptureThread  →  CircularFrameBuffer  →  [consumer threads]

The CaptureThread is a daemon thread that runs forever.  It owns the
cv2.VideoCapture object and pushes decoded frames into the buffer at
camera FPS.  Consumers call `buffer.get_latest()` (non-blocking) or
`buffer.get()` (blocking, with timeout) from any thread.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

from config import cfg, StreamConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Circular Frame Buffer
# ---------------------------------------------------------------------------
@dataclass
class FramePacket:
    """A decoded video frame together with metadata."""
    frame: np.ndarray
    timestamp: float = field(default_factory=time.monotonic)
    camera_id: str = ""
    frame_index: int = 0


class CircularFrameBuffer:
    """
    Thread-safe fixed-size ring buffer for FramePackets.

    * Producers call `put()`.  When the buffer is full the oldest frame is
      silently dropped (overwritten) – this ensures real-time behaviour.
    * Consumers call `get_latest()` to read the newest frame without removing
      it, or `get()` to pop the next unread frame (blocking with timeout).
    """

    def __init__(self, maxsize: int = 30) -> None:
        self._buf: Deque[FramePacket] = deque(maxlen=maxsize)
        self._lock = threading.Lock()
        self._new_frame = threading.Event()

    # ------------------------------------------------------------------
    # Producer API
    # ------------------------------------------------------------------
    def put(self, packet: FramePacket) -> None:
        with self._lock:
            self._buf.append(packet)
        self._new_frame.set()

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------
    def get_latest(self) -> Optional[FramePacket]:
        """Return the newest frame without blocking.  Returns None if empty."""
        with self._lock:
            return self._buf[-1] if self._buf else None

    def get(self, timeout: float = 1.0) -> Optional[FramePacket]:
        """
        Block until a new frame arrives (or timeout).
        Returns the frame and clears the event so the next call blocks again.
        """
        if not self._new_frame.wait(timeout):
            return None
        self._new_frame.clear()
        return self.get_latest()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def drain(self) -> list[FramePacket]:
        """Drain all buffered frames (used for pre-event buffer flush)."""
        with self._lock:
            frames = list(self._buf)
            self._buf.clear()
        return frames


# ---------------------------------------------------------------------------
# Stream Reader
# ---------------------------------------------------------------------------
class RTSPStreamReader:
    """
    Opens a camera source – either an RTSP/HTTP URL or a USB/webcam device
    index – via GStreamer (preferred for RTSP) or plain OpenCV, and continuously
    pushes frames into a :class:`CircularFrameBuffer`.

    Parameters
    ----------
    source:     RTSP URL *or* integer device index as a string ("0", "1", …).
    camera_id:  Friendly name / ID used in log messages and FramePackets.
    config:     :class:`StreamConfig` from ``config.py``.
    """

    def __init__(
        self,
        source: str,
        camera_id: str = "cam0",
        config: Optional[StreamConfig] = None,
    ) -> None:
        self.source = source
        self.url = source          # kept for backward compat
        self.camera_id = camera_id
        self._cfg = config or cfg.stream
        self.buffer: CircularFrameBuffer = CircularFrameBuffer(
            maxsize=self._cfg.buffer_size
        )
        # True when source is an integer device index (USB / DroidCam)
        self._is_device: bool = self._source_is_device(source)

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_index: int = 0

        # Expose stream metadata once opened
        self.fps: float = 0.0
        self.width: int = 0
        self.height: int = 0

    @staticmethod
    def _source_is_device(source: str) -> bool:
        """Return True if *source* looks like an integer device index."""
        try:
            int(source)
            return True
        except (ValueError, TypeError):
            return False

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background capture thread (non-blocking)."""
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name=f"capture-{self.camera_id}"
        )
        self._thread.start()
        src_desc = f"device:{self.source}" if self._is_device else self.source
        logger.info("Stream reader started for %s (%s)", self.camera_id, src_desc)

    def stop(self) -> None:
        """Signal the capture thread to stop and release resources."""
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._release()
        logger.info("Stream reader stopped for %s", self.camera_id)

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_capture(self) -> cv2.VideoCapture:
        """Open the camera source.  USB device index takes priority path."""
        # ─── USB / built-in webcam (device index) ───────────────────────────
        if self._is_device:
            device_idx = int(self.source)
            logger.info(
                "[%s] Opening USB/webcam device index %d …",
                self.camera_id, device_idx,
            )
            cap = cv2.VideoCapture(device_idx, cv2.CAP_DSHOW)  # CAP_DSHOW = Windows
            if not cap.isOpened():
                # Fallback: try without backend hint (Linux / macOS)
                cap.release()
                cap = cv2.VideoCapture(device_idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Request higher resolution if available
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                logger.info("[%s] USB camera opened (device %d).", self.camera_id, device_idx)
            return cap

        # ─── RTSP / HTTP URL ───────────────────────────────────────────
        if not self._cfg.force_opencv_backend and self._gst_available():
            pipeline = self._cfg.gst_pipeline_template.format(url=self.source)
            logger.debug("Opening GStreamer pipeline: %s", pipeline)
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                logger.info("[%s] Using GStreamer backend.", self.camera_id)
                return cap
            logger.warning(
                "[%s] GStreamer pipeline failed – falling back to OpenCV.", self.camera_id
            )
            cap.release()

        # OpenCV FFMPEG fallback
        logger.debug("[%s] Opening RTSP via OpenCV: %s", self.camera_id, self.source)
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        # Minimize buffering for low-latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    @staticmethod
    def _gst_available() -> bool:
        """Return True if GStreamer support is compiled into OpenCV."""
        build_info = cv2.getBuildInformation()
        return "GStreamer" in build_info and "YES" in build_info[
            build_info.index("GStreamer"):build_info.index("GStreamer") + 60
        ]

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _read_metadata(self) -> None:
        if self._cap is None:
            return
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            "[%s] Stream metadata: %dx%d @ %.1f FPS",
            self.camera_id, self.width, self.height, self.fps,
        )

    # ------------------------------------------------------------------
    # Capture loop (runs in background thread)
    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        delay = self._cfg.reconnect_delay_s
        while self._running.is_set():
            try:
                self._cap = self._build_capture()
                if not self._cap.isOpened():
                    src = self.source
                    raise IOError(f"Cannot open camera source: {src}")
                self._read_metadata()
                self._read_frames()
            except Exception as exc:
                logger.error(
                    "[%s] Stream error: %s – reconnecting in %.1fs …",
                    self.camera_id, exc, delay,
                )
            finally:
                self._release()

            # For USB cameras, shorter reconnect delay makes sense
            wait_delay = 1.0 if self._is_device else delay
            self._running.wait(timeout=wait_delay)

    def _read_frames(self) -> None:
        assert self._cap is not None
        consecutive_failures = 0
        max_failures = 30  # ~1 second at 30 FPS

        while self._running.is_set():
            ret, frame = self._cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    raise IOError("Too many consecutive read failures – stream lost.")
                time.sleep(0.01)
                continue

            consecutive_failures = 0
            self._frame_index += 1
            packet = FramePacket(
                frame=frame,
                timestamp=time.monotonic(),
                camera_id=self.camera_id,
                frame_index=self._frame_index,
            )
            self.buffer.put(packet)


# ---------------------------------------------------------------------------
# Multi-Camera Manager
# ---------------------------------------------------------------------------
class StreamManager:
    """
    Manages multiple :class:`RTSPStreamReader` instances (one per camera URL).

    Usage::

        manager = StreamManager()
        manager.start_all()
        packet = manager.get_latest("cam0")
        manager.stop_all()
    """

    def __init__(self, config: Optional[StreamConfig] = None) -> None:
        self._cfg = config or cfg.stream
        self._readers: dict[str, RTSPStreamReader] = {}
        sources = self._cfg.effective_sources()
        if not sources:
            logger.warning("No camera sources configured! Add CAMERA_SOURCES to .env")
        for i, src in enumerate(sources):
            cam_id = f"cam{i}"
            self._readers[cam_id] = RTSPStreamReader(src, cam_id, self._cfg)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start_all(self) -> None:
        for reader in self._readers.values():
            reader.start()

    def stop_all(self) -> None:
        for reader in self._readers.values():
            reader.stop()

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------
    def get_latest(self, camera_id: str) -> Optional[FramePacket]:
        reader = self._readers.get(camera_id)
        return reader.buffer.get_latest() if reader else None

    def get_buffer(self, camera_id: str) -> Optional[CircularFrameBuffer]:
        reader = self._readers.get(camera_id)
        return reader.buffer if reader else None

    @property
    def camera_ids(self) -> list[str]:
        return list(self._readers.keys())

    def reader(self, camera_id: str) -> Optional[RTSPStreamReader]:
        return self._readers.get(camera_id)
