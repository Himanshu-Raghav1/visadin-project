"""
pipeline.py – Per-Camera Processing Pipeline
=============================================
Ties together Stream → Detector → Throttler → Alerter for a single camera.

Each camera gets its own ``CameraPipeline`` which runs a dedicated
processing thread.  The latest annotated frame is exposed via
``get_latest_frame()`` for the MJPEG streaming endpoint.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

from alerter import Alerter
from config import cfg
from detector import InferenceResult, ThreatDetector
from stream_ingest import CircularFrameBuffer, FramePacket, RTSPStreamReader
from throttler import State, ThrottlerEngine

logger = logging.getLogger(__name__)


class CameraPipeline:
    """
    Orchestrates the full processing chain for a single camera.

    Parameters
    ----------
    camera_id:  Friendly camera identifier (e.g. "cam0").
    rtsp_url:   RTSP stream URL.
    detector:   Shared ``ThreatDetector`` (can be shared across cameras).
    alerter:    Shared ``Alerter``.
    loop:       asyncio event loop used by the alerter.
    """

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,           # accepts RTSP URL *or* USB device index string ("0", "1")
        detector: ThreatDetector,
        alerter: Alerter,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.camera_id = camera_id
        self._source = rtsp_url  # kept as rtsp_url param for backward compat
        self._detector = detector
        self._alerter = alerter
        self._loop = loop

        self._reader = RTSPStreamReader(self._source, camera_id)
        self._throttler = ThrottlerEngine(camera_id=camera_id)

        # Latest annotated frame (shared between processing thread & API)
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        # Latest inference result
        self._latest_result: Optional[InferenceResult] = None

        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        self._reader.start()
        self._running.set()
        self._thread = threading.Thread(
            target=self._process_loop,
            daemon=True,
            name=f"pipeline-{self.camera_id}",
        )
        self._thread.start()
        logger.info("Pipeline started for %s", self.camera_id)

    def stop(self) -> None:
        self._running.clear()
        self._reader.stop()
        self._throttler.shutdown()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Pipeline stopped for %s", self.camera_id)

    # ------------------------------------------------------------------
    # Data access (thread-safe)
    # ------------------------------------------------------------------
    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_latest_result(self) -> Optional[InferenceResult]:
        return self._latest_result

    def get_stats(self) -> dict:
        throttler_stats = self._throttler.get_stats()
        reader = self._reader
        # source_label: show "USB:0" for device indices, URL otherwise
        src = self._source
        source_label = f"USB:{src}" if src.isdigit() else src
        return {
            "camera_id": self.camera_id,
            "source": source_label,
            "stream_fps": reader.fps,
            "frame_width": reader.width,
            "frame_height": reader.height,
            **throttler_stats,
        }

    # ------------------------------------------------------------------
    # Dashboard runtime controls (forwarded to throttler / stream)
    # ------------------------------------------------------------------
    def override_fps(self, fps: Optional[int]) -> None:
        self._throttler.override_fps(fps)

    def override_quality(self, quality: Optional[int]) -> None:
        self._throttler.override_quality(quality)

    def force_state(self, state: Optional[str]) -> None:
        if state is None:
            self._throttler.force_state(None)
        else:
            self._throttler.force_state(State(state))

    # ------------------------------------------------------------------
    # Processing loop (daemon thread)
    # ------------------------------------------------------------------
    def _process_loop(self) -> None:
        while self._running.is_set():
            packet: Optional[FramePacket] = self._reader.buffer.get(timeout=1.0)
            if packet is None:
                continue

            is_active = self._throttler.state == State.ACTIVE

            # --- AI Inference (returns None if skipped this frame) ---
            result: Optional[InferenceResult] = self._detector.infer(
                frame=packet.frame,
                camera_id=self.camera_id,
                frame_index=packet.frame_index,
                is_active_state=is_active,
            )

            threat_detected = result.has_threat if result else False

            # --- Feed to throttler / writer ---
            self._throttler.feed(packet, threat_detected=threat_detected)

            # --- Overlay HUD and update latest frame ---
            display_frame = (
                result.frame if result is not None else packet.frame.copy()
            )
            display_frame = self._draw_hud(display_frame, result)

            with self._frame_lock:
                self._latest_frame = display_frame

            self._latest_result = result

            # --- Async alert ---
            if result and result.has_threat:
                self._alerter.fire_and_forget(result, self._loop)

    # ------------------------------------------------------------------
    # HUD overlay
    # ------------------------------------------------------------------
    def _draw_hud(
        self, frame: np.ndarray, result: Optional[InferenceResult]
    ) -> np.ndarray:
        throttler_state = self._throttler.state
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        stats = self._throttler.get_stats()
        fps = stats.get("current_fps", 0)

        # State badge
        state_colour = (0, 200, 0) if throttler_state == State.IDLE else (0, 0, 220)
        state_label = f"● {throttler_state.value.upper()}"
        cv2.putText(
            frame, state_label, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, state_colour, 2, cv2.LINE_AA,
        )

        # Camera ID + timestamp
        cv2.putText(
            frame, f"[{self.camera_id}]  {now_str}", (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )

        # FPS
        cv2.putText(
            frame, f"REC {fps:.0f} FPS", (10, 78),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )

        # Inference time
        if result:
            cv2.putText(
                frame, f"AI {result.inference_ms:.1f}ms", (10, 101),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
            )

        return frame
