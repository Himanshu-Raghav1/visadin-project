"""
throttler.py – Dynamic Storage & FPS Throttling Engine
=======================================================
State machine
-------------
    IDLE  ──(motion or threat)──►  ACTIVE
    ACTIVE ──(timeout with no motion/threat)──►  IDLE

* IDLE  : Save frames at ``idle_fps`` (default 5 FPS) with lower quality.
* ACTIVE: Save frames at ``active_fps`` (default 30 FPS) full-quality MP4.
          The last ``pre_event_buffer_s`` seconds stored in a ring buffer
          are prepended to every event clip so the approach is captured.

The throttler runs in its own daemon thread.  External code calls
``ThrottlerEngine.feed(packet, threat_detected)`` from any thread.

Manual overrides
----------------
The FastAPI layer can call ``override_fps(n)``, ``override_quality(q)``,
or ``set_state(State.ACTIVE)`` directly at runtime so the dashboard can
adjust settings on the fly.
"""
from __future__ import annotations

import collections
import enum
import logging
import threading
import time
from pathlib import Path
from typing import Deque, Optional

import cv2
import numpy as np

from config import cfg, ThrottlerConfig
from stream_ingest import FramePacket

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State Machine
# ---------------------------------------------------------------------------
class State(enum.Enum):
    IDLE = "idle"
    ACTIVE = "active"


# ---------------------------------------------------------------------------
# Video Writer helper
# ---------------------------------------------------------------------------
class ChunkWriter:
    """
    Wraps cv2.VideoWriter to write a single MP4 chunk.
    Closes automatically when ``close()`` is called.
    """

    def __init__(
        self,
        path: Path,
        fps: float,
        width: int,
        height: int,
        codec: str,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        self._path = path
        self._frame_count = 0
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter at {path}")
        logger.debug("ChunkWriter opened: %s  (%.0f FPS, %dx%d)", path, fps, width, height)

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)
        self._frame_count += 1

    def close(self) -> Path:
        self._writer.release()
        logger.info(
            "Chunk closed: %s (%d frames)", self._path.name, self._frame_count
        )
        return self._path

    @property
    def path(self) -> Path:
        return self._path


# ---------------------------------------------------------------------------
# Pre-Event Ring Buffer
# ---------------------------------------------------------------------------
class PreEventBuffer:
    """
    Ring buffer that keeps the last ``duration_s`` seconds of raw frames.
    Thread-safe.
    """

    def __init__(self, fps: float, duration_s: float) -> None:
        capacity = max(1, int(fps * duration_s))
        self._buf: Deque[FramePacket] = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()

    def push(self, packet: FramePacket) -> None:
        with self._lock:
            self._buf.append(packet)

    def drain(self) -> list[FramePacket]:
        """Return all buffered packets in order and clear the buffer."""
        with self._lock:
            frames = list(self._buf)
            self._buf.clear()
        return frames

    def resize(self, fps: float, duration_s: float) -> None:
        capacity = max(1, int(fps * duration_s))
        with self._lock:
            new_buf: Deque[FramePacket] = collections.deque(
                list(self._buf)[-capacity:], maxlen=capacity
            )
            self._buf = new_buf


# ---------------------------------------------------------------------------
# Main Engine
# ---------------------------------------------------------------------------
class ThrottlerEngine:
    """
    Consumes FramePackets from the ingestion pipeline and writes timed MP4
    chunks at the appropriate quality / FPS based on the current state.

    Parameters
    ----------
    camera_id:
        Friendly name used for file naming.
    width, height:
        Frame dimensions (learned from the first packet if 0).
    config:
        ThrottlerConfig.
    """

    def __init__(
        self,
        camera_id: str = "cam0",
        width: int = 0,
        height: int = 0,
        config: Optional[ThrottlerConfig] = None,
    ) -> None:
        self.camera_id = camera_id
        self._cfg = config or cfg.throttler
        self._cfg.storage_dir.mkdir(parents=True, exist_ok=True)

        self._state = State.IDLE
        self._state_lock = threading.Lock()

        self._width = width
        self._height = height

        # Motion background subtractor (MOG2)
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=50, detectShadows=False
        )

        # Pre-event ring buffer sized for ACTIVE fps
        self._pre_buf = PreEventBuffer(
            fps=self._cfg.active_fps,
            duration_s=self._cfg.pre_event_buffer_s,
        )

        # Active chunk writer (created on IDLE→ACTIVE transition)
        self._writer: Optional[ChunkWriter] = None
        self._chunk_start: float = time.monotonic()

        # Timestamps for state-switching logic
        self._last_activity_ts: float = 0.0
        self._last_frame_written_ts: float = 0.0

        # Frame counter per state (used for sub-sampling in IDLE)
        self._frame_counter: int = 0

        # ---- Manual overrides (set from dashboard) ----
        self._override_fps: Optional[int] = None
        self._override_quality: Optional[int] = None   # 0–100
        self._override_state: Optional[State] = None

        # Statistics
        self.stats: dict = {
            "state": State.IDLE.value,
            "frames_written_idle": 0,
            "frames_written_active": 0,
            "chunks_saved": 0,
            "current_fps": self._cfg.idle_fps,
        }
        self._stats_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        with self._state_lock:
            return self._state

    def override_fps(self, fps: Optional[int]) -> None:
        """Dashboard override.  Pass None to disable override."""
        self._override_fps = fps
        logger.info("[%s] FPS override set to %s", self.camera_id, fps)

    def override_quality(self, quality: Optional[int]) -> None:
        """Dashboard override for MJPEG quality (0–100).  None = auto."""
        self._override_quality = quality
        logger.info("[%s] Quality override set to %s", self.camera_id, quality)

    def force_state(self, state: Optional[State]) -> None:
        """Force a specific state (None = let engine decide)."""
        self._override_state = state
        logger.info("[%s] State override: %s", self.camera_id, state)

    def get_stats(self) -> dict:
        with self._stats_lock:
            return dict(self.stats)

    # ------------------------------------------------------------------
    # Main entry point – called per frame from the processing thread
    # ------------------------------------------------------------------
    def feed(self, packet: FramePacket, threat_detected: bool = False) -> None:
        """
        Process a single FramePacket.

        * Updates the state machine.
        * Writes frames to the appropriate MP4 chunk.

        This method is thread-safe (called from the AI pipeline thread).
        """
        frame = packet.frame

        # Infer frame dimensions on first call
        if self._width == 0 or self._height == 0:
            self._height, self._width = frame.shape[:2]

        self._frame_counter += 1
        now = time.monotonic()

        # --- Motion Analysis (cheap, every frame) ---
        motion_detected = self._analyse_motion(frame)

        # --- State Transition ---
        new_state = self._compute_state(
            motion=motion_detected, threat=threat_detected, now=now
        )
        old_state = self.state
        if new_state != old_state:
            self._transition(old_state, new_state, now)

        current_state = self.state

        # --- Write to pre-event ring buffer (always) ---
        self._pre_buf.push(packet)

        # --- Determine effective FPS for current state ---
        target_fps = self._effective_fps(current_state)

        # --- Frame dropping for IDLE sub-sampling ---
        if not self._should_write_frame(now, target_fps):
            return

        self._ensure_writer(current_state)
        if self._writer:
            self._writer.write(frame)
            self._last_frame_written_ts = now
            with self._stats_lock:
                if current_state == State.IDLE:
                    self.stats["frames_written_idle"] += 1
                else:
                    self.stats["frames_written_active"] += 1
                self.stats["current_fps"] = target_fps
                self.stats["state"] = current_state.value

        # --- Roll chunk if duration exceeded ---
        if now - self._chunk_start >= self._cfg.chunk_duration_s:
            self._roll_chunk(current_state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyse_motion(self, frame: np.ndarray) -> bool:
        """Return True if foreground pixel count exceeds threshold."""
        fg_mask = self._mog2.apply(frame)
        fg_pixels = int(np.count_nonzero(fg_mask))
        return fg_pixels > self._cfg.motion_pixel_threshold

    def _compute_state(self, motion: bool, threat: bool, now: float) -> State:
        # Manual override takes absolute priority
        if self._override_state is not None:
            return self._override_state

        if motion or threat:
            self._last_activity_ts = now
            return State.ACTIVE

        # Stay ACTIVE until idle timeout elapses
        if self.state == State.ACTIVE:
            if now - self._last_activity_ts < self._cfg.idle_timeout_s:
                return State.ACTIVE
        return State.IDLE

    def _transition(self, old: State, new: State, now: float) -> None:
        with self._state_lock:
            self._state = new
        logger.info(
            "[%s] State: %s → %s", self.camera_id, old.value, new.value
        )
        if new == State.ACTIVE and old == State.IDLE:
            # Flush pre-event buffer into a new ACTIVE chunk
            self._flush_pre_event_to_active()
        elif new == State.IDLE and old == State.ACTIVE:
            self._close_writer()

    def _flush_pre_event_to_active(self) -> None:
        """
        Open a fresh ACTIVE writer and write buffered pre-event frames first,
        so no context is lost.
        """
        self._close_writer()
        path = self._make_path(State.ACTIVE)
        try:
            self._writer = ChunkWriter(
                path,
                fps=self._cfg.active_fps,
                width=self._width,
                height=self._height,
                codec=self._cfg.active_codec,
            )
            pre_frames = self._pre_buf.drain()
            for pkt in pre_frames:
                self._writer.write(pkt.frame)
            logger.info(
                "[%s] Pre-event: wrote %d buffered frames.", self.camera_id, len(pre_frames)
            )
        except Exception as exc:
            logger.error("[%s] Failed to open ACTIVE writer: %s", self.camera_id, exc)
            self._writer = None
        self._chunk_start = time.monotonic()

    def _ensure_writer(self, state: State) -> None:
        """Open a writer if none is open for the current state."""
        if self._writer is not None:
            return
        path = self._make_path(state)
        fps = self._effective_fps(state)
        try:
            self._writer = ChunkWriter(
                path, fps=fps, width=self._width, height=self._height,
                codec=self._cfg.active_codec if state == State.ACTIVE else self._cfg.idle_codec,
            )
            self._chunk_start = time.monotonic()
        except Exception as exc:
            logger.error("[%s] Cannot open writer: %s", self.camera_id, exc)
            self._writer = None

    def _close_writer(self) -> None:
        if self._writer:
            try:
                self._writer.close()
                with self._stats_lock:
                    self.stats["chunks_saved"] += 1
            except Exception as exc:
                logger.warning("Error closing writer: %s", exc)
            self._writer = None

    def _roll_chunk(self, state: State) -> None:
        """Close the current chunk and start a new one."""
        self._close_writer()
        self._ensure_writer(state)

    def _should_write_frame(self, now: float, target_fps: float) -> bool:
        """Throttle writes to match target_fps using wall-clock intervals."""
        if target_fps <= 0:
            return False
        interval = 1.0 / target_fps
        return (now - self._last_frame_written_ts) >= interval

    def _effective_fps(self, state: State) -> float:
        if self._override_fps is not None:
            return float(self._override_fps)
        return float(
            self._cfg.active_fps if state == State.ACTIVE else self._cfg.idle_fps
        )

    def _make_path(self, state: State) -> Path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = f"{self.camera_id}_{state.value}_{ts}.mp4"
        return self._cfg.storage_dir / name

    def shutdown(self) -> None:
        """Clean up – close any open writer."""
        self._close_writer()
        logger.info("[%s] ThrottlerEngine shut down.", self.camera_id)
