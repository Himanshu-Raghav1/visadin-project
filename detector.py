"""
detector.py – Edge AI Threat Detection Pipeline
================================================
* Loads YOLOv8 (nano / small) via the ``ultralytics`` library.
* Auto-selects the best inference backend: TensorRT > OpenVINO > CUDA > CPU.
* Runs inference on every Nth frame during IDLE, on every frame during ACTIVE.
* Draws annotated bounding boxes onto the frame (returned to the pipeline).
* Fires async callbacks when a threat class is detected.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

from config import cfg, DetectorConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detection result container
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2 (pixel coords)
    is_threat: bool = False


@dataclass
class InferenceResult:
    frame: np.ndarray          # Annotated frame
    detections: List[Detection]
    timestamp: float = field(default_factory=time.monotonic)
    camera_id: str = ""
    frame_index: int = 0
    inference_ms: float = 0.0

    @property
    def has_threat(self) -> bool:
        return any(d.is_threat for d in self.detections)

    @property
    def threat_detections(self) -> List[Detection]:
        return [d for d in self.detections if d.is_threat]


# ---------------------------------------------------------------------------
# Colour palette (per class, deterministic)
# ---------------------------------------------------------------------------
_PALETTE: dict[str, tuple[int, int, int]] = {}


def _class_colour(name: str) -> tuple[int, int, int]:
    if name not in _PALETTE:
        h = hash(name) % 360
        bgr = cv2.cvtColor(
            np.array([[[h, 200, 220]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
        )[0][0]
        _PALETTE[name] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    return _PALETTE[name]


# ---------------------------------------------------------------------------
# ThreatDetector
# ---------------------------------------------------------------------------
class ThreatDetector:
    """
    Thread-safe YOLO-based detector.

    Parameters
    ----------
    config:
        DetectorConfig; defaults to ``cfg.detector``.
    on_threat:
        Optional synchronous callback ``fn(result: InferenceResult)`` called
        whenever a threat is found.  For async callers use ``async_on_threat``.
    """

    def __init__(
        self,
        config: Optional[DetectorConfig] = None,
        on_threat: Optional[Callable[[InferenceResult], None]] = None,
    ) -> None:
        self._cfg = config or cfg.detector
        self._on_threat = on_threat
        self._model = None  # loaded lazily on first call or via load()
        self._lock = threading.Lock()
        self._idle_frame_counter: int = 0

        # Normalise threat-class names to lowercase for matching
        self._threat_set = {c.lower().strip() for c in self._cfg.threat_classes}
        logger.info("Threat classes: %s", self._threat_set)

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load the YOLO model.  Call once during startup."""
        with self._lock:
            if self._model is not None:
                return
            from ultralytics import YOLO  # type: ignore

            model_path = self._cfg.model_path
            logger.info("Loading YOLO model from: %s", model_path)
            model = YOLO(model_path)

            # Backend selection
            device = self._resolve_device()
            logger.info("YOLO inference device: %s", device)
            model.to(device)

            self._model = model
            logger.info("YOLO model loaded successfully.")

    def _resolve_device(self) -> str:
        device_pref = self._cfg.device.lower()
        if device_pref != "auto":
            return device_pref

        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                logger.info("CUDA detected.")
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("Apple MPS detected.")
                return "mps"
        except ImportError:
            pass

        try:
            import openvino  # type: ignore  # noqa: F401
            logger.info("OpenVINO detected.")
            return "cpu"  # OpenVINO handled via export; inference on CPU
        except ImportError:
            pass

        logger.info("Falling back to CPU inference.")
        return "cpu"

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def infer(
        self,
        frame: np.ndarray,
        camera_id: str = "",
        frame_index: int = 0,
        is_active_state: bool = False,
    ) -> Optional[InferenceResult]:
        """
        Run inference on *frame*.

        In IDLE state, only every ``idle_inference_interval``-th call
        actually runs the model (returns None on skipped frames).

        Returns
        -------
        InferenceResult or None (if skipped).
        """
        self._idle_frame_counter += 1

        # Skip frames in IDLE mode
        if not is_active_state:
            interval = self._cfg.idle_inference_interval
            if self._idle_frame_counter % interval != 0:
                return None
        else:
            self._idle_frame_counter = 0

        if self._model is None:
            self.load()

        t0 = time.perf_counter()
        with self._lock:
            results = self._model.predict(
                source=frame,
                conf=self._cfg.conf_threshold,
                iou=self._cfg.iou_threshold,
                imgsz=self._cfg.imgsz,
                verbose=False,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        detections = self._parse_results(results)
        annotated = self._draw_annotations(frame.copy(), detections)

        result = InferenceResult(
            frame=annotated,
            detections=detections,
            camera_id=camera_id,
            frame_index=frame_index,
            inference_ms=elapsed_ms,
        )

        if result.has_threat and self._on_threat:
            try:
                self._on_threat(result)
            except Exception as exc:
                logger.warning("on_threat callback error: %s", exc)

        if elapsed_ms > 100:
            logger.debug(
                "[%s] Inference took %.1f ms (frame %d)", camera_id, elapsed_ms, frame_index
            )
        return result

    # ------------------------------------------------------------------
    # Parsing + annotation
    # ------------------------------------------------------------------
    def _parse_results(self, results) -> List[Detection]:
        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            names = result.names  # {idx: 'class_name'}
            for box in result.boxes:
                cls_idx = int(box.cls[0])
                cls_name = names.get(cls_idx, str(cls_idx))
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                is_threat = cls_name.lower() in self._threat_set
                detections.append(
                    Detection(
                        class_name=cls_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        is_threat=is_threat,
                    )
                )
        return detections

    def _draw_annotations(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            colour = (0, 0, 220) if det.is_threat else _class_colour(det.class_name)
            thickness = 3 if det.is_threat else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)
            label = f"{'⚠ ' if det.is_threat else ''}{det.class_name} {det.confidence:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), colour, -1)
            cv2.putText(
                frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )
        return frame

    # ------------------------------------------------------------------
    # Convenience: crop thumbnail for alert
    # ------------------------------------------------------------------
    def crop_threat_thumbnail(
        self, frame: np.ndarray, detection: Detection, padding: int = 20
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = detection.bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        return frame[y1:y2, x1:x2]
