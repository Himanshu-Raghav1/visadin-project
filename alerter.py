"""
alerter.py – Alerting & Notification Module
============================================
* Sends asynchronous HTTP webhook payloads (JSON + base64 thumbnail) when a
  threat is detected.
* Implements a per-class cooldown so repeated alerts don't flood downstream
  systems.
* Saves cropped thumbnails to disk for the dashboard to serve.
* Sends SMS via Twilio and/or email when a threat is detected or the manual
  ALARM button is pressed.
* Designed to be awaited inside an asyncio event loop; a helper
  ``fire_and_forget`` allows calling from synchronous code.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import smtplib
import time
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from config import cfg, AlertConfig
from detector import Detection, InferenceResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Async Alerter
# ---------------------------------------------------------------------------
class Alerter:
    """
    Sends webhook alerts and persists thumbnail images for threat detections.

    Usage (async context)::

        alerter = Alerter()
        await alerter.handle_inference_result(result)

    Usage (sync context, e.g. callback)::

        alerter.fire_and_forget(result, loop)
    """

    def __init__(self, config: Optional[AlertConfig] = None) -> None:
        self._cfg = config or cfg.alert
        self._cfg.thumbnail_dir.mkdir(parents=True, exist_ok=True)
        # last alert timestamp per class name
        self._last_alert: Dict[str, float] = {}
        # lazy-initialised aiohttp session
        self._session = None

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------
    async def handle_inference_result(self, result: InferenceResult) -> None:
        """
        Called whenever the detector produces a result.  Iterates through
        threat detections, applies cooldown, saves thumbnail, and fires webhook.
        """
        if not result.has_threat:
            return

        for det in result.threat_detections:
            now = time.time()
            last = self._last_alert.get(det.class_name, 0.0)
            if now - last < self._cfg.alert_cooldown_s:
                logger.debug(
                    "Alert suppressed (cooldown) for class '%s'", det.class_name
                )
                continue

            self._last_alert[det.class_name] = now
            thumbnail_path = self._save_thumbnail(result.frame, det)
            await self._send_webhook(
                result=result,
                detection=det,
                thumbnail_path=thumbnail_path,
            )

    # ------------------------------------------------------------------
    # Synchronous fire-and-forget helper
    # ------------------------------------------------------------------
    def fire_and_forget(
        self, result: InferenceResult, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Submit alert handling as a background task in *loop*."""
        asyncio.run_coroutine_threadsafe(
            self.handle_inference_result(result), loop
        )

    # ------------------------------------------------------------------
    # Thumbnail persistence
    # ------------------------------------------------------------------
    def _save_thumbnail(
        self, frame: np.ndarray, detection: Detection, padding: int = 20
    ) -> Optional[Path]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = detection.bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"{detection.class_name}_{ts}.jpg"
        path = self._cfg.thumbnail_dir / fname
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        logger.debug("Thumbnail saved: %s", path)
        return path

    # ------------------------------------------------------------------
    # Webhook
    # ------------------------------------------------------------------
    async def _send_webhook(
        self,
        result: InferenceResult,
        detection: Detection,
        thumbnail_path: Optional[Path],
    ) -> None:
        url = self._cfg.webhook_url
        if not url:
            logger.info(
                "ALERT [no webhook] – Camera %s | Class: %s | Conf: %.2f",
                result.camera_id, detection.class_name, detection.confidence,
            )
            return

        # Build payload
        payload: dict = {
            "event": "threat_detected",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "camera_id": result.camera_id,
            "frame_index": result.frame_index,
            "threat_class": detection.class_name,
            "confidence": round(detection.confidence, 4),
            "bbox": list(detection.bbox),
            "inference_ms": round(result.inference_ms, 1),
            "thumbnail_b64": None,
        }

        if thumbnail_path and thumbnail_path.exists():
            try:
                with open(thumbnail_path, "rb") as fh:
                    payload["thumbnail_b64"] = base64.b64encode(fh.read()).decode()
            except OSError:
                pass

        try:
            import aiohttp  # type: ignore

            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()

            async with self._session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self._cfg.webhook_timeout_s),
            ) as resp:
                status = resp.status
                if status < 300:
                    logger.info(
                        "Webhook sent → %s [HTTP %d] | %s (%.2f)",
                        url, status, detection.class_name, detection.confidence,
                    )
                else:
                    body = await resp.text()
                    logger.warning("Webhook non-2xx %d: %s", status, body[:200])

        except Exception as exc:
            logger.error("Webhook failed: %s", exc)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # Manual alarm (called directly from dashboard)
    # ------------------------------------------------------------------
    async def manual_alarm(
        self,
        camera_id: str = "unknown",
        note: str = "",
        snapshot: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Triggered by the dashboard TRIGGER ALARM button.
        Sends SMS + email + webhook immediately, bypassing cooldown.
        Returns a status dict.
        """
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        local_ts = time.strftime("%d %b %Y %H:%M:%S")
        location = self._cfg.location_name

        # Save snapshot if provided
        thumb_path: Optional[Path] = None
        if snapshot is not None:
            fname = f"ALARM_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            thumb_path = self._cfg.thumbnail_dir / fname
            cv2.imwrite(str(thumb_path), snapshot, [cv2.IMWRITE_JPEG_QUALITY, 92])
            logger.info("Alarm snapshot saved: %s", thumb_path)

        sms_msg = (
            f"[VISADIN AI ⚠ ALARM]\n"
            f"Location: {location}\n"
            f"Camera: {camera_id}\n"
            f"Time: {local_ts}\n"
            f"{('Note: '+note) if note else 'Manual alarm triggered from dashboard.'}\n"
            f"PLEASE RESPOND IMMEDIATELY."
        )

        sms_ok   = await self._send_sms_all(sms_msg)
        mail_ok  = await asyncio.to_thread(
            self._send_email,
            subject=f"[VISADIN ALARM] {location} – {local_ts}",
            body=sms_msg,
            attachment=thumb_path,
        )
        wh_payload = {
            "event": "manual_alarm",
            "timestamp": ts,
            "camera_id": camera_id,
            "location": location,
            "note": note,
        }
        await self._post_webhook(wh_payload, thumb_path)

        return {"sms": sms_ok, "email": mail_ok, "webhook": bool(self._cfg.webhook_url)}

    # ------------------------------------------------------------------
    # SMS via Twilio
    # ------------------------------------------------------------------
    async def _send_sms_all(self, message: str) -> bool:
        """Send SMS to all configured emergency numbers. Returns True if any succeeded."""
        sid   = self._cfg.twilio_account_sid
        token = self._cfg.twilio_auth_token
        from_ = self._cfg.twilio_from_number
        numbers = [n for n in self._cfg.emergency_numbers if n.strip()]

        if not (sid and token and from_ and numbers):
            logger.info("Twilio not configured – SMS skipped.")
            return False

        results = await asyncio.gather(
            *[asyncio.to_thread(self._send_sms_one, sid, token, from_, to, message)
              for to in numbers],
            return_exceptions=True,
        )
        ok = any(r is True for r in results)
        return ok

    @staticmethod
    def _send_sms_one(sid: str, token: str, from_: str, to: str, body: str) -> bool:
        try:
            from twilio.rest import Client  # type: ignore
            client = Client(sid, token)
            msg = client.messages.create(body=body, from_=from_, to=to)
            logger.info("SMS sent to %s – SID %s", to, msg.sid)
            return True
        except Exception as exc:
            logger.error("SMS to %s failed: %s", to, exc)
            return False

    # ------------------------------------------------------------------
    # Email via SMTP
    # ------------------------------------------------------------------
    def _send_email(
        self,
        subject: str,
        body: str,
        attachment: Optional[Path] = None,
    ) -> bool:
        host = self._cfg.smtp_host
        if not host:
            return False
        try:
            msg = MIMEMultipart()
            msg["From"]    = self._cfg.smtp_user
            msg["To"]      = self._cfg.alert_email_to
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))
            if attachment and attachment.exists():
                with open(attachment, "rb") as f:
                    img = MIMEImage(f.read(), name=attachment.name)
                    msg.attach(img)

            with smtplib.SMTP(host, self._cfg.smtp_port) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.login(self._cfg.smtp_user, self._cfg.smtp_pass)
                smtp.sendmail(
                    self._cfg.smtp_user,
                    self._cfg.alert_email_to,
                    msg.as_string(),
                )
            logger.info("Alert email sent to %s", self._cfg.alert_email_to)
            return True
        except Exception as exc:
            logger.error("Email failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Generic webhook POST (refactored from _send_webhook)
    # ------------------------------------------------------------------
    async def _post_webhook(self, payload: dict, thumb_path: Optional[Path] = None) -> None:
        url = self._cfg.webhook_url
        if not url:
            return
        if thumb_path and thumb_path.exists():
            try:
                with open(thumb_path, "rb") as fh:
                    payload["thumbnail_b64"] = base64.b64encode(fh.read()).decode()
            except OSError:
                pass
        try:
            import aiohttp
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()
            async with self._session.post(
                url, json=payload,
                timeout=aiohttp.ClientTimeout(total=self._cfg.webhook_timeout_s),
            ) as resp:
                if resp.status >= 300:
                    logger.warning("Webhook %d: %s", resp.status, (await resp.text())[:200])
        except Exception as exc:
            logger.error("Webhook failed: %s", exc)
