import concurrent.futures
import json
import os
import re
import threading

import cv2
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QCoreApplication

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.logger import logger
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult


class _DialogResponse:
    """Mutable holder used to ferry a popup result back to the worker thread."""

    __slots__ = ("event", "cancel")

    def __init__(self) -> None:
        self.event = threading.Event()
        # Default behaviour if the dialog never fires (no app, app shutdown):
        # treat as cancel so we don't loop forever.
        self.cancel = True


class Gemini_API(Model):
    """Google Gemini API model for open-vocabulary object detection.

    Uses the google-genai SDK to send an image plus a natural language
    text prompt (e.g. "helmet . no-helmet . person . fire") and receives
    bounding boxes in box_2d format ([ymin, xmin, ymax, xmax] normalized
    to 0-1000), which are converted to absolute pixel coordinates.
    """

    # Per-attempt wait before asking the user whether to keep waiting.
    TIMEOUT_SECONDS = 300  # 5 minutes

    # Internal signal: emitted from the worker thread to ask the main
    # thread to display a QMessageBox. Carries the elapsed seconds and a
    # mutable ``_DialogResponse`` holder the slot fills in.
    _ask_timeout_decision = QtCore.pyqtSignal(int, object)

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_name",
            "conf_threshold",
        ]
        widgets = [
            "edit_text",
            "button_send",
            "input_conf",
            "edit_conf",
            "toggle_preserve_existing_annotations",
            "button_set_api_token",
        ]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        super().__init__(model_config, on_message)

        self.model_name = self.config.get(
            "model_name", "gemini-3.1-flash-lite-preview"
        )
        self.conf_threshold = float(self.config.get("conf_threshold", 0.25))
        self.replace = True
        self.api_key = os.getenv("GEMINI_API_KEY", "") or os.getenv(
            "GOOGLE_API_KEY", ""
        )
        self._client = None

        # The instance is constructed on the main (UI) thread by
        # ModelManager. Connect the signal with QueuedConnection so that
        # emitting from a worker thread always dispatches the slot on the
        # main thread, where Qt widgets must live.
        self._ask_timeout_decision.connect(
            self._show_timeout_dialog,
            type=QtCore.Qt.ConnectionType.QueuedConnection,
        )

    def set_auto_labeling_api_token(self, token):
        """Set the API token from the UI."""
        self.api_key = token
        self._client = None

    def set_auto_labeling_conf(self, value):
        """Set confidence threshold (filter low-score boxes)."""
        if value > 0:
            self.conf_threshold = value

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle preserving existing annotations."""
        self.replace = not state

    @QtCore.pyqtSlot(int, object)
    def _show_timeout_dialog(self, elapsed_seconds: int, response) -> None:
        """Slot run on the main UI thread to show the timeout popup."""
        try:
            minutes = max(1, elapsed_seconds // 60)
            text = (
                f"Gemini API에서 {minutes}분({elapsed_seconds}초) 동안 "
                "응답이 없습니다.\n\n"
                "분석을 종료할까요?\n\n"
                "  • '예'  → 이 이미지의 분석을 즉시 종료합니다.\n"
                "  • '아니오' → 같은 시간만큼 더 기다립니다."
            )
            reply = QtWidgets.QMessageBox.question(
                None,
                "Gemini API 응답 없음",
                text,
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.Yes,
            )
            response.cancel = (
                reply == QtWidgets.QMessageBox.StandardButton.Yes
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to show Gemini timeout dialog: {e}")
            response.cancel = True
        finally:
            response.event.set()

    def _ask_user_should_cancel(self, elapsed_seconds: int) -> bool:
        """Block worker thread until the user answers the popup."""
        response = _DialogResponse()
        # If no QApplication exists (e.g. unit test / headless), bail out.
        if QtWidgets.QApplication.instance() is None:
            logger.warning(
                "No QApplication available for timeout dialog; cancelling."
            )
            return True
        self._ask_timeout_decision.emit(elapsed_seconds, response)
        # Cap the user-wait at 1 hour so a forgotten popup can't deadlock
        # any caller that holds a lock.
        if not response.event.wait(timeout=3600):
            logger.warning("Timeout dialog was not answered within 1 hour.")
            return True
        return response.cancel

    def _invoke_sdk(self, image_bytes: bytes, prompt: str):
        """Single SDK call. Runs inside a worker thread."""
        client = self._get_client()
        from google.genai import types as genai_types

        return client.models.generate_content(
            model=self.model_name,
            contents=[
                genai_types.Part.from_bytes(
                    data=image_bytes, mime_type="image/png"
                ),
                prompt,
            ],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )

    def _call_with_timeout(self, image_bytes: bytes, prompt: str):
        """Run the SDK call with periodic timeouts, prompting the user.

        Returns the SDK response object on success, or None on cancel/error.
        """
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(self._invoke_sdk, image_bytes, prompt)
            elapsed = 0
            while True:
                try:
                    return future.result(timeout=self.TIMEOUT_SECONDS)
                except concurrent.futures.TimeoutError:
                    elapsed += self.TIMEOUT_SECONDS
                    logger.warning(
                        f"Gemini API silent for {elapsed}s — asking user."
                    )
                    cancel = self._ask_user_should_cancel(elapsed)
                    if cancel:
                        logger.info(
                            "User chose to cancel Gemini analysis after timeout."
                        )
                        try:
                            self.on_message(
                                self.tr(
                                    "Gemini API analysis cancelled (no response)."
                                )
                            )
                        except Exception:  # noqa: BLE001
                            pass
                        return None
                    # User wants to wait → loop and re-arm timeout.
                    continue
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        f"Gemini API call failed: {e}", exc_info=True
                    )
                    return None
        finally:
            # Don't block on futures still running; the worker thread will
            # finish on its own when (or if) the SDK call completes.
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                # Python < 3.9 fallback (no cancel_futures param).
                executor.shutdown(wait=False)

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise ValueError(
                "Gemini API key is not configured. Set GEMINI_API_KEY env var "
                "or use the UI 'Set API Token' button."
            )
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai SDK is not installed. "
                "Run: pip install google-genai"
            ) from e
        self._client = genai.Client(api_key=self.api_key)
        return self._client

    @staticmethod
    def _normalize_label(raw):
        """Strip surrounding whitespace; collapse internal whitespace."""
        if raw is None:
            return ""
        return re.sub(r"\s+", " ", str(raw).strip())

    @staticmethod
    def _build_prompt(text_prompt: str) -> str:
        """Build the JSON-bbox detection prompt from user labels.

        Accepts labels separated by `.` (DINO-style) or `,`, e.g.
        "helmet . no-helmet . person" or "helmet, no-helmet, person".
        """
        raw_labels = [
            lbl.strip()
            for lbl in re.split(r"[.,]", text_prompt)
            if lbl.strip()
        ]
        labels_csv = ", ".join(raw_labels) if raw_labels else text_prompt
        return (
            "Detect every visible instance of the following object classes "
            f"in the image: {labels_csv}.\n"
            "Return STRICT JSON: a single JSON array. Each element MUST be an "
            'object with exactly these keys: "box_2d" (an array of 4 integers '
            "[ymin, xmin, ymax, xmax] normalized to 0-1000) and \"label\" (one "
            "of the class names listed above, lowercase). Optionally include a "
            '"score" key (0.0-1.0 confidence).\n'
            "Rules:\n"
            "- Only include objects that match one of the listed class names. "
            "Do not invent new classes.\n"
            "- Return one entry per detected instance (no merging).\n"
            "- If a class is not present in the image, simply omit it.\n"
            "- If nothing is detected, return [].\n"
            "- Do NOT wrap the JSON in markdown fences or add commentary."
        )

    def predict_shapes(self, image, image_path=None, text_prompt=None):
        """Run Gemini detection and convert results to X-AnyLabeling Shapes."""
        if image is None:
            logger.warning("Input image is None.")
            return AutoLabelingResult([], replace=self.replace)

        if not text_prompt:
            raise ValueError("Empty text prompt.")

        cv_image = qt_img_to_rgb_cv_img(image, image_path)
        if cv_image is None:
            raise ValueError("Failed to convert input image to OpenCV format.")

        height, width = cv_image.shape[:2]

        is_success, buffer = cv2.imencode(".png", cv_image)
        if not is_success:
            raise ValueError("Failed to encode image.")
        image_bytes = buffer.tobytes()

        prompt = self._build_prompt(text_prompt)
        logger.info(
            f"Gemini request | model={self.model_name} | "
            f"prompt_labels='{text_prompt}' | conf>={self.conf_threshold}"
        )

        response = self._call_with_timeout(image_bytes, prompt)
        if response is None:
            return AutoLabelingResult([], replace=self.replace)

        raw_text = (response.text or "").strip()
        if not raw_text:
            logger.warning("Gemini returned empty response.")
            return AutoLabelingResult([], replace=self.replace)

        # Defensive: strip markdown fences if model adds them anyway.
        if raw_text.startswith("```"):
            raw_text = re.sub(
                r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.IGNORECASE
            ).strip()

        try:
            detections = json.loads(raw_text)
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse Gemini JSON: {e}. Raw: {raw_text[:300]}"
            )
            return AutoLabelingResult([], replace=self.replace)

        if isinstance(detections, dict):
            for key in ("detections", "objects", "results", "data"):
                if key in detections and isinstance(detections[key], list):
                    detections = detections[key]
                    break
            else:
                detections = [detections]

        if not isinstance(detections, list):
            logger.warning(f"Unexpected Gemini response shape: {type(detections)}")
            return AutoLabelingResult([], replace=self.replace)

        shapes = []
        for det in detections:
            if not isinstance(det, dict):
                continue
            box = det.get("box_2d") or det.get("bbox") or det.get("box")
            label = self._normalize_label(
                det.get("label") or det.get("category") or det.get("class")
            )
            score = det.get("score", det.get("confidence", 1.0))

            if not box or not label:
                continue
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 1.0
            if score < self.conf_threshold:
                continue

            try:
                if len(box) != 4:
                    continue
                ymin, xmin, ymax, xmax = [float(v) for v in box]
            except (TypeError, ValueError):
                logger.warning(f"Skipping malformed box: {box}")
                continue

            x1 = int(round(xmin / 1000.0 * width))
            y1 = int(round(ymin / 1000.0 * height))
            x2 = int(round(xmax / 1000.0 * width))
            y2 = int(round(ymax / 1000.0 * height))

            x1, x2 = sorted((max(0, x1), min(width, x2)))
            y1, y2 = sorted((max(0, y1), min(height, y2)))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue

            shape = Shape(
                label=label,
                score=score,
                shape_type="rectangle",
            )
            shape.add_point(QtCore.QPointF(x1, y1))
            shape.add_point(QtCore.QPointF(x2, y1))
            shape.add_point(QtCore.QPointF(x2, y2))
            shape.add_point(QtCore.QPointF(x1, y2))
            shapes.append(shape)

        logger.info(f"Gemini returned {len(shapes)} shapes after filtering.")
        return AutoLabelingResult(shapes, replace=self.replace)

    def unload(self):
        """Unload the model."""
        self._client = None
