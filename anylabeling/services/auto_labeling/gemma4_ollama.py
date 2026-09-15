import base64
import concurrent.futures
import json
import re
import threading

import cv2
import requests
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
        self.cancel = True


class Gemma4_Ollama(Model):
    """Gemma 4 (Apache 2.0) running locally via Ollama.

    Calls Ollama's HTTP /api/generate endpoint with an image plus a
    natural-language label prompt (e.g. "helmet . person . fire") and
    receives bounding boxes in box_2d format ([ymin, xmin, ymax, xmax]
    normalized to 0-1000), converted to absolute pixel coordinates.
    """

    TIMEOUT_SECONDS = 300

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
        ]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        super().__init__(model_config, on_message)

        self.model_name = self.config.get("model_name", "gemma4:26b")
        self.conf_threshold = float(self.config.get("conf_threshold", 0.25))
        self.endpoint = self.config.get(
            "ollama_endpoint", "http://localhost:11434"
        ).rstrip("/")
        timeout_override = self.config.get("request_timeout_seconds")
        if timeout_override:
            try:
                self.TIMEOUT_SECONDS = int(timeout_override)
            except (TypeError, ValueError):
                pass
        self.replace = True

        self._ask_timeout_decision.connect(
            self._show_timeout_dialog,
            type=QtCore.Qt.ConnectionType.QueuedConnection,
        )

    def set_auto_labeling_conf(self, value):
        if value > 0:
            self.conf_threshold = value

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        self.replace = not state

    @QtCore.pyqtSlot(int, object)
    def _show_timeout_dialog(self, elapsed_seconds: int, response) -> None:
        try:
            minutes = max(1, elapsed_seconds // 60)
            text = (
                f"Gemma 4 (Ollama)에서 {minutes}분({elapsed_seconds}초) 동안 "
                "응답이 없습니다.\n\n"
                "분석을 종료할까요?\n\n"
                "  • '예'  → 이 이미지의 분석을 즉시 종료합니다.\n"
                "  • '아니오' → 같은 시간만큼 더 기다립니다."
            )
            reply = QtWidgets.QMessageBox.question(
                None,
                "Gemma 4 응답 없음",
                text,
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.Yes,
            )
            response.cancel = (
                reply == QtWidgets.QMessageBox.StandardButton.Yes
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to show Gemma 4 timeout dialog: {e}")
            response.cancel = True
        finally:
            response.event.set()

    def _ask_user_should_cancel(self, elapsed_seconds: int) -> bool:
        response = _DialogResponse()
        if QtWidgets.QApplication.instance() is None:
            logger.warning(
                "No QApplication available for timeout dialog; cancelling."
            )
            return True
        self._ask_timeout_decision.emit(elapsed_seconds, response)
        if not response.event.wait(timeout=3600):
            logger.warning("Timeout dialog was not answered within 1 hour.")
            return True
        return response.cancel

    @staticmethod
    def _normalize_label(raw):
        if raw is None:
            return ""
        return re.sub(r"\s+", " ", str(raw).strip())

    @staticmethod
    def _build_prompt(text_prompt: str) -> str:
        """Build the JSON-bbox detection prompt from user labels.

        Accepts labels separated by `.` or `,`, e.g.
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

    def _invoke_ollama(self, image_bytes: bytes, prompt: str):
        """Single Ollama HTTP call. Runs inside a worker thread.

        Returns the raw model output string (the contents of the JSON
        envelope's `response` field), or None on connection / 404 /
        non-OK status / non-JSON envelope.
        """
        url = f"{self.endpoint}/api/generate"
        b64_image = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [b64_image],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.0,
            },
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.TIMEOUT_SECONDS)
        except requests.ConnectionError as e:
            logger.error(f"Cannot reach Ollama at {self.endpoint}: {e}")
            try:
                self.on_message(
                    self.tr(
                        "Ollama 서버에 연결할 수 없습니다. "
                        "Ollama 앱을 실행하거나 'ollama serve'를 실행해주세요."
                    )
                )
            except Exception:  # noqa: BLE001
                pass
            return None

        if resp.status_code == 404 and "model" in resp.text.lower():
            logger.error(f"Ollama model not found: {self.model_name}")
            try:
                self.on_message(
                    self.tr(
                        "모델이 설치되어 있지 않습니다. 터미널에서 "
                        f"`ollama pull {self.model_name}` 실행 후 다시 시도해주세요."
                    )
                )
            except Exception:  # noqa: BLE001
                pass
            return None

        if not resp.ok:
            logger.error(
                f"Ollama HTTP {resp.status_code}: {resp.text[:300]}"
            )
            return None

        try:
            envelope = resp.json()
        except ValueError as e:
            logger.error(f"Ollama returned non-JSON envelope: {e}")
            return None

        return envelope.get("response", "")

    def _call_with_timeout(self, image_bytes: bytes, prompt: str):
        """Run the HTTP call with periodic timeouts, prompting the user."""
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(self._invoke_ollama, image_bytes, prompt)
            elapsed = 0
            while True:
                try:
                    return future.result(timeout=self.TIMEOUT_SECONDS)
                except concurrent.futures.TimeoutError:
                    elapsed += self.TIMEOUT_SECONDS
                    logger.warning(
                        f"Gemma 4 (Ollama) silent for {elapsed}s — asking user."
                    )
                    cancel = self._ask_user_should_cancel(elapsed)
                    if cancel:
                        logger.info(
                            "User cancelled Gemma 4 analysis after timeout."
                        )
                        try:
                            self.on_message(
                                self.tr(
                                    "Gemma 4 analysis cancelled (no response)."
                                )
                            )
                        except Exception:  # noqa: BLE001
                            pass
                        return None
                    continue
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        f"Gemma 4 (Ollama) call failed: {e}", exc_info=True
                    )
                    return None
        finally:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)

    def predict_shapes(self, image, image_path=None, text_prompt=None):
        """Run Gemma 4 (Ollama) detection and convert results to Shapes."""
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
            f"Gemma 4 request | model={self.model_name} | "
            f"prompt_labels='{text_prompt}' | conf>={self.conf_threshold}"
        )

        raw_text = self._call_with_timeout(image_bytes, prompt)
        if raw_text is None:
            return AutoLabelingResult([], replace=self.replace)

        raw_text = (raw_text or "").strip()
        if not raw_text:
            logger.warning("Gemma 4 returned empty response.")
            return AutoLabelingResult([], replace=self.replace)

        logger.info(f"Gemma 4 raw response: {raw_text[:1000]}")

        if raw_text.startswith("```"):
            raw_text = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                raw_text,
                flags=re.IGNORECASE,
            ).strip()

        try:
            detections = json.loads(raw_text)
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse Gemma 4 JSON: {e}. Raw: {raw_text[:300]}"
            )
            return AutoLabelingResult([], replace=self.replace)

        logger.info(
            f"Gemma 4 parsed {len(detections) if isinstance(detections, list) else 'non-list'} "
            f"raw detections (before filter)."
        )

        if isinstance(detections, dict):
            for key in ("detections", "objects", "results", "data"):
                if key in detections and isinstance(detections[key], list):
                    detections = detections[key]
                    break
            else:
                detections = [detections]

        if not isinstance(detections, list):
            logger.warning(
                f"Unexpected Gemma 4 response shape: {type(detections)}"
            )
            return AutoLabelingResult([], replace=self.replace)

        # Gemma 4 groups multiple boxes under a single entry in two ways:
        #   shared label: {box_2d: [[y,x,y,x], [y,x,y,x]], label: "person"}
        #   parallel labels:
        #     {box_2d: [[..],[..],[..]], label: ["person","car","person"]}
        # Expand both into one entry per box, with label/score zipped when
        # they are parallel lists of the same length.
        _PARALLEL_KEYS = ("label", "category", "class", "score", "confidence")
        expanded = []
        for det in detections:
            if not isinstance(det, dict):
                continue
            box_field = (
                det.get("box_2d") or det.get("bbox") or det.get("box")
            )
            if (
                isinstance(box_field, list)
                and len(box_field) > 0
                and isinstance(box_field[0], list)
            ):
                n = len(box_field)
                for i, single_box in enumerate(box_field):
                    new_det = {**det, "box_2d": single_box}
                    for key in _PARALLEL_KEYS:
                        v = det.get(key)
                        if isinstance(v, list) and len(v) == n:
                            new_det[key] = v[i]
                    expanded.append(new_det)
            else:
                expanded.append(det)
        detections = expanded

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

        logger.info(f"Gemma 4 returned {len(shapes)} shapes after filtering.")
        return AutoLabelingResult(shapes, replace=self.replace)

    def unload(self):
        """Unload the model. Ollama manages its own lifecycle."""
        pass
