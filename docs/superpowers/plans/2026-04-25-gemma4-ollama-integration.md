# Gemma 4 (Ollama) Auto-Labeling Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new auto-labeling model `gemma4_ollama` that performs open-vocabulary rectangle object detection by calling a locally-running Ollama server hosting Gemma 4, mirroring the existing `gemini_api` model's UX exactly.

**Architecture:** A self-contained module (`gemma4_ollama.py`) subclasses `Model` and implements `predict_shapes()` by POSTing the image (PNG, base64) plus a label-detection prompt to `http://localhost:11434/api/generate`. The response JSON's `response` field is parsed as a JSON array of `{box_2d, label, score}` objects with `box_2d` in 0–1000 normalized `[ymin, xmin, ymax, xmax]` form, then converted to absolute pixel rectangles. The parsing/clamping logic is copied verbatim from `gemini_api.py` because the box format is identical.

**Tech Stack:** Python 3, PyQt6, OpenCV (`cv2`), `requests` (HTTP), `concurrent.futures` (timeout pattern), Ollama (`gemma4:26b` default).

**Spec:** `docs/superpowers/specs/2026-04-25-gemma4-ollama-integration-design.md`

---

## File Structure

| Path | Purpose | New / Modified |
| --- | --- | --- |
| `anylabeling/services/auto_labeling/gemma4_ollama.py` | Model class, HTTP client, prompt, parsing, shape conversion | **New** |
| `anylabeling/configs/auto_labeling/gemma4_ollama.yaml` | Default config (model name, endpoint, conf, token budget) | **New** |
| `anylabeling/configs/models.yaml` | Append one entry registering the model | Modified |
| `anylabeling/services/auto_labeling/model_manager.py` | Add `elif type == "gemma4_ollama"` branch (around line 866) | Modified |
| `anylabeling/services/auto_labeling/__init__.py` | Add `"gemma4_ollama"` to `_AUTO_LABELING_CONF_MODELS` | Modified |
| `tests/test_models/test_gemma4_ollama.py` | Unit tests for prompt builder and parsing helpers | **New** |

**Out of scope this iteration** (deferred — YAGNI):
- New UI widget for editing the Ollama endpoint at runtime. Endpoint is read from the YAML config; users who need a non-default endpoint edit the YAML. If a UI is later requested, it will be a follow-up plan.
- i18n catalog updates for new error strings — strings are localized via `self.tr()` and Korean fallback strings, but no new entries are added to `.qm` translation files in this plan.

---

## Pre-flight Checks

Before any task, run from the repo root (`X-AnyLabeling/`):

```bash
git status
```

Expected: clean working tree on the branch you intend to develop on (or the worktree created by brainstorming). If there are uncommitted changes you do not own, stop and ask.

```bash
python -c "import requests; print(requests.__version__)"
```

Expected: a version number printed. If `ModuleNotFoundError`, install: `pip install requests` (it is already a transitive dependency in the PyQt environment, but verifying is cheap).

```bash
ollama --version
```

Expected: a version string (e.g. `ollama version 0.X.Y`). If not installed, the user must install Ollama before manual smoke tests in Task 7. Implementation tasks 1–6 can proceed without Ollama installed.

---

## Task 1: Create the model config YAML

**Files:**
- Create: `anylabeling/configs/auto_labeling/gemma4_ollama.yaml`

- [ ] **Step 1: Create the YAML file with default config**

Create `anylabeling/configs/auto_labeling/gemma4_ollama.yaml` with exactly this content:

```yaml
type: gemma4_ollama
name: gemma4_ollama-r20260425
provider: Google (Local)
display_name: Gemma 4 (Local via Ollama)
model_name: gemma4:26b
ollama_endpoint: http://localhost:11434
conf_threshold: 0.25
iou_threshold: 0.80
request_timeout_seconds: 300
```

- [ ] **Step 2: Verify file is registered in MANIFEST / package data**

Check that the existing build picks up YAML files from `anylabeling/configs/auto_labeling/`:

```bash
grep -r "gemini_api.yaml" "MANIFEST.in" "pyproject.toml" 2>&1 | head -5
```

Expected: either explicit listing of `gemini_api.yaml` (then add `gemma4_ollama.yaml` similarly) or a glob/wildcard for `*.yaml` in that directory (no change needed). If neither, add a glob entry.

If `MANIFEST.in` lists individual YAMLs, add a line near the `gemini_api.yaml` entry:

```
include anylabeling/configs/auto_labeling/gemma4_ollama.yaml
```

- [ ] **Step 3: Commit**

```bash
git add anylabeling/configs/auto_labeling/gemma4_ollama.yaml MANIFEST.in
git commit -m "feat(gemma4): add Ollama-backed Gemma 4 model config"
```

(Drop `MANIFEST.in` from the `git add` if it was not modified.)

---

## Task 2: Create the `Gemma4_Ollama` class skeleton

**Files:**
- Create: `anylabeling/services/auto_labeling/gemma4_ollama.py`

- [ ] **Step 1: Create the module with imports, class skeleton, and copied helper classes**

Create `anylabeling/services/auto_labeling/gemma4_ollama.py` with the following content. Note: `_DialogResponse`, `_normalize_label`, the `_ask_timeout_decision` signal, the `_show_timeout_dialog` slot, and `_ask_user_should_cancel` method are copied verbatim from `gemini_api.py` — the engineer should open `gemini_api.py`, copy those pieces, and adapt only the labels in dialog text. The `predict_shapes` body is filled in Task 4; for now leave it as a stub that returns an empty `AutoLabelingResult` so the import-time wiring (Task 5) can be validated.

```python
import base64
import concurrent.futures
import json
import os
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

    TIMEOUT_SECONDS = 300  # Per-attempt wait before asking the user to cancel.

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
        """Single Ollama HTTP call. Stub - implemented in Task 3."""
        raise NotImplementedError("Implemented in Task 3")

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
        """Stub — implemented in Task 4."""
        return AutoLabelingResult([], replace=self.replace)

    def unload(self):
        """Unload the model. Ollama manages its own lifecycle."""
        pass
```

- [ ] **Step 2: Verify the module imports cleanly**

Run from the repo root:

```bash
python -c "from anylabeling.services.auto_labeling.gemma4_ollama import Gemma4_Ollama; print('OK')"
```

Expected: `OK`. If `ImportError`, fix the path/typo before continuing.

- [ ] **Step 3: Commit**

```bash
git add anylabeling/services/auto_labeling/gemma4_ollama.py
git commit -m "feat(gemma4): add module skeleton with timeout/dialog plumbing"
```

---

## Task 3: Implement the Ollama HTTP call (`_invoke_ollama`)

**Files:**
- Modify: `anylabeling/services/auto_labeling/gemma4_ollama.py` (replace `_invoke_ollama` stub)

- [ ] **Step 1: Replace the `_invoke_ollama` stub with the real implementation**

Open `anylabeling/services/auto_labeling/gemma4_ollama.py` and replace the body of `_invoke_ollama`:

```python
    def _invoke_ollama(self, image_bytes: bytes, prompt: str):
        """Single Ollama HTTP call. Runs inside a worker thread.

        Returns:
            The raw model output string (the contents of the JSON
            envelope's `response` field), or raises on connection /
            HTTP errors so the caller can decide what to do.
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
        # Ollama may return 404 with a body containing "model not found"
        # if the user has not pulled the model. Surface this clearly.
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
```

- [ ] **Step 2: Re-verify import**

```bash
python -c "from anylabeling.services.auto_labeling.gemma4_ollama import Gemma4_Ollama; print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add anylabeling/services/auto_labeling/gemma4_ollama.py
git commit -m "feat(gemma4): implement Ollama HTTP call with friendly errors"
```

---

## Task 4: Implement `predict_shapes` (parse + convert + filter)

**Files:**
- Modify: `anylabeling/services/auto_labeling/gemma4_ollama.py` (replace `predict_shapes` stub)

- [ ] **Step 1: Replace `predict_shapes` with the real implementation**

The body is structurally identical to `Gemini_API.predict_shapes` because the model returns the same `box_2d`/`label`/`score` JSON. The only differences: (1) calling `_invoke_ollama` instead of `_call_with_timeout(... SDK ...)`, (2) the SDK envelope is already unwrapped by `_invoke_ollama` so we work directly with the raw response string.

Replace the stub:

```python
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

        # Defensive: strip markdown fences if model adds them anyway.
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
```

- [ ] **Step 2: Verify import**

```bash
python -c "from anylabeling.services.auto_labeling.gemma4_ollama import Gemma4_Ollama; print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add anylabeling/services/auto_labeling/gemma4_ollama.py
git commit -m "feat(gemma4): implement predict_shapes with bbox JSON parsing"
```

---

## Task 5: Wire up `model_manager.py`

**Files:**
- Modify: `anylabeling/services/auto_labeling/model_manager.py` (add branch around line 866)

- [ ] **Step 1: Add the `gemma4_ollama` dispatch branch**

Find the existing `elif model_config["type"] == "gemini_api":` block in `model_manager.py` (line 866 in the current source). Immediately **after** that block's `return` (line 885), insert the following new `elif` block. The structure mirrors the Gemini branch exactly.

```python
        elif model_config["type"] == "gemma4_ollama":
            from .gemma4_ollama import Gemma4_Ollama

            try:
                model_config["model"] = Gemma4_Ollama(
                    model_config, on_message=self.new_model_status.emit
                )
                self.auto_segmentation_model_unselected.emit()
                logger.info(
                    f"✅ Model loaded successfully: {model_config['type']}"
                )
            except Exception as e:  # noqa
                template = "Error in loading model: {error_message}"
                translated_template = self.tr(template)
                error_text = translated_template.format(error_message=str(e))
                self.new_model_status.emit(error_text)
                logger.error(
                    f"❌ Error in loading model: {model_config['type']} with error: {str(e)}"
                )
                return
```

- [ ] **Step 2: Verify model_manager imports cleanly**

```bash
python -c "from anylabeling.services.auto_labeling.model_manager import ModelManager; print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add anylabeling/services/auto_labeling/model_manager.py
git commit -m "feat(gemma4): register gemma4_ollama in ModelManager dispatch"
```

---

## Task 6: Register capability sets and model registry

**Files:**
- Modify: `anylabeling/services/auto_labeling/__init__.py`
- Modify: `anylabeling/configs/models.yaml`

- [ ] **Step 1: Add `gemma4_ollama` to the conf-threshold capability set**

Open `anylabeling/services/auto_labeling/__init__.py`. Find `_AUTO_LABELING_CONF_MODELS` (around line 186). Add `"gemma4_ollama"` to the list, alphabetically near `"gemini_api"`:

```python
_AUTO_LABELING_CONF_MODELS = [
    "remote_server",
    "upn",
    "damo_yolo",
    "gold_yolo",
    "grounding_dino",
    "grounding_dino_api",
    "gemini_api",
    "gemma4_ollama",   # <-- add this line
    "rtdetr",
    # ... rest unchanged
]
```

Do **not** add `gemma4_ollama` to `_AUTO_LABELING_API_TOKEN_MODELS`. Gemma 4 (local) does not need an API token; the YAML config supplies the endpoint.

- [ ] **Step 2: Append the model entry in `models.yaml`**

Open `anylabeling/configs/models.yaml`. Find the existing `gemini_api-r20260419` entry. Append the following two lines immediately after that entry block:

```yaml
- model_name: "gemma4_ollama-r20260425"
  config_file: ":/gemma4_ollama.yaml"
```

(YAML resource path `:/gemma4_ollama.yaml` resolves to `anylabeling/configs/auto_labeling/gemma4_ollama.yaml` via the project's Qt resource system, mirroring the `gemini_api.yaml` pattern.)

- [ ] **Step 3: Verify the registry loads without YAML error**

```bash
python -c "import yaml; yaml.safe_load(open('anylabeling/configs/models.yaml'))" && echo "models.yaml OK"
python -c "import yaml; yaml.safe_load(open('anylabeling/configs/auto_labeling/gemma4_ollama.yaml'))" && echo "gemma4_ollama.yaml OK"
```

Expected: both lines print `OK`. Any YAML syntax error stops here — fix indentation or quoting.

- [ ] **Step 4: Commit**

```bash
git add anylabeling/services/auto_labeling/__init__.py anylabeling/configs/models.yaml
git commit -m "feat(gemma4): register gemma4_ollama in model registry and capability sets"
```

---

## Task 7: Add unit tests for parsing helpers

The bulk of `gemma4_ollama.py` is verified by manual smoke test (Task 8) because the HTTP call and PyQt UI are integration-bound. But the pure helpers — `_build_prompt` and the box-conversion math — are deterministic and worth a small test pass to catch regressions.

**Files:**
- Create: `tests/test_models/test_gemma4_ollama.py`

- [ ] **Step 1: Write failing tests for the prompt builder and box math**

Create `tests/test_models/test_gemma4_ollama.py`:

```python
"""Unit tests for Gemma4_Ollama static helpers.

The module imports PyQt6 at the top level, so these tests skip
gracefully on a headless environment without PyQt installed.
"""
import pytest

pytest.importorskip("PyQt6")

from anylabeling.services.auto_labeling.gemma4_ollama import Gemma4_Ollama


def test_build_prompt_with_dot_separated_labels():
    prompt = Gemma4_Ollama._build_prompt("helmet . person . fire")
    assert "helmet, person, fire" in prompt
    assert "box_2d" in prompt
    assert "STRICT JSON" in prompt


def test_build_prompt_with_comma_separated_labels():
    prompt = Gemma4_Ollama._build_prompt("car, truck, bicycle")
    assert "car, truck, bicycle" in prompt


def test_build_prompt_strips_whitespace():
    prompt = Gemma4_Ollama._build_prompt("  helmet .   person  ")
    assert "helmet, person" in prompt


def test_build_prompt_empty_string_falls_through():
    # Empty input shouldn't crash; the actual empty-text-prompt rejection
    # happens in predict_shapes before _build_prompt is called.
    prompt = Gemma4_Ollama._build_prompt("")
    assert "Detect every visible instance" in prompt


def test_normalize_label_strips_and_collapses_whitespace():
    assert Gemma4_Ollama._normalize_label("  Hard\tHat  ") == "Hard Hat"
    assert Gemma4_Ollama._normalize_label(None) == ""
    assert Gemma4_Ollama._normalize_label("person") == "person"


def test_normalize_label_handles_non_string():
    assert Gemma4_Ollama._normalize_label(42) == "42"
```

- [ ] **Step 2: Run the tests**

```bash
pytest tests/test_models/test_gemma4_ollama.py -v
```

Expected: all 6 tests pass. If `_build_prompt` or `_normalize_label` are missing or signatures differ, fix Task 2 before continuing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_models/test_gemma4_ollama.py
git commit -m "test(gemma4): unit-test prompt builder and label normalizer"
```

---

## Task 8: Manual smoke test (UI + Ollama)

This task requires a running Ollama server with `gemma4:26b` (or another Gemma 4 tag) pulled. Skip this task if Ollama is not available; the integration is still correct based on Tasks 1–7, and a developer with Ollama can run the smoke test later.

- [ ] **Step 1: Pull the model and start Ollama**

```bash
ollama pull gemma4:26b
ollama serve   # or open the Ollama desktop app
```

Verify Ollama is reachable:

```bash
curl http://localhost:11434/api/tags
```

Expected: JSON with a `models` array including `gemma4:26b`.

- [ ] **Step 2: Launch X-AnyLabeling**

From the repo root:

```bash
python -m anylabeling.app
```

Expected: GUI opens.

- [ ] **Step 3: Verify the model appears**

In the auto-label model dropdown, scroll to find **"Gemma 4 (Local via Ollama)"**. Select it.

Expected: model loads without an error toast. The text input, send button, and confidence input become visible.

- [ ] **Step 4: Run a detection on a real image**

Open any image with people/cars (an example from `assets/` or any test image). In the label-text input, type:

```
person . car
```

Click Send.

Expected within 5–30 seconds (first request loads weights into VRAM):
- Bounding boxes appear over visible people/cars on the canvas.
- Each box has a label and confidence score.
- No exception in the console.

- [ ] **Step 5: Test the failure paths**

A. **Stop Ollama mid-session**, then click Send again.

Expected: status message "Ollama 서버에 연결할 수 없습니다." and no crash.

B. **Restart Ollama**, then in `gemma4_ollama.yaml` change `model_name` to `gemma4:nonexistent-tag`, restart X-AnyLabeling, select Gemma 4, click Send.

Expected: status message "모델이 설치되어 있지 않습니다. 터미널에서 `ollama pull gemma4:nonexistent-tag` 실행 후 다시 시도해주세요."

C. Restore `model_name` to `gemma4:26b` after this test.

- [ ] **Step 6: Test confidence threshold**

Slide the conf input to 0.6. Re-run Send. Expected: fewer boxes (only high-confidence detections).

- [ ] **Step 7: Test "preserve existing annotations"**

Toggle the preserve switch ON. Run Send a second time on the same image. Expected: previous boxes remain; new boxes are added (or overlap).

- [ ] **Step 8: Test other Gemma 4 sizes (optional)**

Stop X-AnyLabeling. Edit `gemma4_ollama.yaml`, set `model_name: gemma4:e4b`. (Pull it first: `ollama pull gemma4:e4b`.) Restart, run a detection. Expected: works, faster, possibly lower quality.

Restore to `gemma4:26b` afterwards.

- [ ] **Step 9: Final regression check — Gemini still works**

Switch back to the existing **Gemini 3.1 Flash Lite (API)** model. Verify it loads and (if a key is configured) runs detections normally. This confirms our additions didn't break the existing path.

- [ ] **Step 10: Commit any docs you wrote during smoke testing**

If you found a meaningful behavior to record (e.g. a specific quirk worth a CHANGELOG line), commit it now:

```bash
git add CHANGELOG.md
git commit -m "docs(gemma4): note Gemma 4 (Ollama) integration in changelog"
```

Otherwise skip this step.

---

## Acceptance Criteria

The integration is complete when **all** of the following are true:

1. `python -c "from anylabeling.services.auto_labeling.gemma4_ollama import Gemma4_Ollama"` succeeds.
2. `pytest tests/test_models/test_gemma4_ollama.py -v` reports 6 passed.
3. Launching X-AnyLabeling and selecting "Gemma 4 (Local via Ollama)" loads the model without errors (with Ollama running and the model pulled).
4. With Ollama running, a detection on a sample image returns bounding boxes drawn on the canvas.
5. With Ollama stopped, the model returns an empty result with a clear "Ollama 서버에 연결할 수 없습니다." message — no crash.
6. With a non-existent `model_name`, the model returns a clear "ollama pull" instruction message — no crash.
7. The existing Gemini API model still loads and works (regression check).

## Rollback

If the integration causes issues, revert with:

```bash
git revert <last-commit-hash>..HEAD
```

There are no destructive migrations or shared-state changes, so a revert fully removes the feature.
