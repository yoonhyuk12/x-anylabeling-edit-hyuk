# Gemma 4 (Local via Ollama) Integration — Design Spec

- **Date:** 2026-04-25
- **Project:** X-AnyLabeling
- **Author / Requester:** sales@umusun.com
- **Status:** Approved (pending implementation plan)

## Goal

Add Gemma 4 as a new auto-labeling model in X-AnyLabeling, running **locally** via Ollama, providing **open-vocabulary rectangle (bounding-box) object detection** through natural-language label prompts. The user experience should match the existing `gemini_api` model exactly, with the only differences being (a) no API key required and (b) Ollama server endpoint configuration in place of the API token.

## Background

X-AnyLabeling already integrates Google Gemini via the cloud `gemini_api` model (`anylabeling/services/auto_labeling/gemini_api.py`), which sends an image plus a label list (e.g. `"helmet . person . fire"`) and receives JSON bounding boxes in `box_2d` format (`[ymin, xmin, ymax, xmax]` normalized to 0–1000), converted to absolute pixel coordinates.

Gemma 4 (Apache 2.0, released 2026-04-02) is Google DeepMind's open-weight multimodal family with **native bounding-box output** in JSON, multiple sizes (E2B, E4B, 26B MoE, 31B Dense), and a configurable image token budget (70 / 140 / 280 / 560 / 1120). The Ollama library publishes ready-to-run weights at `gemma4:e2b`, `gemma4:e4b`, `gemma4:26b`, etc.

Running locally avoids API costs, removes the need for an internet connection, and keeps user data on-device — important for the labeling workflow.

## Non-Goals

- OCR (text-in-image reading) — handled by the existing PaddleOCR models.
- Pointing / segmentation modes — rectangle output only for this integration.
- Replacing the existing `gemini_api` cloud integration — Gemma is added alongside.
- Bundling Ollama with X-AnyLabeling — users install Ollama themselves (one-time).
- Automated test suites — this matches existing precedent (`gemini_api.py` is verified manually).

## User-Facing Behavior

1. User installs Ollama and runs `ollama pull gemma4:26b` once.
2. In X-AnyLabeling, model dropdown now lists **"Gemma 4 (Local via Ollama)"**.
3. User selects it; the panel exposes the same widgets as `gemini_api`: label-text input, send button, confidence input, preserve-existing toggle.
4. The "Set API Token" button is replaced by an **"Ollama Endpoint"** field (default `http://localhost:11434`).
5. User enters labels (e.g. `helmet . person . fire`), clicks Send, sees rectangles drawn on the canvas.
6. Errors (server down, model not pulled, timeout) surface as clear localized messages — they do not crash the UI.

## Architecture

The integration follows the existing `gemini_api` pattern. Four files are touched; only two are new.

| File | New / Modified | Purpose |
| --- | --- | --- |
| `anylabeling/services/auto_labeling/gemma4_ollama.py` | **New** | Ollama HTTP client, prompt construction, JSON parsing, shape conversion |
| `anylabeling/configs/auto_labeling/gemma4_ollama.yaml` | **New** | Default model config (model name, endpoint, thresholds, token budget) |
| `anylabeling/configs/models.yaml` | Modified | Register the new model in the global model list (1 line added) |
| `anylabeling/services/auto_labeling/model_manager.py` | Modified | Add `elif model_config["type"] == "gemma4_ollama"` branch (mirrors `gemini_api` branch) |

No changes to UI views, base `Model` class, or any existing model file.

### Component: `Gemma4_Ollama` class

Subclass of `anylabeling.services.auto_labeling.model.Model`, mirroring `Gemini_API`.

**Required config keys (`Meta.required_config_names`):**
`type`, `name`, `display_name`, `model_name`, `conf_threshold`

**Widgets (`Meta.widgets`):**
`edit_text`, `button_send`, `input_conf`, `edit_conf`, `toggle_preserve_existing_annotations`, `edit_ollama_endpoint`

The `edit_ollama_endpoint` widget replaces `button_set_api_token` from the Gemini variant. It is a text field, not a modal dialog. If this widget identifier does not already exist in the auto-labeling widget system, the implementation plan will define it (or, as a fallback, reuse `button_set_api_token` semantically retitled — to be decided in the plan based on existing widget code).

**Output mode:** `rectangle` only.

**Public methods (matching `Gemini_API`):**
- `__init__(model_config, on_message)` — read config, init endpoint, no model load (Ollama lazy-loads on first request).
- `set_auto_labeling_conf(value)` — set confidence threshold.
- `set_auto_labeling_preserve_existing_annotations_state(state)` — toggle replace.
- `set_ollama_endpoint(url)` — set HTTP endpoint from UI.
- `predict_shapes(image, image_path=None, text_prompt=None)` — main entry.
- `unload()` — no-op (no in-process model state).

**Internal helpers:**
- `_build_prompt(text_prompt)` — reuse the exact `_build_prompt` body from `Gemini_API` (the `box_2d` 0–1000 contract is identical).
- `_invoke_ollama(image_bytes, prompt)` — single HTTP POST.
- `_call_with_timeout(image_bytes, prompt)` — reuse the existing `concurrent.futures` + `_ask_user_should_cancel` pattern verbatim. The 5-minute per-attempt timeout is appropriate for Gemma 4's first-load case (Ollama loads weights into VRAM on the first request, ~10–30 s on a 5070 Ti) and for slow inference on smaller GPUs.
- `_DialogResponse` and `_show_timeout_dialog` — copied from `gemini_api.py` (kept as private classes inside the module to keep the two files independent).
- `_normalize_label(raw)` — copy from `Gemini_API`.

### Component: Ollama HTTP call

```
POST {ollama_endpoint}/api/generate
Content-Type: application/json
{
  "model": "gemma4:26b",
  "prompt": "<bbox detection prompt from _build_prompt>",
  "images": ["<base64 PNG of input image>"],
  "format": "json",
  "stream": false,
  "options": {
    "temperature": 0.0
  }
}
```

Response (non-streaming) returns a JSON envelope whose `response` field is the model's stringified JSON (a JSON array of `{box_2d, label, score?}` objects). The implementation parses `response` then reuses the existing parsing/clamping/shape-construction code from `Gemini_API.predict_shapes`.

The HTTP request uses the `requests` library (already a transitive dependency in the PyQt environment). Image is encoded as PNG via OpenCV (`cv2.imencode`), then base64-encoded for the `images` array.

The image is sent **before** the text in the prompt structure as recommended by the Gemma 4 multimodal docs. With Ollama's `/api/generate` schema, this is achieved by placing image content in the `images` field while the textual instruction is in `prompt` — Ollama composes them with image-first ordering automatically for vision-capable models.

### Component: Model config (`gemma4_ollama.yaml`)

```yaml
type: gemma4_ollama
name: gemma4_ollama-r20260425
provider: Google (Local)
display_name: Gemma 4 (Local via Ollama)
model_name: gemma4:26b
ollama_endpoint: http://localhost:11434
conf_threshold: 0.25
iou_threshold: 0.80
image_token_budget: 280
```

`model_name` is user-editable and accepts any Ollama tag (`gemma4:e2b`, `gemma4:e4b`, `gemma4:26b`, `gemma4:31b`, or quantized variants like `gemma4:e4b-it-q4_K_M`).

`image_token_budget` is one of `70 / 140 / 280 / 560 / 1120`. Higher = more visual detail, slower inference. `280` is the balanced default. Implementation should pass this through Ollama's `options` payload if Ollama exposes it; otherwise it is logged for future use and applied via prompt-side guidance.

### Component: `models.yaml` registration

Append after the `gemini_api-r20260419` entry:

```yaml
- model_name: "gemma4_ollama-r20260425"
  config_file: ":/gemma4_ollama.yaml"
```

### Component: `model_manager.py` registration

Add an `elif` branch directly after the existing `gemini_api` branch (around line 866 of the current file). The structure mirrors the Gemini branch exactly: import the class, instantiate with `model_config` and `on_message`, emit `auto_segmentation_model_unselected`, log success, catch and report failures.

## Data Flow

```
[User clicks Send]
        |
        v
predict_shapes(image, path, text_prompt)
        |
        v
qt_img_to_rgb_cv_img -> cv2.imencode(".png") -> base64
        |
        v
_build_prompt(text_prompt)   ----------+
        |                              |
        v                              |
_call_with_timeout                     |
   |--> _invoke_ollama (HTTP POST) ----+--> Ollama server
        |                                       |
        |<------- JSON response ----------------+
        v
parse `response` field -> JSON array of detections
        |
        v
filter by conf_threshold; clamp box to image; build Shape per item
        |
        v
AutoLabelingResult(shapes, replace=self.replace)
```

## Error Handling

| Failure | Detection | User-Facing Behavior |
| --- | --- | --- |
| Ollama not running / unreachable | `requests.ConnectionError` | Toast: "Ollama 서버에 연결할 수 없습니다. Ollama 앱을 실행하거나 `ollama serve`를 실행해주세요." Empty result returned. |
| Model not pulled | HTTP 404 with `"model ... not found"` body | Toast: "모델이 설치되어 있지 않습니다. 터미널에서 `ollama pull <model_name>` 실행 후 다시 시도해주세요." |
| Per-request timeout (5 min) | `concurrent.futures.TimeoutError` | Same dialog as Gemini: "{N}분 동안 응답이 없습니다. 분석을 종료할까요?" — Yes cancels, No waits another 5 min. |
| Empty response text | `response.strip() == ""` | Log warning, empty result. |
| JSON parse failure | `json.JSONDecodeError` | Log error with first 300 chars of raw text, empty result. Strip markdown code fences first (reuse existing regex). |
| Malformed box (wrong length, non-numeric) | `len(box) != 4` or `ValueError` | Skip the entry, log warning, continue with remaining detections. |
| Box outside image | clamp via `min/max` | Drop boxes with width or height < 1 px after clamping. |

The UI never raises an unhandled exception out of `predict_shapes` — failures degrade to an empty `AutoLabelingResult` with a logged + on-message notification.

## Testing & Validation

Manual verification, matching the precedent of `gemini_api.py` (no automated tests exist for that module).

**Pre-conditions**
1. `ollama serve` running (or Ollama desktop app open).
2. `ollama pull gemma4:26b` completed.

**Golden path**
- Load an image with people and vehicles, prompt `person . car`, click Send → boxes appear within ~5–15 s on a 5070 Ti at `image_token_budget=280`.
- Confidence slider at 0.25 → low-confidence detections are filtered out.
- Toggling "preserve existing annotations" preserves prior shapes when re-running.

**Edge cases**
- Empty detection set → no shapes added, no error.
- Model returns text wrapped in ```` ```json ``` ```` fences → parsed correctly.
- 4K image → still produces results; user can switch to `image_token_budget=560` for higher fidelity.
- Box that exceeds image bounds → clamped without crash.
- Switching between `gemma4:e4b` and `gemma4:26b` mid-session via config edit → next request uses the new model (Ollama re-loads weights).

**Failure mode coverage**
- Stop Ollama mid-session → next request shows "서버에 연결할 수 없습니다." message, UI remains responsive.
- Set `model_name` to a tag that isn't pulled → "모델이 설치되어 있지 않습니다." message.
- Disconnect from network → still works (local).

**Regression check**
- Existing `gemini_api` model still works unchanged (no shared code modified).
- All other YOLO/SAM models load and run normally (only the model registry gained one entry).

## Open Questions for Implementation Plan

These are not blockers for this design but should be resolved when writing the plan:

1. Whether `edit_ollama_endpoint` is an existing widget identifier in X-AnyLabeling's auto-labeling widget framework, or whether a new widget needs to be wired in. If the latter is non-trivial, the fallback is to reuse `button_set_api_token` retitled to "Set Ollama Endpoint" — same modal dialog flow, just storing a URL instead of a key.
2. Whether Ollama's `/api/generate` `options` payload accepts `image_token_budget` directly (or an equivalent like `num_ctx` for image tokens). If not, the config field is recorded for future use and image preprocessing handles fidelity.
3. Whether the project requires Korean and English translations for the new error messages (i18n catalog updates) — to be checked against the `docs/en` and `docs/zh_cn` patterns.

## Hardware Reference

Target user environment confirmed for this design: NVIDIA RTX 5070 Ti (16 GB VRAM). With `gemma4:26b` (MoE, ~14 GB at q4 quantization) this fits comfortably and inference uses ~4 B active parameters per token, giving E4B-class latency at much higher quality. Users on lower-VRAM cards switch to `gemma4:e4b` or `gemma4:e2b` via the config.

## References

- Gemma 4 release: https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/
- Gemma 4 on Ollama: https://ollama.com/library/gemma4
- HuggingFace Gemma 4 multimodal blog: https://huggingface.co/blog/gemma4
- Existing X-AnyLabeling Gemini integration: `anylabeling/services/auto_labeling/gemini_api.py`
