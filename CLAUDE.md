# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 이 저장소는

X-AnyLabeling: PyQt6 기반 데스크톱 이미지/비디오 어노테이션 툴. AI 자동 라벨링 엔진(ONNX Runtime / TensorRT / OpenCV DNN 백엔드, 원격 서버 및 LLM API 모델 포함)과 Ultralytics 학습 패널을 내장한다. Python >= 3.11, GPLv3, 패키지명 `x-anylabeling-cvhub`, 진입점 `xanylabeling = anylabeling.app:main`.

## 명령어

```bash
# 개발 설치 (런타임 extra는 반드시 하나만 선택. onnxruntime과 onnxruntime-gpu는 동시에 설치하면 안 됨)
pip install -e ".[cpu,dev]"        # 또는 [gpu,dev] (CUDA 12), [gpu-cu11,dev], [gpu-cu13,dev]

# 소스에서 앱 실행
python anylabeling/app.py           # 또는: xanylabeling
python anylabeling/app.py --filename /path/to/images --logger-level debug
python anylabeling/app.py checks    # 시스템/패키지 정보 출력
python anylabeling/app.py convert --task yolo2xlabel ...   # 라벨 포맷 변환기 (docs/en/cli.md 참고)

# 테스트 (pytest 설정은 pyproject.toml에 있음. testpaths=tests, --doctest-modules 활성화)
QT_QPA_PLATFORM=offscreen pytest
pytest tests/test_canvas/test_shape_lock.py
pytest tests/test_canvas/test_shape_lock.py::TestCanvasShapeLock::test_lock_state_round_trip
pytest --ignore=tests/test_widgets/test_toolbar_layout.py   # 릴리스 CI가 실행하는 명령

# 포맷 / 린트 (black line-length 79, py311 타겟. tests/ 와 resources.py 는 제외)
bash scripts/format_code.sh         # == black .
flake8                              # 설정은 .flake8 / pyproject
pre-commit run --all-files

# 번역 + Qt 리소스 (self.tr()/translate() 문자열, 아이콘, 모델 YAML을 추가·변경한 뒤 실행)
python scripts/generate_languages.py   # pylupdate6 -> .ts, lrelease -> .qm, .ui는 pyuic6, resources.py 재생성
python scripts/compile_languages.py    # lrelease + resources.py 재생성만 (문자열 재추출 없음)

# 패키징
bash scripts/build_executable.sh {win-cpu|win-gpu|linux-cpu|linux-gpu|macos}   # PyInstaller spec은 packaging/pyinstaller/specs
```

이 Windows 머신에서는 시스템 `python`(3.14, pytest 있음)으로 테스트가 통과한다. 저장소 안의 `venv/`(3.12)에는 PyQt6와 onnxruntime-gpu는 있지만 pytest/black/flake8이 없다. Qt 테스트는 각자 `QT_QPA_PLATFORM=offscreen`을 설정하고 `setUp`에서 `QApplication`을 만든다. `conftest.py`는 없다.

## 아키텍처

### 시작 흐름과 설정
- `anylabeling/app.py`가 CLI 인자를 파싱하고, GUI가 아닌 서브커맨드(`checks`, `convert`, `train-worker` 등)는 지연 import로 처리한 뒤 `QApplication` -> `views/mainwindow.py:MainWindow` -> `LabelingWrapper` -> `views/labeling/label_widget.py:LabelingWidget`을 만든다. 이 LabelingWidget(약 7천 줄)이 캔버스, 독, 메뉴, 모델 매니저, 모든 다이얼로그를 소유하는 허브다.
- `anylabeling/config.py`는 세 계층을 병합한다: 패키지 기본값 `configs/xanylabeling_config.yaml` -> 사용자 파일 `<work-dir>/.xanylabelingrc` -> CLI 인자. 알 수 없는 키는 경고와 함께 버려지므로(`update_dict`) 새 설정 옵션은 패키지 YAML에 먼저 추가해야 한다. `normalize_user_config`가 레거시 키를 마이그레이션한다. work dir 기본값은 `~`이고 `--work-dir`로 바꿀 수 있다. 다운로드된 모델은 `<work-dir>/xanylabeling_data/models/<모델명>/`에 저장된다.
- `anylabeling/app_info.py`에 `__version__`(setuptools가 동적으로 읽음)이 있고, `__preferred_device__`(CPU/GPU)는 `views/common/device_manager.py`를 통해 지연 계산된다.

### 자동 라벨링 모델 시스템 (`anylabeling/services/auto_labeling/`)
- `model.py:Model`이 추상 베이스(`QObject`)다. 서브클래스는 내부 `Meta`(`required_config_names`, `widgets`, `output_modes`, `default_output_mode`)를 정의하고 `predict_shapes(image, filename) -> AutoLabelingResult`와 `unload()`를 구현한다. `get_model_abs_path()`가 로컬 경로를 찾거나 GitHub 릴리스 / ModelScope에서 재시도·취소 지원과 함께 다운로드한다.
- `model_manager.py:ModelManager`가 단일 디스패처다. `_load_model`은 약 100개 분기의 `if/elif model_config["type"] == "..."` 체인이며 분기마다 `from .xxx import Class`를 지연 import한다. 추론은 `worker.py:GenericWorker`를 통해 `QThread`에서 돌고 결과는 Qt 시그널(`new_auto_labeling_result`, `new_model_status` 등)로 돌아온다.
- `__init__.py`에 기능 목록(`_AUTO_LABELING_CONF_MODELS`, `_AUTO_LABELING_MARKS_MODELS`, `_BATCH_PROCESSING_TEXT_PROMPT_MODELS` 등)이 있다. 이 목록에 포함되는지가 모델 타입별 UI 위젯과 매니저 동작을 켜는 기준이다.
- 공용 베이스는 `__base__/`에 있다(`yolo.py:YOLO`가 `trackers/`를 이용한 트래킹까지 대부분의 YOLO 변형을 담당, `sam*.py`, `grounding_dino.py`, `clip.py` 등). 백엔드는 `engines/`(`OnnxBaseModel`, TensorRT, DNN). 후처리 헬퍼는 `utils/`(NMS, letterbox, SAHI 슬라이싱, PPOCR 유틸).
- 모델 레지스트리: `configs/models.yaml`이 `model_name` + `config_file: ":/foo.yaml"`을 나열한다. `:/` 접두사는 `configs/auto_labeling/` 패키지 디렉터리에서 YAML을 읽는다는 뜻이다. 각 YAML은 최소 `type`, `name`, `display_name`이 필요하다. 사용자가 추가한 커스텀 모델은 `.xanylabelingrc`의 `custom_models`에 저장되고 이름에 `_custom_` 접두사가 붙는다.

**새 모델 타입을 추가할 때 손대야 하는 곳:** `configs/auto_labeling/<name>.yaml`, `configs/models.yaml` 항목, `services/auto_labeling/<name>.py`의 `Model` 서브클래스, `ModelManager._load_model`의 새 `elif` 분기, `services/auto_labeling/__init__.py`의 해당 목록. 예제는 `docs/en/custom_model.md`에 있다. 테스트는 `tests/test_models/` 아래에 추가한다.

### 라벨링 뷰 (`anylabeling/views/labeling/`)
- `shape.py:Shape`가 어노테이션 기본 단위다(polygon, rectangle, rotation, circle, line, linestrip, point, cuboid 등). `to_dict`/`load_from_dict`를 가진다. `label_file.py:LabelFile`이 이미지 옆의 JSON 사이드카를 읽고 쓴다. `label_converter.py`가 COCO/VOC/YOLO/DOTA/MOT/PPOCR 등 import/export를 구현하며 GUI와 `convert` CLI 양쪽이 사용한다.
- `widgets/canvas.py:Canvas`(약 5천 줄)가 모든 그리기, 선택, 편집, 자동 라벨링 마크(SAM 계열 모델에 넘기는 점/사각형)를 처리한다. `widgets/auto_labeling/`은 모델 툴바다(`.ui` 파일은 `generate_languages.py`가 컴파일).
- 기능 패널은 서브패키지다: `chatbot/`, `vqa/`, `classifier/`, `video_classifier/`, `ppocr/`, `settings/`. 각각 `widgets/` 아래에 다이얼로그/위젯이 있고 보통 전용 테스트 디렉터리가 있다.
- `utils/`에는 Qt 헬퍼, 테마/스타일, EXIF, 시각화 내보내기, 배치 처리가 있다.

### 학습 (`services/auto_training/ultralytics/` + `views/training/`)
학습은 숨겨진 `xanylabeling train-worker --payload ...` 서브커맨드로 별도 프로세스에서 실행된다. 진행 상황은 `__XANYLABELING_TRAIN_EVENT__=` 접두사가 붙은 stdout 줄로 스트리밍되고 `trainer.py:TrainingManager`가 파싱한다.

### 리소스와 다국어
`anylabeling/resources/resources.py`는 생성된 PyQt6 리소스 모듈(아이콘, `.qm` 번역)이다. 직접 편집하지 말고 위 스크립트로 재생성한다. UI 문자열은 `self.tr(...)` 또는 `QCoreApplication.translate("Model", ...)`를 쓴다. 지원 로케일: `en_US`, `zh_CN`, `ja_JP`, `ko_KR`.

## 규약

- 새 함수/클래스에는 Google 스타일 docstring과 타입 힌트가 필요하다(CONTRIBUTING.md).
- 커밋 메시지는 `<이모지> <type>(<scope>): <요약>` 형식을 따른다(예: `🐛 fix(canvas): ...`, `🚀 feat(sam2): ...`).
- 모델의 `type`과 `name` 필드는 `configs/auto_labeling/` 전체에서 유일해야 한다. `name`은 관례적으로 `-rYYYYMMDD`로 끝난다.
- 모델 서브클래스는 들어온 `QImage`를 `qt_img_to_rgb_cv_img`로 변환하고, 추론 실패 시 예외를 던지지 않고 로그를 남긴 뒤 반환한다. 기존 도형을 유지하려면 `AutoLabelingResult([], replace=False)`(SAM 계열 패턴)를 반환하는 편이 좋다.
- `docs/en/`과 `docs/zh_cn/`은 함께 갱신한다. `docs/**` 또는 `examples/**`를 건드리는 push는 웹사이트 동기화 워크플로를 트리거한다.
- `vendor/`(SAM3 소스)와 `venv/`는 추적되지 않는 로컬 부가물이며 패키지의 일부가 아니다.
