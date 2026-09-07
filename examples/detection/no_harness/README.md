# no_harness 자동 라벨링

SafetyVision v2의 `No_Harness` 클래스(ID 10)만 검출하고 결과 라벨을 `no_harness`로 저장하는 설정입니다.
모델의 13개 클래스 순서는 유지하며, 출력만 필터링합니다. 단일 클래스로 재학습한 모델은 아닙니다.

## 설치 및 실행

저장소 루트 폴더에서 다음 명령을 실행하세요. Python 3.11 이상이 필요하며, 다운로드에는 추가 패키지가 필요 없습니다.

```powershell
py -3.12 scripts/download_safetyvision.py
```

약 45 MB의 ONNX 모델을 `weights/safetyvision-v2/best_896.onnx`에 다운로드합니다.
원본 PyTorch 모델도 필요하면 `py -3.12 scripts/download_safetyvision.py --include-pt`를 실행하세요.
다운로드 스크립트는 파일 크기와 SHA-256을 검증하며, 이미 검증된 파일은 다시 받지 않습니다.

1. `X-AnyLabeling 실행.bat`로 앱을 실행합니다.
2. 자동 라벨링 패널에서 사용자 정의 모델을 불러옵니다.
3. 이 폴더의 **`no_harness.yaml`**을 선택합니다.
4. 이미지 몇 장에 먼저 실행해 검출 결과를 확인한 다음 전체 작업에 적용합니다.

YAML의 모델 경로는 이 폴더 기준 상대 경로입니다. 저장소의 폴더 구조를 유지하세요.
초기 신뢰도 임계값은 0.25입니다. 누락·오검출은 직접 검토하고 수정하세요.
기존 라벨을 유지하려면 자동 라벨링 패널의 기존 주석 유지 옵션을 켜세요.

설정과 다운로드 스크립트는 Git으로 공유하고, 모델 가중치는 각 PC에서 다운로드합니다.

## 출처

- 모델: [ayushgupta7777/safetyvision-yolov8](https://huggingface.co/ayushgupta7777/safetyvision-yolov8)
- 고정 리비전: `56a71758b55f0e9f2b4b2d6b51a779a1f882da10`
- 사용 모델: `v2/best_896.onnx` (896×896 입력)
- 게시자 표기 라이선스: AGPL-3.0. 모델 이용 조건은 원본 모델 카드를 확인하세요.
