# YOLO 학습 성능 이상: 근본 원인 진단

- GPU: RTX 5070 Ti (nvidia-smi 16303 MiB, torch 17,094,475,776 B = 17.09 GB)
- 모델: yolo26m, imgsz=640, AMP, workers=8, cache=False, Windows 11 WDDM
- Run A: AutoBatch (학습 batch=5), ~50 img/s, nvidia-smi ~4410 MiB
- Run B: batch=32, ~10.9 img/s, nvidia-smi 15855–15918 MiB, reserved 17.2–17.8G

## 1. 단일 근본 원인

**Windows WDDM의 CUDA System Memory Fallback(전용 VRAM 초과분을 시스템 RAM으로 조용히 페이징).**

학습 batch=32가 전용 VRAM을 초과한다. 드라이버가 OOM을 내지 않고 초과 할당을 호스트 RAM으로 폴백한다. 매 스텝마다 GPU MMU 페이지 폴트 → PCIe 페이징이 발생해 학습만 4.6배 느려진다.

핵심 숫자:

| 항목 | 값 | 의미 |
|---|---|---|
| torch `total_memory` | 17.09 GB | 전용 VRAM 상한 |
| ultralytics `GPU_mem` (`memory_reserved()/1e9`) | 17.2–17.8G, **2247/2247 샘플이 17.09G 초과** | CUDA 캐싱 할당자가 물리 VRAM보다 많이 예약 |
| nvidia-smi `memory.used` | 15855–15918 / 16303 MiB (97.3–97.6%) | 전용 VRAM 포화. 남는 ~400 MiB는 컨텍스트/워크스페이스용 |
| 학습 처리량 | 50 → 10.9 img/s (×0.218) | 배치가 6.4배 커졌는데 이미지당 속도는 4.6배 하락 |
| 스텝 시간 | batch 5: 100 ms/iter → batch 32: 2936 ms/iter (×29) | 페이징 절벽 |
| 검증 | 양쪽 모두 ~158 img/s, ~40 s | 동일 디스크/동일 GPU에서 검증은 정상 → 병목은 학습 메모리뿐 |
| 디스크 | idle 98.6–99.9%, queue 0.001–0.036 | I/O 병목 아님 |
| CPU | 20–40% | 데이터로더 병목 아님 |
| 온도/전력 제한 | 54°C, 62–75 W / 300 W, `clocks_event_reasons=0` | 스로틀 아님 |
| 로그 | OOM/예외/경고 없음 | 실패가 아니라 침묵 폴백 |

검증 이터레이션으로 학습 배치를 역산하면 Run A는 확실하다.

- Run A val: 6324 / 633 ≈ 10 → val batch=10 = 2×train → **train batch=5**
- Run B val: 6324 / 99 ≈ 64 → val batch=64 = 2×32

그래서 비교는 batch 5 @ 4410 MiB (VRAM 안에 들어감) vs batch 32 @ reserved 17.5G (VRAM을 넘김)이다.

**다른 가설이 기각되는 이유**

- 디스크/워커: 디스크가 거의 놀고, CPU 20–40%, 검증은 같은 데이터 경로로 158 img/s.
- 열/전력 스로틀: 54°C, 전력 한도의 21%, 클럭 2820/3090 MHz, event reason 0.
- sm_120/블랙웰 커널 버그: 검증이 양쪽 동일하고 Run A 학습도 50 img/s로 정상.
- “큰 배치는 원래 느리다”: 배치가 커지면 이미지당 처리량은 올라가야 한다. 50 → 10.9는 반대 방향이다.
- 실제 OOM: 프로세스가 끝까지 돈다. Windows가 실패한 할당을 시스템 메모리로 메운 것이다.

## 2. SM 99–100% 인데 전력 ~21%, VRAM 대역폭 2%인 이유

이건 연산 포화 신호가 아니라 **페이지 폴트로 스톨된 커널** 신호다.

- `utilization.gpu`는 “지난 1초 동안 커널이 GPU에 상주한 시간 비율”이다. 커널은 끝나지 않고 돌아가므로 99–100%로 찍힌다. ALU/Tensor Core가 바쁜 것과는 무관하다.
- 워프가 디바이스에 없는 페이지를 건드리면 GPU MMU 폴트가 나고, WDDM이 해당 페이지를 시스템 RAM에서 PCIe로 가져온다. 그 동안 Tensor Core/FP 유닛은 게이트된다 → **62–75 W / 300 W**. 유휴에 가까운 전력이 2.8 GHz 부스트 클럭과 같이 나오는 것은 “클럭은 올라가 있으나 연산 유닛은 거의 안 쓰는” 상태다. 전력/열 제한이 아니므로 `clocks_event_reasons=0`과도 맞다.
- `utilization.memory`는 **로컬 VRAM 컨트롤러가 DRAM 트랜잭션을 처리 중인 시간**이다. 데이터가 GDDR이 아니라 호스트 RAM에 있으면 이 카운터는 거의 0이다. 관측값 2%는 가중치/액티베이션이 VRAM에 상주하며 GEMM을 돌리는 학습이 아님을 의미한다. 정상 YOLO-m 640 학습이면 메모리 컨트롤러는 수십 % 이상이어야 한다.
- 호스트 쪽 숫자도 같다. 파이썬 WorkingSet 9.33 GB, 시스템 RAM 27.1/31.1 GB. 초과 GPU 페이지가 프로세스 작업 집합으로 보인다.

즉 SM 100%는 “GPU가 계산 중”이 아니라 “커널이 페이지 폴트를 기다리며 점유 중”이다.

## 3. 검증이 batch 64인데도 빠른 이유

검증은 `model.eval()`, gradient graph 없음, activation을 backward용으로 유지하지 않는다. ultralytics는 검증 배치를 학습 배치의 2배로 잡으므로 Run B 검증은 64다. 그래도 메모리 수요는 학습 batch 32보다 작다.

학습 메모리 ≈ 고정(가중치+옵티마이저+CUDA 컨텍스트) + batch × (forward 활성화 + backward 저장 활성화).
검증 메모리 ≈ 같은 고정분(학습 루프 중이라 옵티마이저 상태는 남아 있음) + batch × forward 활성화만. backward 저장이 없어서 이미지당 활성화가 대략 1/2–1/3이다.

아래 선형 모델(4절)에서 학습 m ≈ 455 MiB/장이다. 검증을 m_fwd ≈ 150–200 MiB/장으로 보면:

- val batch 64: 2136 + 64×180 ≈ 13.7 GB 이하 → 16.3 GiB 안에 들어감
- train batch 32: 2136 + 32×455 ≈ 16.7 GiB → 전용 VRAM 초과

그래서 검증은 VRAM에 상주한 채 158 img/s로 돌고, 학습만 페이징에 빠진다. 검증 속도가 두 런에서 동일한 것이 이 해석의 교차검증이다. 데이터 파이프라인·디스크·PCIe 링크·커널 생성은 공통이고, 차이 나는 축은 학습 backward 메모리뿐이다.

## 4. 권장 배치와 메모리 예측

두 점으로 선형 모델을 닫는다.

- batch 5: nvidia-smi 4410 MiB (폴백 없음, 실제 사용량)
- batch 32: reserved 중앙값 17.5e9 B = 16689 MiB (가상 할당량, 전용 VRAM을 넘김)

```
m = (16689 − 4410) / (32 − 5) = 12279 / 27 = 454.8 MiB/image
M_fixed = 4410 − 5×454.8 = 2136 MiB

M(b) = 2136 + 454.8 × b   (MiB, nvidia-smi 스케일)
GPU_mem(b) ≈ M(b) × 1.048576 / 1000  (ultralytics /1e9 표기)
```

전용 16303 MiB의 80% = 13042 MiB를 상한으로 잡으면 워크스페이스/단편화 헤드룸 ~3 GiB가 남는다.

```
2136 + 454.8 × b ≤ 13042
b ≤ 24.0
```

| batch | 예측 nvidia-smi | 예측 GPU_mem | 판정 |
|---|---|---|---|
| 5 (실측) | 4410 MiB | — | AutoBatch. VRAM의 27%만 사용 |
| 16 | 2136+7277 = **9413 MiB** | **9.87G** | 권장. 2의 거듭제곱, 여유 ~6.7 GiB |
| 24 | 2136+10915 = **13051 MiB** | **13.69G** | 실용 상한. 80% 라인 |
| 32 (실측) | 전용 포화 15855–15918 + sysmem | 17.2–17.8G | 폴백. 사용 금지 |

**권장: `batch=16`.** 예측 nvidia-smi ≈ **9.4 GiB**, ultralytics GPU_mem ≈ **9.9G**. AMP·cudnn workspace 스파이크를 넣어도 16.3 GiB 안에 남는다. 처리량을 더 뽑고 싶으면 24까지 올려 보고, 32는 다시 폴백한다.

참고: AutoBatch가 5를 고른 것은 프로브 중 workspace가 커 보였거나 fraction이 보수적이었기 때문이다. 실측 4410 MiB는 16이 안전하다는 쪽을 지지한다.

## 5. 침묵 저하 대신 OOM으로 드러내는 설정

**드라이버: NVIDIA 제어판 → 3D 설정 관리 → `CUDA - Sysmem Fallback Policy` → `Prefer No Sysmem Fallback`.**

Windows 11 + Game Ready 드라이버(대략 535+)는 전용 VRAM이 가득 차면 `cudaMalloc`을 시스템 메모리로 폴백한다. 기본값이 이 침묵 경로다. `Prefer No Sysmem Fallback`이면 초과 할당이 `cudaErrorMemoryAllocation` / PyTorch OOM으로 바로 실패한다. Run B는 시작 시점에 죽었을 것이고, 10.9 img/s로 2,361초를 쓰지 않았을 것이다.

애플리케이션 측 보조 장치:

```python
torch.cuda.set_per_process_memory_fraction(0.90)  # 16303×0.90 ≈ 14673 MiB에서 PyTorch가 OOM
```

GeForce 5070 Ti는 TCC 모드를 지원하지 않으므로 WDDM을 우회할 수는 없다. 실패를 크게 내는 스위치는 Sysmem Fallback Policy다.
