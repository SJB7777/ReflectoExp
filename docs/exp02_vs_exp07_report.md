# XRR 딥러닝 파라미터 회귀: exp02 vs exp07 비교 보고서

> 범위: 신경망 단독 성능. GenX 자동 피팅(하이브리드 정련) 파이프라인은 본 보고서 범위에서 제외.
> 대상 코드: `runs/exp02_one_layer/`, `runs/exp07_fourier_hires/` (단, `engine/`, `analyze.py` 제외)

---

## 0. 요약

| 축 | exp02 | exp07 | 개선 |
|---|---|---|---|
| 관측 q 범위 | 0.0071–0.4271 Å⁻¹ | 0.0036–1.0651 Å⁻¹ | 2.53× |
| q 샘플링 | 200 pt | 2000 pt | 10× |
| 분해 가능 두께 구간 | 15.0 – 1489 Å | 5.9 – 5916 Å | 하한 2.5×, 상한 4.0× |
| 학습 두께 범위 | 5 – 200 Å | 10 – 1200 Å | 6.1× |
| 모델 파라미터 | 1.37 M | 5.30 M | 3.87× |
| 유효 학습 샘플/epoch | 700 K | 2.00 M | 2.86× |
| 측정 물리 모델링 | 고정 footprint + Poisson | 4단계 확률적 물리 augmentation | — |
| 실측 데이터 전이 | 불가(그리드 종속 정규화) | 가능(재샘플링 + mask 채널) | — |

**핵심 주장:** exp07은 모델 용량 확대보다 **관측 도메인의 물리적 정합성 확보**가 본질적 개선. exp02는 라벨 범위(5–200 Å)의 하한이 측정 그리드의 이론적 분해 한계(15 Å)를 밑돌아 구조적 오차 하한이 존재.

---

## 1. 측정 도메인 (q-grid) 분석

정의:
- `d_min = 2π / (q_max − q_min)` — 관측 창 안에 Kiessig fringe가 최소 1개 들어오는 두께
- `d_max = π / Δq` — Nyquist 한계, fringe당 최소 2점 샘플링

| 지표 | exp02 | exp07 |
|---|---|---|
| 2θ 범위 | 0.1 – 6.0° | 0.05 – 15.0° |
| 파장 | 1.54 Å | 1.54 Å |
| q_min | 0.00712 Å⁻¹ | 0.00356 Å⁻¹ |
| q_max | 0.42706 Å⁻¹ | 1.06509 Å⁻¹ |
| q span | 0.4199 Å⁻¹ | 1.0615 Å⁻¹ |
| q 포인트 | 200 | 2000 |
| Δq | 2.110 × 10⁻³ | 5.310 × 10⁻⁴ |
| **d_min** | **15.0 Å** | **5.9 Å** |
| **d_max (Nyquist)** | **1489 Å** | **5916 Å** |
| 라벨 두께 범위 | 5 – 200 Å | 10 – 1200 Å |
| 라벨-관측 정합 | **불일치** (5–15 Å 관측 불가) | 정합 (10–1200 ⊂ 5.9–5916) |

**해석:** exp02는 학습 라벨의 약 5 % 구간(5–15 Å)이 물리적으로 복원 불가능한 정보를 회귀하도록 강제됨 → 해당 구간에서 모델이 사전분포 평균으로 회귀, MAE 하한을 형성. exp07은 전 라벨 구간이 관측 가능 영역 내부.

---

## 2. 모델 아키텍처

| 항목 | exp02 `XRR1DRegressor` | exp07 `XRRPhysicsModel` |
|---|---|---|
| 총 파라미터 | 1,371,203 | 5,302,531 |
| encoder | 1,206,592 | 4,907,776 |
| regressor(MLP) | 164,611 | 394,755 |
| fp32 체크포인트 | 5.48 MB | 21.21 MB |
| 입력 | 1ch (log10 R) | 2ch (log10 R, valid-mask) |
| 위치 인코딩 | 없음 | Random Fourier Features 32 freq → 64ch (scale 15.0) |
| encoder 실입력 채널 | 1 | 66 |
| conv 블록 | 4 (k=7, BN, ReLU, Drop, MaxPool2) | 6 (k=7, BN, LeakyReLU 0.2, Drop, MaxPool2) |
| 채널 진행 | 64→128→256→512 | 64→128→256→512→512→512 |
| 수용영역 | 91 px = 0.192 Å⁻¹ | 379 px = 0.201 Å⁻¹ |
| 다운샘플 | 16× (200 → 12) | 64× (2000 → 31) |
| 출력 | 3 (thickness, roughness, SLD) | 3 (동일) |
| 초기화 | Kaiming (fan_out, relu) | Kaiming (fan_out, leaky_relu) |

**해석:** 물리 단위 수용영역은 두 모델이 거의 동일(0.19 vs 0.20 Å⁻¹)이나, exp07은 같은 q 구간을 **4.2배 조밀한 샘플로 관측**. 고주파 fringe(두꺼운 막)를 aliasing 없이 인코딩 가능. Fourier feature는 q 좌표를 명시 입력해 위치 불변 conv의 한계를 보완.

---

## 3. 데이터 생성 및 학습 예산

| 항목 | exp02 | exp07 |
|---|---|---|
| 시뮬레이터 | refnx `ReflectModel` | GenX `spec_nx.Specular` |
| 층 구조 | Air / Film / Substrate(SLD 20) | Air / SurfaceSiO₂ / Film / Si Substrate |
| 은닉 nuisance 변수 | 없음 | SiO₂ 3파라미터 + substrate roughness (랜덤화되나 회귀 대상 아님) |
| 원본 샘플 수 | 1,000,000 | 50,000 |
| split (train/val/test) | 70 / 20 / 10 % | 80 / 10 / 10 % |
| 샘플 수 | 700,000 / 200,000 / 100,000 | 40,000 / 5,000 / 5,000 |
| augmentation | 없음 (생성 시 1회 고정) | on-the-fly, `expand_factor = 50`, `p = 0.9` |
| **유효 학습 샘플/epoch** | 700,000 | **2,000,000** |
| batch size | 128 | 64 |
| optimizer | AdamW, lr 1e-3, wd 1e-5 | AdamW, lr 2e-4, wd 1e-4 |
| scheduler | ReduceLROnPlateau(p=7, ×0.5) | ReduceLROnPlateau(p=7, ×0.5) |
| grad clip | 1.0 | 1.0 |
| epochs (상한) | 50 | 150 |
| **step 수 (상한)** | 273,400 | **4,687,500** |
| early stop patience | 15 | 50 |
| 데이터 로딩 | 전량 RAM (약 800 MB) | h5py lazy + SWMR, `num_workers=4` |
| 체크포인트 | `best.pt` | `best.pt` + `last.pt` (resume 지원) |

**해석:** exp07은 원본 샘플이 1/20이지만 물리 augmentation ×50으로 epoch당 유효 샘플이 2.86배. 동일 원본이 매 epoch 다른 노이즈·footprint·해상도·q-오프셋으로 제시되어 **측정 조건에 대한 불변성**을 학습.

---

## 4. 측정 물리 모델링

### exp02
- refnx `Footprint(sample_length=10.0, beam_height=0.1)` — **고정값**
- `add_noise()` 1회 적용 후 HDF5에 동결
- 결과: 단일 측정 조건에만 특화

### exp07 (`augmentations.py`) — 실제 XRR 측정 체인 순서대로 구현

| 단계 | 물리 현상 | 구현 | 랜덤 범위 |
|---|---|---|---|
| 1 | 기하학적 footprint (빔 넘침) | GenX `GaussIntensity` erf 해석해 | beam_w U(0.005, 0.015) mm, sample_len U(5, 30) mm |
| 2 | 기기 분해능 | `gaussian_filter1d` smearing | σ_q U(1×10⁻⁴, 6×10⁻³) Å⁻¹ |
| 3a | I₀ 변동 | 전역 스케일 | ×U(0.75, 1.25) |
| 3b | 배경 (dark + 산란) | `get_background_noise` | 10^U(−8, −5) |
| 3c | 계수 통계 (shot noise) | Poisson | s = 10^U(5, 8) |
| 4 | q축 정렬 오차 | `np.interp` shift | N(0, 0.004) Å⁻¹ |

시뮬레이션 단계는 `has_noise=False`(clean) — 노이즈는 전부 학습 시점에 부여. 따라서 원본 HDF5 재생성 없이 노이즈 모델 교체 가능.

---

## 5. 전처리 및 실측 데이터 전이 가능성

| 항목 | exp02 (`dataset.py`) | exp07 (`XRRPreprocessor`) |
|---|---|---|
| 강도 정규화 | q점별 z-score `(logR − μ_q)/σ_q` | max 정규화 후 `log10`, 상한 0.0 클리핑 |
| q 그리드 처리 | 없음 (고정 200pt 가정) | cubic spline로 target_q 재샘플링, 실패 시 선형 보간 |
| 관측 범위 밖 | 처리 없음 | −15.0 채움 + **valid-mask 채널로 명시** |
| 라벨 정규화 | z-score, `stats.pt` | z-score, `stats.pt` |
| **임의 q-grid 실측 투입** | **불가** | **가능** |

**해석:** exp02의 per-q z-score는 시뮬레이션 그리드에 종속되어 다른 각도 스텝의 실측 데이터에 적용 불가. exp07은 재샘플링 + 유효성 마스크로 임의 측정 조건의 `.dat`를 그대로 입력 가능하며, 부분 q 범위만 측정된 데이터도 마스크로 처리.

---

## 6. 평가 산출물

| 산출물 | exp02 | exp07 |
|---|---|---|
| MAE, RMSE | ✅ | ✅ |
| MAPE | ❌ | ✅ |
| R² (결정계수) | ❌ | ✅ |
| 오차 히스토그램 | ✅ | ✅ |
| Parity(True vs Pred) 산점도 | ❌ | ✅ |
| 오차 히트맵 (두께 × 거칠기 난이도 맵) | ❌ | ✅ |
| Worst-case top-50 CSV | ❌ | ✅ |
| 곡선 재구성 + residual (worst 3 + random 3) | ❌ | ✅ |
| 전체 결과 CSV export | ❌ | ✅ |
| 학습 곡선 + LR 스케줄 이미지 | ❌ | ✅ (§8.1-2에서 구현) |

exp07의 **오차 히트맵**은 파라미터 조합별 예측 난이도를 정량화 — 보고서에 "어떤 물성 영역에서 모델이 취약한가"를 직접 제시 가능.

---

## 7. 정량 성능 지표 (측정 필요)

> **현재 상태: 학습 산출물 부재로 수치 미확보.**
> - `C:\WorkSpace\05_Resources\Data\XRR_AI\exp07\1230\` → `config.json`, `dataset.h5`(401 MB, 5만 샘플), `qs.npy`, `stats.pt`만 존재. `best.pt` 없음.
> - `D:\03_Resources\Data\XRR_AI\data\one_layer\` → 경로 자체 부재. exp02 데이터·체크포인트 소실.

### 7.1 GPU 학습 블로커 (해결)

증상:
```
torch 2.9.0+cu126  (arch: sm_50 … sm_90)
GPU:  NVIDIA GeForce RTX 5070  →  sm_120 (Blackwell)
결과: torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

`torch.cuda.is_available()`가 `True`를 반환하므로 코드는 GPU 경로로 진입하지만 첫 커널 실행(BatchNorm)에서 실패.
원인은 `pyproject.toml`이 wheel 인덱스를 `cu126`으로 고정한 것 — sm_120은 CUDA 12.8 빌드부터 지원.

조치: 인덱스를 `pytorch-cu128`(`https://download.pytorch.org/whl/cu128`)로 교체 후 `uv sync` + `uv pip install -e .`.

```
torch 2.11.0+cu128
arch  ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']
→ conv1d/BatchNorm forward+backward 정상
```

### 7.3 학습 예산 실측 (RTX 5070, 기준 조건 A_full)

| 항목 | 측정치 |
|---|---|
| GPU 연산 처리량 (batch 64, q 2000, depth 6, Fourier) | 54.5 it/s |
| DataLoader 처리량 (augment on, workers 4) | 76.8 it/s |
| 병목 | **GPU** (augmentation은 병목 아님) |
| steps / epoch | 31,250 |
| 1 epoch | 약 9.6 분 |
| **150 epoch 완주** | **약 24 시간** |

Ablation 5개 variant를 동일 step 예산으로 돌리면 총 약 120 시간. 실무적으로는 `--epochs`로 예산을 축소하거나(예: 30 epoch → variant당 약 4.8 h, 전체 약 24 h) early stopping(patience 50)에 의존할 것.
`training.batch_size` 상향 시 GPU 병목이 완화될 여지 있음 — DataLoader 여유(76.8 vs 54.5 it/s)가 있으므로 batch 128까지는 데이터 공급이 버팀.

재학습 후 채울 표:

| 파라미터 | 지표 | exp02 | exp07 |
|---|---|---|---|
| Thickness (Å) | MAE / RMSE / MAPE / R² | TBD | TBD |
| Roughness (Å) | MAE / RMSE / MAPE / R² | TBD | TBD |
| SLD (10⁻⁶ Å⁻²) | MAE / RMSE / MAPE / R² | TBD | TBD |

**공정 비교 조건 (필수):** 현재 두 실험은 q-grid, 시뮬레이터, 층 구조, 라벨 범위가 전부 상이하여 수치 직접 비교가 성립하지 않음. 따라서 단일 코드베이스(exp07) 위에서 개선 요소를 하나씩 제거하는 ablation으로 대체 (§7.2).

### 7.2 Ablation 설계 — `runs/exp07_fourier_hires/ablation.py`

동일 `dataset.h5` 하나를 5개 조건이 공유. q 해상도는 `XRRPreprocessor`가 원본 2000점 그리드에서 target_q로 보간하므로 **데이터 재생성 불필요**.

| Variant | q_pts | depth | Fourier | augment | expand | params |
|---|---:|---:|---|---|---:|---:|
| `A_full` (기준) | 2000 | 6 | ✅ | ✅ | 50 | 5,302,531 |
| `B_no_fourier` | 2000 | 6 | ❌ | ✅ | 50 | 5,273,859 |
| `C_no_augment` | 2000 | 6 | ✅ | ❌ | 1 | 5,302,531 |
| `D_lowres_200` | 200 | 4 | ✅ | ✅ | 50 | 1,630,467 |
| `E_exp02_like` | 200 | 4 | ❌ | ❌ | 1 | 1,601,795 |

- `A − B` = Fourier feature 기여도
- `A − C` = 물리 augmentation 기여도
- `A − D` = q 해상도 + 모델 깊이 기여도
- `A − E` = exp07 전체 개선폭 (exp02 조건 근사)

**Step 예산 정렬:** augmentation의 `expand_factor` 차이로 variant마다 epoch당 step 수가 다름(A: 31,250 / C: 625). epoch 수를 맞추면 불공정하므로 러너는 기본적으로 기준 variant의 **총 optimizer step 수에 맞춰 각 variant의 epoch를 자동 산정**(`--budget-mode steps`).

실행:
```bash
python runs/exp07_fourier_hires/ablation.py --dry-run      # 계획만 출력
python runs/exp07_fourier_hires/ablation.py                # 전체 실행
python runs/exp07_fourier_hires/ablation.py --only A_full C_no_augment --epochs 5
```

산출물: variant별 `evaluation_results.csv`, `evaluation_correlation.png`, `error_heatmap.png`,
`reconstruction_analysis.png`, `worst_cases_log.csv`, `training_history.png`
+ 통합 `ablation/ablation_summary.csv` (MAE/RMSE/MAPE/R² × 3파라미터, 파라미터 수, 총 step 수).

**검증 상태:** CPU 1-epoch 스모크 테스트로 파이프라인 전 구간(데이터 로드 → 학습 → 평가 → 리포트 8종 생성) 정상 동작 확인. 정식 수치는 §7.1 GPU 블로커 해소 후 산출.

---

## 8. 재현성·정확성 이슈

### 8.1 조치 완료

| # | 위치 | 문제 | 조치 |
|---|---|---|---|
| 1 | `exp07/simulate.py` | config의 SiO₂ 범위를 시뮬레이터에 전달하지 않아 GenX 기본값(10–25, 2–5, 5–22)이 사용됨. 저장된 `config.json`은 (10–30, 1–5, 5–22)로 기록 → 재현성 표 오기 | `param_ranges` → 시뮬레이터 인자 매핑 테이블 추가, `sio2_*` 및 `sub_roughness` 전달 |
| 2 | `exp07/evaluate.py` | history plot 로직이 `pass` → `training_history.png` 미생성 | `save_history_plot()` 구현 (loss 곡선 + best epoch 표시 + LR 스케줄). `evaluate_pipeline`이 metrics 반환하도록 변경 |
| 3 | `exp07/main.py` | `register_debug_hooks()` 상시 활성 → 전 레이어 NaN 검사로 속도 손실 | `CONFIG["debug"]["hooks"]` 플래그로 게이팅, 기본 `False` |
| 4 | `exp02/evaluate.py` | 라벨 `"Thickness (nm)"` — 실제 단위 Å | `"Thickness (Å)"`, `"SLD (10⁻⁶ Å⁻²)"`로 수정 |
| 5 | `exp02/dataset.py` | 정규화 통계 인덱스 `0.7` 하드코딩, `val_ratio`와 독립 | `self.train_end`(실제 분할 경계) 사용 |
| 6 | `exp07/dataset.py`, `config.py` | test set이 clean simulation으로만 평가 → 실측 대비 낙관적 | `augment_eval` 플래그 추가. `False`(기본)=clean-sim 평가, `True`=측정 노이즈 포함 평가 |
| 7 | `exp07/dataset.py` | augmentation의 `delta_q`에 **target 그리드** 간격을 사용. augmentation은 원본 그리드 위에서 수행되므로 해상도가 다를 때 smearing σ가 어긋남 | `source_q` 간격으로 교정 (해상도 ablation의 전제조건) |

### 8.2 미해결

| # | 항목 | 내용 |
|---|---|---|
| 8 | exp02 데이터 소실 | `D:\03_Resources\Data\XRR_AI\data\one_layer\` 부재. exp02 직접 재현 시 100만 샘플 refnx 재시뮬레이션 필요. §7.2 ablation의 `E_exp02_like`가 대체 근사 |
| 9 | 정량 수치 미확보 | §7 표가 TBD 상태. ablation 실행이 유일한 잔여 작업 |

> 참고: PyTorch/CUDA 블로커(구 #8)는 §7.1에서 해결됨.

---

## 9. 결론

exp07의 개선은 3개 축으로 분해됨:

1. **관측 도메인 확장** (q 2.53×, 샘플링 10×) → 분해 가능 두께 범위 15–1489 Å에서 5.9–5916 Å로. 라벨 범위와 물리적 관측 한계의 정합 확보.
2. **측정 물리 모델링** → footprint·분해능·노이즈·정렬오차의 확률적 augmentation으로 측정 조건 불변성 학습. 원본 5만 샘플로 epoch당 200만 유효 샘플.
3. **실측 전이 설계** → 재샘플링 + valid-mask 채널로 임의 q-grid 실측 데이터 직접 입력 가능.

모델 용량 증가(1.37 M → 5.30 M)는 위 3개 축을 수용하기 위한 결과이며 단독 개선 요인이 아님.

정량적 성능 우위 주장은 §7의 ablation 실험 완료 후 기술할 것.
