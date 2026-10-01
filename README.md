# ESS 배터리 수명 예측

ESS(에너지저장시스템) 배터리의 운전 초기 100 사이클 데이터만으로 교체 시점(SOH 80%, 방전용량 0.88 Ah 도달 사이클)을 예측하는 회귀 모델

## 프로젝트 개요
- 데이터셋 : MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 학습 데이터 : Batch 1 (2017-05-12) — 라벨 신뢰 가능 36셀
- 평가 데이터 : Batch 2 (2018-02-20) 39셀, 추가 평가 Batch 3 (2018-04-12) 40셀
- 태스크 : Regression (Cycle Life 예측), 목표변수 log10(cycle_life)
- 평가 지표 : MAPE (보조: RMSE, MAE, 수명 구간별 MAPE)

## 파일 구조
```
├── data/
│   └── README.md            # 원본 .mat 다운로드·배치 방법
├── src/
│   ├── config.py            # 경로, RANDOM_SEED=42, 셀 제외 기준
│   ├── load_data.py         # .mat(HDF5) → 셀별 요약·방전 곡선
│   ├── features.py          # 피처 생성(ΔQ 통계량, 충전 속도 등), 라벨 품질 플래그
│   ├── split.py             # Batch 1 충전 정책 단위 Hold-out 분할
│   ├── models.py            # 후보 모델·탐색 범위
│   ├── evaluate.py          # MAPE·RMSE·MAE, 구간별 MAPE
│   ├── train.py             # 전체 파이프라인 (학습·검증·평가)
│   ├── make_report.py       # 성능 표·그림 생성
│   └── calibration.py       # [추가 분석] 신규 배치 소수 셀 보정 시나리오
├── outputs/                 # features.csv, model_comparison.csv, predictions_final.csv, figures/
├── reports/performance.md   # 성능 보고 표
├── requirements.txt
└── README.md
```

## 실행 방법
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.train --data-dir data/raw                # 학습·평가 (약 30초, 첫 실행 시 .mat 로딩 포함)
python -m src.make_report                              # reports/performance.md, outputs/figures/*.png
python -m src.calibration                              # (선택) 보정 시나리오
```

## 방법 요약
| 단계 | 내용 |
| --- | --- |
| 데이터 정제 | 수명 라벨 없는 셀 제외(B2 8, B3 2), 0.88 Ah 미도달 셀 제외(B1 10), 원논문 품질 제외 셀 제외(B3 4) |
| 전처리 | 사이클 1 제외(B1 기록 0), 방전용량 급등락 보정(주변 9사이클 중앙값 대비 0.02 Ah), 내부저항 0 → 결측 |
| 핵심 피처 | ΔQ(V) = Q₁₀₀(V) − Q₁₀(V) 의 log10 분산·log10 \|최솟값\|·2.0 V 값, SOC 0→80% 평균 충전 속도 |
| 목표변수 | log10(cycle_life) — 왜도 1.04 → 0.03, 예측 후 원 단위로 환산해 MAPE 계산 |
| 분할 | Batch 1을 충전 정책 단위로 Train 27 / Valid 9 분할(수명 구간 층화), CV는 GroupKFold(정책) 5-fold |
| 후보 모델 | Linear(기준), ElasticNet, Ridge, Huber, RandomForest, LightGBM — StandardScaler는 학습 데이터로만 fit |
| 모델 선정 | Train(CV) MAPE 최소 모델. Valid는 확인용, Test는 선정에 사용하지 않음 |
| 최종 학습 | 선정된 하이퍼파라미터로 Batch 1 전체(36셀) 재학습 후 Batch 2·3 각 1회 평가 |

## 결과 (최종 모델: Ridge, 피처 4개)
| 구분 |  | MAPE (%) | 비고 |
| --- | --- | --- | --- |
| Train (Batch 1 CV) |  | 8.20 | 정책 단위 GroupKFold 평균 |
| Valid (Batch 1 Hold-out) |  | 9.90 | 정책 5종 9셀 |
| Test (Batch 2) |  | 27.54 | 39셀 |
|  | Gap (Train-Valid) | +1.70 | (+) : 과적합 의심 |
|  | Gap (Valid-Test) | +17.64 | (+) : 배치간 일반화 저하 의심 |
|  | Gap (Target-Test) | +18.44 | Target : 원논문 9.1% |
| Test (Batch 3) |  | 12.05 | 40셀 |
|  | Gap (Batch2-Batch3) | −15.49 | Test 성능 간 비교 |
|  | Gap (Target-Test) | +2.95 | Batch 3 기준, 원논문 성능 비교 |

Gap = 뒤 항목 MAPE − 앞 항목 MAPE. 전체 후보 비교와 민감도는 `reports/performance.md` 참고.

## 결과 해석
- **과적합은 작다.** Train CV와 Valid 차이가 1.7%p로, 학습 배치 안에서는 논문 수준(8~10%)이 재현된다.
- **Batch 2 오차는 배치 간 수명 수준 차이에서 온다.** 예측이 실측보다 중앙값 29% 길게 나오는 한쪽 방향 편향이며, 학습 범위 안(≥550)인 셀에서도 MAPE 19.6%다. 학습 최소 수명(534) 미만 셀은 29.9%로 더 크다.
- **Batch 3는 12.05%로 논문 대비 +2.95%p**다. 다만 1,600 사이클 이상 장수명 셀은 과소 예측된다.
- 트리 계열(RandomForest, LightGBM)은 학습 범위 밖 예측이 불가해 Batch 2·3 모두 선형 모델보다 3~5%p 나쁘다.
- 지급된 Batch 2(2018-02-20)는 원논문의 평가 배치(2017-06-30)와 다른 파일이라, 원논문 9.1%와의 직접 비교에는 한계가 있다.

## 추가 분석 — 신규 배치 소수 셀 보정 (운영 시나리오, 최종 성능 아님)
신규 배치에서 k개 셀의 실측 수명으로 로그 오차 중앙값을 구해 나머지 셀 예측을 보정 (무작위 200회 평균)

| 평가 배치 | 보정 없음 | 3셀 보정 | 5셀 보정 | 10셀 보정 |
| --- | --- | --- | --- | --- |
| Batch 2 | 27.54 | 9.55 | 8.77 | 8.21 |
| Batch 3 | 12.05 | 14.54 | 13.65 | 12.87 |

배치 편향이 큰 경우(Batch 2)에만 효과가 있어, 운영 시에는 소수 셀로 편향을 먼저 확인한 뒤 보정 여부를 결정해야 한다.

## 재현성
- `RANDOM_SEED = 42` (numpy, scikit-learn, LightGBM)
- 개발 환경: Python 3.10, numpy 2.2, pandas 2.3, h5py 3.16, scikit-learn 1.7, lightgbm 4.7, matplotlib 3.10
