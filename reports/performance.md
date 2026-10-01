# 성능 보고 — 최종 모델: Ridge (F3)

- 피처: `log_dq_var,log_abs_dq_min,dq_at_2V,avgC_0_80` / 하이퍼파라미터: `{"ridge__alpha": 0.001}`
- 셀 수: Train 27 · Valid 9 (Batch 1 36) · Test Batch 2 39 · Batch 3 40
- Gap = 뒤 항목 MAPE − 앞 항목 MAPE (+이면 뒤 항목 오차가 큼)

| 구분 |  | MAPE (%) | 비고 |
| --- | --- | --- | --- |
| Train (Batch 1 CV) |  | 8.20 | 정책 단위 GroupKFold 5-fold 평균 |
| Valid (Batch 1 Hold-out) |  | 9.90 | 정책 5종 9셀 |
| Test (Batch 2) |  | 27.54 | 39셀, 최종 1회 평가 |
|  | Gap (Train-Valid) | +1.70 | (+) : 과적합 의심 |
|  | Gap (Valid-Test) | +17.64 | (+) : 배치간 일반화 저하 의심 |
|  | Gap (Target-Test) | +18.44 | Target : 원논문 9.1% |
| Test (Batch 3) |  | 12.05 | 40셀 (원논문 제외 4셀 제외) |
|  | Gap (Batch2-Batch3) | -15.49 | Test 성능 간 비교 |
|  | Gap (Target-Test) | +2.95 | Batch 3 기준, 원논문 성능 비교 |

## 보조 지표 (최종 모델)

| 데이터 | MAPE | RMSE (사이클) | MAE (사이클) | 예측/실측 중앙값 | MAPE <550 | MAPE ≥550 |
| --- | --- | --- | --- | --- | --- | --- |
| Batch 2 | 27.54 | 155 | 143 | 1.29 | 29.92 | 19.61 |
| Batch 3 | 12.05 | 224 | 139 | 0.99 | 20.79 | 11.83 |

## 후보 모델 비교 (선정 기준: Train CV MAPE)

| 모델 | Train CV | Valid | Test B2 | Test B3 |
| --- | --- | --- | --- | --- |
| Ridge (F3) | 8.20 | 9.90 | 27.54 | 12.05 |
| ElasticNet (F3) | 8.22 | 9.91 | 27.72 | 11.98 |
| Huber (F2) | 8.41 | 9.88 | 26.63 | 12.63 |
| Ridge (F2) | 8.83 | 9.48 | 26.96 | 12.25 |
| ElasticNet (F2) | 8.88 | 9.46 | 27.12 | 12.21 |
| Baseline (Linear, F1) | 9.70 | 7.96 | 28.56 | 11.75 |
| RandomForest (F3) | 10.48 | 6.11 | 30.66 | 16.60 |
| LightGBM (F3) | 12.32 | 8.76 | 31.74 | 16.43 |

## ΔQ 기준 사이클 민감도 (최종 모델 구성)

| ΔQ | Train CV | Valid | Test B2 | Test B3 |
| --- | --- | --- | --- | --- |
| Q(100) − Q(5) | 7.77 | 9.16 | 30.76 | 13.65 |
| Q(100) − Q(10) | 8.20 | 9.90 | 27.54 | 12.05 |
| Q(100) − Q(20) | 9.56 | 9.75 | 31.14 | 12.14 |