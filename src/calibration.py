"""[추가 분석 — 최종 성능 아님] 신규 배치 소수 셀 실측 보정 시나리오
운영 제언("신규 로트는 일부 셀 실측값으로 보정")의 효과를 확인한다.
Batch 2(또는 3)에서 k개 셀의 실측 수명으로 log 오차 중앙값(배치 오프셋)을 구해 나머지 셀 예측에 더한다.
무작위 추출을 200회 반복한 평균 MAPE를 보고한다.  python -m src.calibration"""
import numpy as np
import pandas as pd
from .config import OUT_DIR, RANDOM_SEED
from .evaluate import mape


RULE = False


def main():
    pred = pd.read_csv(OUT_DIR / "predictions_final.csv")
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for split in ["test(B2)", "test(B3)"]:
        d = pred[pred.split == split].reset_index(drop=True)
        rows.append(dict(split=split, k=0, MAPE_mean=mape(d.cycle_life, d.pred), MAPE_p10=np.nan, MAPE_p90=np.nan))
        for k in [3, 5, 10]:
            scores = []
            for _ in range(200):
                idx = rng.choice(len(d), k, replace=False)
                off = np.median(np.log10(d.cycle_life[idx]) - np.log10(d.pred[idx]))
                if RULE and abs(off) < np.log10(1.10):   # 편향이 10% 미만이면 보정하지 않음
                    off = 0.0
                rest = d.drop(index=idx)
                scores.append(mape(rest.cycle_life, rest.pred * 10 ** off))
            rows.append(dict(split=split, k=k, MAPE_mean=np.mean(scores), MAPE_p10=np.percentile(scores, 10), MAPE_p90=np.percentile(scores, 90)))
    return pd.DataFrame(rows)


def report():
    global RULE
    RULE = False; a = main().assign(rule="항상 보정")
    RULE = True; b = main().assign(rule="편향 10% 이상일 때만 보정")
    r = pd.concat([a, b[b.k > 0]])
    r.to_csv(OUT_DIR / "calibration_scenario.csv", index=False, encoding="utf-8-sig")
    print(r.round(2).to_string(index=False))


if __name__ == "__main__":
    report()
