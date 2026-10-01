"""셀 단위 피처 생성. 사이클 100 이후 정보는 사용하지 않는다(누수 방지).
라벨 품질 플래그(label_missing, censored, paper_excluded)도 함께 만든다."""
import re
import numpy as np
import pandas as pd
from .config import CENSOR_AH, PAPER_EXCLUDED

POLICY_RE = re.compile(r"([\d.]+)C\((\d+)%\)-([\d.]+)C")


def _despike(x, tol=0.02, win=9):
    """주변 사이클 중앙값보다 tol 이상 벗어난 측정값을 중앙값으로 대체"""
    med = pd.Series(x).rolling(win, center=True, min_periods=1).median().to_numpy()
    x = x.copy()
    bad = np.abs(x - med) > tol
    x[bad] = med[bad]
    return x


def _moments(d):
    z = (d - d.mean()) / d.std()
    return (z ** 3).mean(), (z ** 4).mean() - 3


def parse_policy(p):
    """'C1(Q1%)-C2' → C1, Q1(%), C2, SOC 0→80% 평균 충전 속도(C)"""
    m = POLICY_RE.match(p)
    if not m:
        return dict(C1=np.nan, Q1=np.nan, C2=np.nan, avgC_0_80=np.nan)
    c1, q1, c2 = float(m.group(1)), float(m.group(2)) / 100, float(m.group(3))
    hours = q1 / c1 + (0.8 - q1) / c2
    return dict(C1=c1, Q1=q1 * 100, C2=c2, avgC_0_80=0.8 / hours)


def dq_features(q_late, q_early, prefix=""):
    d = q_late - q_early
    sk, ku = _moments(d)
    return {
        f"{prefix}log_dq_var": np.log10(d.var()),
        f"{prefix}log_abs_dq_min": np.log10(abs(d.min())),
        f"{prefix}log_abs_dq_mean": np.log10(abs(d.mean())),
        f"{prefix}dq_at_2V": d[-1],
        f"{prefix}dq_skew": sk,
        f"{prefix}dq_kurt": ku,
    }


def build_features(cells, early_cycle=10, late_cycle=100):
    rows = []
    for c in cells:
        s = c["summary"]
        qd = _despike(s["QDischarge"][1:100])          # 사이클 2~100 (사이클 1은 Batch 1에서 0으로 기록)
        x = np.arange(2, 101)
        ir = s["IR"][1:100].copy()
        ir[ir <= 0] = np.nan                            # 0 기록은 결측
        row = dict(
            cell_id=c["cell_id"], batch=c["batch"], policy=c["policy"],
            cycle_life=c["cycle_life"], n_cycles=c["n_cycles"], qd_last=c["qd_last"],
            label_missing=bool(np.isnan(c["cycle_life"])),
            censored=bool(c["qd_last"] > CENSOR_AH),
            paper_excluded=c["cell_id"] in PAPER_EXCLUDED,
            newstructure="newstructure" in c["policy"],
            qd_2=qd[0], qd_max_minus_2=qd.max() - qd[0], qd_100_minus_2=qd[-1] - qd[0],
            qd_slope_2_100=np.polyfit(x, qd, 1)[0],
            ir_2=ir[0], ir_min=np.nanmin(ir) if np.isfinite(ir).any() else np.nan,
            tavg_mean=s["Tavg"][1:100].mean(), tmax_max=s["Tmax"][1:100].max(),
            chargetime_first5=np.median(s["chargetime"][1:6]),
        )
        row.update(parse_policy(c["policy"]))
        row.update(dq_features(c["qdlin"][late_cycle], c["qdlin"][early_cycle]))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["usable"] = ~df.label_missing & ~df.censored & ~df.paper_excluded
    return df
