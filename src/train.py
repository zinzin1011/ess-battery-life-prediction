"""전체 파이프라인
python -m src.train --data-dir <.mat 폴더>

1) .mat → 셀 단위 피처 (outputs/features.csv 캐시)
2) Batch 1(라벨 신뢰 가능 36셀)을 충전 정책 단위로 Train / Hold-out Valid 분할
3) 후보 모델별 GroupKFold(정책 단위) 교차검증으로 하이퍼파라미터 탐색 → Train(CV) MAPE
4) Train으로 학습 → Valid MAPE, Batch 1 전체로 재학습 → Test(Batch 2, Batch 3) MAPE
5) Train(CV) MAPE가 가장 낮은 모델을 최종 모델로 선정 (Valid는 확인용, Test는 선정에 사용하지 않음)
"""
import argparse
import json
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, GridSearchCV

from .config import RANDOM_SEED, DATA_DIR, OUT_DIR, FIG_DIR, BATCH_FILES, TARGET_PAPER_MAPE
from .load_data import load_batch
from .features import build_features
from .split import policy_holdout
from .models import candidates
from .evaluate import metrics, mape, log_mape_scorer

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def get_cells(data_dir):
    cache = OUT_DIR / "cells.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    cells = []
    for tag, fname in BATCH_FILES.items():
        print(f"[load] {fname}")
        cells += load_batch(data_dir / fname, tag)
    pickle.dump(cells, open(cache, "wb"))
    return cells


def fit_candidate(est, grid, X, y, groups, n_splits=5):
    cv = GroupKFold(n_splits=n_splits)
    if grid:
        gs = GridSearchCV(est, grid, scoring=log_mape_scorer, cv=cv, n_jobs=1)
        gs.fit(X, y, groups=groups)
        return gs.best_estimator_.get_params(), -gs.best_score_, gs.best_params_
    scores = []
    for tr, va in cv.split(X, y, groups):
        m = clone(est).fit(X.iloc[tr], y.iloc[tr])
        scores.append(mape(10 ** y.iloc[va], 10 ** m.predict(X.iloc[va])))
    return est.get_params(), float(np.mean(scores)), {}


def run(df, label=""):
    b1 = df[(df.batch == "b1") & df.usable]
    b2 = df[(df.batch == "b2") & df.usable]
    b3 = df[(df.batch == "b3") & df.usable]
    tr, va = policy_holdout(b1)
    rows, preds, fitted = [], {}, {}
    for name, feats, est, grid in candidates():
        Xtr, ytr = tr[feats], np.log10(tr.cycle_life)
        params, cv_mape, best = fit_candidate(est, grid, Xtr, ytr, tr.policy)
        m_tr = clone(est).set_params(**params).fit(Xtr, ytr)
        p_va = 10 ** m_tr.predict(va[feats])
        m_all = clone(est).set_params(**params).fit(b1[feats], np.log10(b1.cycle_life))
        p_b2, p_b3 = 10 ** m_all.predict(b2[feats]), 10 ** m_all.predict(b3[feats])
        r = dict(model=name, features=",".join(feats), best_params=json.dumps(best, default=float),
                 train_cv_MAPE=cv_mape, valid_MAPE=mape(va.cycle_life, p_va))
        for k, v in metrics(b2.cycle_life, p_b2).items(): r[f"b2_{k}"] = v
        for k, v in metrics(b3.cycle_life, p_b3).items(): r[f"b3_{k}"] = v
        rows.append(r)
        preds[name] = dict(valid=p_va, b2=p_b2, b3=p_b3)
        fitted[name] = m_all
        print(f"{label}{name:24s} CV {cv_mape:6.2f}  Valid {r['valid_MAPE']:6.2f}  B2 {r['b2_MAPE']:6.2f}  B3 {r['b3_MAPE']:6.2f}")
    res = pd.DataFrame(rows)
    res["select_score"] = res.train_cv_MAPE   # 모델 선정 기준: Batch 1 교차검증 MAPE
    return res, preds, fitted, (tr, va, b1, b2, b3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    args = ap.parse_args()
    from pathlib import Path
    OUT_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)

    cells = get_cells(Path(args.data_dir))
    df = build_features(cells)
    df.to_csv(OUT_DIR / "features.csv", index=False, encoding="utf-8-sig")
    print(df.groupby("batch")[["label_missing", "censored", "paper_excluded", "usable"]].sum())

    res, preds, fitted, (tr, va, b1, b2, b3) = run(df)
    res.to_csv(OUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    best = res.sort_values("select_score").iloc[0]
    print(f"\n[최종 모델] {best.model}  (CV {best.train_cv_MAPE:.2f}, Valid {best.valid_MAPE:.2f})")

    # 예측 결과 저장
    p = preds[best.model]
    out = pd.concat([
        va.assign(split="valid(B1 hold-out)", pred=p["valid"]),
        b2.assign(split="test(B2)", pred=p["b2"]),
        b3.assign(split="test(B3)", pred=p["b3"]),
    ])[["cell_id", "batch", "policy", "split", "cycle_life", "pred"]]
    out["APE_%"] = (out.pred - out.cycle_life).abs() / out.cycle_life * 100
    out.to_csv(OUT_DIR / "predictions_final.csv", index=False, encoding="utf-8-sig")
    pickle.dump(fitted[best.model], open(OUT_DIR / "final_model.pkl", "wb"))

    # 기준 사이클 민감도 (최종 모델 구성으로, 선택은 CV·Valid 기준)
    sens = []
    name = best.model
    for early in [5, 10, 20]:
        d2 = build_features(cells, early_cycle=early)
        r2, _, _, _ = run(d2, label=f"[ref {early:>2}] ")
        r2 = r2[r2.model == name].iloc[0]
        sens.append(dict(early_cycle=early, train_cv_MAPE=r2.train_cv_MAPE, valid_MAPE=r2.valid_MAPE,
                         b2_MAPE=r2.b2_MAPE, b3_MAPE=r2.b3_MAPE))
    pd.DataFrame(sens).to_csv(OUT_DIR / "sensitivity_ref_cycle.csv", index=False, encoding="utf-8-sig")

    info = dict(n_train=len(tr), n_valid=len(va), n_b1=len(b1), n_b2=len(b2), n_b3=len(b3),
                valid_policies=sorted(va.policy.unique()), final_model=best.model, target=TARGET_PAPER_MAPE)
    json.dump(info, open(OUT_DIR / "run_info.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
