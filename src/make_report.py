"""학습 결과(outputs/*.csv)로 성능 보고 표(reports/performance.md)와 그림(outputs/figures)을 만든다.
python -m src.make_report"""
import os, json
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from .config import OUT_DIR, FIG_DIR, ROOT, TARGET_PAPER_MAPE
from .evaluate import metrics

for name in ["AppleGothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "Malgun Gothic"]:
    if any(name in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name; break
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"


def f2(x): return f"{x:.2f}"
def sgn(x): return f"{x:+.2f}"


def main():
    res = pd.read_csv(OUT_DIR / "model_comparison.csv")
    pred = pd.read_csv(OUT_DIR / "predictions_final.csv")
    sens = pd.read_csv(OUT_DIR / "sensitivity_ref_cycle.csv")
    info = json.load(open(OUT_DIR / "run_info.json"))
    best = res[res.model == info["final_model"]].iloc[0]
    T = TARGET_PAPER_MAPE
    tr, va, b2, b3 = best.train_cv_MAPE, best.valid_MAPE, best.b2_MAPE, best.b3_MAPE

    md = [f"# 성능 보고 — 최종 모델: {best.model}\n",
          f"- 피처: `{best.features}` / 하이퍼파라미터: `{best.best_params}`",
          f"- 셀 수: Train {info['n_train']} · Valid {info['n_valid']} (Batch 1 {info['n_b1']}) · Test Batch 2 {info['n_b2']} · Batch 3 {info['n_b3']}",
          "- Gap = 뒤 항목 MAPE − 앞 항목 MAPE (+이면 뒤 항목 오차가 큼)\n",
          "| 구분 |  | MAPE (%) | 비고 |", "| --- | --- | --- | --- |",
          f"| Train (Batch 1 CV) |  | {f2(tr)} | 정책 단위 GroupKFold 5-fold 평균 |",
          f"| Valid (Batch 1 Hold-out) |  | {f2(va)} | 정책 5종 9셀 |",
          f"| Test (Batch 2) |  | {f2(b2)} | 39셀, 최종 1회 평가 |",
          f"|  | Gap (Train-Valid) | {sgn(va - tr)} | (+) : 과적합 의심 |",
          f"|  | Gap (Valid-Test) | {sgn(b2 - va)} | (+) : 배치간 일반화 저하 의심 |",
          f"|  | Gap (Target-Test) | {sgn(b2 - T)} | Target : 원논문 {T}% |",
          f"| Test (Batch 3) |  | {f2(b3)} | 40셀 (원논문 제외 4셀 제외) |",
          f"|  | Gap (Batch2-Batch3) | {sgn(b3 - b2)} | Test 성능 간 비교 |",
          f"|  | Gap (Target-Test) | {sgn(b3 - T)} | Batch 3 기준, 원논문 성능 비교 |\n",
          "## 보조 지표 (최종 모델)\n", "| 데이터 | MAPE | RMSE (사이클) | MAE (사이클) | 예측/실측 중앙값 | MAPE <550 | MAPE ≥550 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, lab in [("b2", "Batch 2"), ("b3", "Batch 3")]:
        md.append(f"| {lab} | {best[k+'_MAPE']:.2f} | {best[k+'_RMSE']:.0f} | {best[k+'_MAE']:.0f} | {best[k+'_median_ratio']:.2f} | "
                  f"{best[k+'_MAPE_lt550']:.2f} | {best[k+'_MAPE_ge550']:.2f} |")
    md += ["\n## 후보 모델 비교 (선정 기준: Train CV MAPE)\n",
           "| 모델 | Train CV | Valid | Test B2 | Test B3 |", "| --- | --- | --- | --- | --- |"]
    for _, r in res.sort_values("train_cv_MAPE").iterrows():
        md.append(f"| {r.model} | {r.train_cv_MAPE:.2f} | {r.valid_MAPE:.2f} | {r.b2_MAPE:.2f} | {r.b3_MAPE:.2f} |")
    md += ["\n## ΔQ 기준 사이클 민감도 (최종 모델 구성)\n", "| ΔQ | Train CV | Valid | Test B2 | Test B3 |", "| --- | --- | --- | --- | --- |"]
    for _, r in sens.iterrows():
        md.append(f"| Q(100) − Q({int(r.early_cycle)}) | {r.train_cv_MAPE:.2f} | {r.valid_MAPE:.2f} | {r.b2_MAPE:.2f} | {r.b3_MAPE:.2f} |")
    (ROOT / "reports").mkdir(exist_ok=True)
    open(ROOT / "reports" / "performance.md", "w", encoding="utf-8").write("\n".join(md))

    # 그림 1: 모델 비교
    r = res.sort_values("train_cv_MAPE")
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=200)
    x = np.arange(len(r)); w = 0.2
    for i, (col, lab, c) in enumerate([("train_cv_MAPE", "Train (B1 CV)", "#86b6ef"), ("valid_MAPE", "Valid (B1 Hold-out)", "#2a78d6"),
                                        ("b2_MAPE", "Test (Batch 2)", "#eb6834"), ("b3_MAPE", "Test (Batch 3)", "#1baf7a")]):
        ax.bar(x + (i - 1.5) * w, r[col], w, color=c, label=lab, edgecolor="white")
    ax.axhline(T, color=INK, ls="--", lw=1); ax.text(len(r) - 0.5, T + 0.6, f"원논문 {T}%", ha="right", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(r.model, rotation=15, fontsize=9); ax.set_ylabel("MAPE (%)")
    ax.set_title("후보 모델별 MAPE (왼쪽부터 Train CV 낮은 순)", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper left")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(FIG_DIR / "model_comparison.png", facecolor="white"); plt.close(fig)

    # 그림 2: 예측 vs 실측
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200)
    for ax, (split, c) in zip(axes, [("valid(B1 hold-out)", "#2a78d6"), ("test(B2)", "#eb6834"), ("test(B3)", "#1baf7a")]):
        d = pred[pred.split == split]
        lim = [300, 2100]
        ax.fill_between(lim, [l * (1 - T / 100) for l in lim], [l * (1 + T / 100) for l in lim], color=MUTED, alpha=0.12, label=f"±{T}%")
        ax.plot(lim, lim, color=INK, lw=1)
        ax.axvspan(300, 534, color="#eb6834", alpha=0.05)
        ax.scatter(d.cycle_life, d.pred, s=30, color=c, edgecolor="white", lw=0.8, zorder=3)
        m = metrics(d.cycle_life, d.pred)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
        for a in (ax.xaxis, ax.yaxis): a.set_minor_formatter(plt.NullFormatter())
        ticks = [400, 600, 1000, 1500, 2000]
        ax.set_xticks(ticks); ax.set_xticklabels(ticks); ax.set_yticks(ticks); ax.set_yticklabels(ticks)
        ax.set_title(f"{split}  MAPE {m['MAPE']:.1f}%", loc="left", fontsize=12, color=INK)
        ax.set_xlabel("실측 수명 (사이클)"); ax.set_ylabel("예측 수명 (사이클)")
        ax.grid(color=GRID); ax.set_axisbelow(True)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    axes[1].text(310, 1900, "음영: 학습 최소 수명(534) 미만", fontsize=8.5, color=MUTED)
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle(f"최종 모델 {best.model} — 예측 vs 실측", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(); fig.savefig(FIG_DIR / "pred_vs_actual.png", facecolor="white"); plt.close(fig)
    print("\n".join(md))


if __name__ == "__main__":
    main()
