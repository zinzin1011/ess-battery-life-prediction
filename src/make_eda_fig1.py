"""그림 1(EDA 수명 분포) 생성: 수명 분포(A)는 분석 대상 119셀, ΔQ 산점도(B)는 제외 10셀을 점선 원으로 함께 표시"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import glob
for p in glob.glob("/usr/share/fonts/opentype/noto/NotoSansCJK-*.ttc"):   # 리눅스용 (없으면 건너뜀)
    fm.fontManager.addfont(p)
_have = {f.name for f in fm.fontManager.ttflist}
_kr = next((f for f in ["AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"] if f in _have), None)
plt.rcParams.update({"font.family": _kr or "sans-serif", "axes.unicode_minus": False, "font.size": 10})

d = pd.read_csv(ROOT / "outputs" / "features.csv")
lab = d[~d.label_missing].copy()            # 수명 값이 있는 129셀
use = lab[~lab.censored].copy()             # 분석 대상 119셀
COL = {"b1": "#2a77d4", "b2": "#e8673a", "b3": "#1ab37b"}
NAME = {"b1": "Batch 1 (2017-05-12)", "b2": "Batch 2 (2018-02-20)", "b3": "Batch 3 (2018-04-12)"}

fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"wspace": 0.18})

# ---------- A. 수명 분포 (119셀) ----------
bins = np.arange(300, 2101, 100)   # 500, 1,000 경계와 구간 경계를 맞춤
ax.axvspan(150, 500, color="#e8673a", alpha=0.05, lw=0)
ax.axvspan(1000, 2300, color="#2a77d4", alpha=0.05, lw=0)
bottom = np.zeros(len(bins) - 1)
for b in ["b1", "b2", "b3"]:
    h, _ = np.histogram(use.loc[use.batch == b, "cycle_life"], bins)
    ax.bar(bins[:-1], h, width=95, align="edge", bottom=bottom, color=COL[b],
           edgecolor="white", lw=0.8, label=f"{NAME[b]}  n={(use.batch == b).sum()}")
    bottom += h
for x in (500, 1000):
    ax.axvline(x, color="#666", ls="--", lw=1)
n = len(use); s = (use.cycle_life < 500).sum(); l = (use.cycle_life > 1000).sum(); m = n - s - l
top = bottom.max() * 1.22
for x, t in [(325, f"단수명 <500\n{s}개 ({s/n*100:.1f}%)"),
             (750, f"500~1,000\n{m}개 ({m/n*100:.1f}%)"),
             (1375, f"장수명 >1,000\n{l}개 ({l/n*100:.1f}%)")]:
    ax.text(x, top * 0.93, t, ha="center", va="top", fontsize=10)
q1, q3 = use.cycle_life.quantile([.25, .75]); up = q3 + 1.5 * (q3 - q1)
k = (use.cycle_life > up).sum()
ax.axvline(up, color="#666", ls=":", lw=1)
ax.text(up + 15, top * 0.70, f"IQR 상한 {up:,.0f}\n(초과 {k}셀, 모두 Batch 3)", fontsize=9, color="#555")
ax.set_xlim(150, 2300); ax.set_ylim(0, top)
from matplotlib.ticker import MaxNLocator
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel("Cycle Life (사이클)"); ax.set_ylabel("셀 수")
ax.set_title(f"A. 배치별 수명 분포 (분석 대상 n={n}, 100 사이클 구간)", loc="left", fontsize=12)
ax.legend(loc="center right", frameon=False, fontsize=9)
ax.grid(axis="y", color="#e5e5e5"); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)

# ---------- B. ΔQ 분산과 수명 (129셀, 제외 10셀은 점선 원) ----------
for b in ["b1", "b2", "b3"]:
    g = lab[lab.batch == b]
    new, old = g[g.newstructure], g[~g.newstructure]
    bx.scatter(new.log_dq_var, new.cycle_life, s=42, color=COL[b], zorder=3)
    bx.scatter(old.log_dq_var, old.cycle_life, s=42, facecolor="white", edgecolor=COL[b], lw=1.6, zorder=3)
c = lab[lab.censored]
bx.scatter(c.log_dq_var, c.cycle_life, s=190, facecolor="none", edgecolor="#333", lw=1, ls=(0, (1.5, 1.5)), zorder=4)
bx.set_yscale("log")
bx.set_yticks([400, 500, 700, 1000, 1500, 2000]); bx.set_yticklabels(["400", "500", "700", "1,000", "1,500", "2,000"])
bx.minorticks_off()
bx.axhline(500, color="#666", ls="--", lw=1)
sb = use[use.cycle_life < 500]
bx.annotate(f"500 사이클 미만 {len(sb)}셀: 모두 Batch 2 기존 구조", xy=(sb.log_dq_var.median(), 445),
            xytext=(-4.95, 425), fontsize=9, arrowprops=dict(arrowstyle="-", color="#666", lw=0.8))
bx.set_xlabel(r"log10 Var($\Delta Q_{100-10}(V)$)  — 클수록 초기 열화 큼")
bx.set_ylabel("Cycle Life (log 축)")
bx.set_title("B. 초기 ΔQ 분산과 수명의 관계", loc="left", fontsize=12)
h = [Line2D([], [], ls="", marker="o", color=COL[b], label=b) for b in ["b1", "b2", "b3"]]
h += [Line2D([], [], ls="", marker="o", color="#666", label="채움: 신규 구조(newstructure)"),
      Line2D([], [], ls="", marker="o", markerfacecolor="white", markeredgecolor="#666", label="빈 원: 기존 구조"),
      Line2D([], [], ls="", marker="o", markersize=12, markerfacecolor="none", markeredgecolor="#333",
             label=f"점선 원: 0.88 Ah 미도달 {len(c)}셀\n(관측 종료 시점, 실제 수명은 더 김 · A에서 제외)")]
bx.legend(handles=h, loc="upper right", frameon=False, fontsize=9)
bx.grid(color="#e5e5e5"); bx.set_axisbelow(True)
for sp in ("top", "right"): bx.spines[sp].set_visible(False)

fig.savefig(ROOT / "outputs" / "figures" / "eda_cycle_life_distribution.png", dpi=150, bbox_inches="tight")
print(n, s, m, l, round(up), k, len(c))
