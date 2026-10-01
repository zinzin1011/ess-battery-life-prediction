"""MATLAB v7.3(.mat, HDF5) 배치 파일에서 셀별 요약 데이터와 방전 곡선(Qdlin)을 읽는다."""
import h5py
import numpy as np


def _to_str(f, ref):
    return "".join(chr(int(c)) for c in f[ref][()].flatten())


def load_batch(path, tag, qdlin_cycles=(5, 10, 20, 100)):
    """배치 파일 하나를 읽어 셀 단위 dict 리스트로 반환.
    - summary: 사이클별 방전용량·내부저항·온도·충전시간 (사이클 1~100만 보관)
    - qdlin: 전압 기준 방전용량 곡선(1,000점), 지정 사이클만 보관
    """
    cells = []
    with h5py.File(path, "r") as f:
        b = f["batch"]
        for i in range(b["summary"].shape[0]):
            s = f[b["summary"][i, 0]]
            c = f[b["cycles"][i, 0]]
            qd_all = s["QDischarge"][()].flatten()
            cells.append(dict(
                cell_id=f"{tag}c{i}",
                batch=tag,
                policy=_to_str(f, b["policy_readable"][i, 0]),
                cycle_life=float(f[b["cycle_life"][i, 0]][()].flatten()[0]),
                n_cycles=len(qd_all),
                qd_last=float(qd_all[-1]),
                summary={k: s[k][()].flatten()[:100].astype(float)
                         for k in ["QDischarge", "IR", "Tavg", "Tmax", "Tmin", "chargetime"]},
                qdlin={cy: f[c["Qdlin"][cy - 1, 0]][()].flatten().astype(float) for cy in qdlin_cycles},
            ))
    return cells
