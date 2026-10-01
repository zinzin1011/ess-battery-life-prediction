"""Batch 1 내부 분할: 충전 정책(그룹) 단위 Hold-out + GroupKFold
같은 정책 셀이 학습·검증에 동시에 들어가지 않도록 정책 단위로 나눈다."""
import numpy as np
import pandas as pd


def policy_holdout(df, n_groups=5):
    """정책을 평균 수명 순으로 정렬한 뒤 일정 간격으로 n_groups개 정책을 검증용으로 선택
    → 검증셋이 수명 전 구간을 고르게 포함(층화)하고, 정책 단위로 분리된다."""
    g = df.groupby("policy")["cycle_life"].mean().sort_values()
    step = len(g) / n_groups
    picks = [g.index[int(step * k + step / 2)] for k in range(n_groups)]
    is_valid = df["policy"].isin(picks)
    return df[~is_valid].copy(), df[is_valid].copy()
