"""후보 모델 정의. 목표변수는 log10(cycle_life), 입력변수는 학습 데이터로만 표준화."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from .config import RANDOM_SEED

F1 = ["log_dq_var"]                                   # Variance 모델 (기준)
F2 = F1 + ["log_abs_dq_min", "dq_at_2V"]              # ΔQ 확장
F3 = F2 + ["avgC_0_80"]                               # + 충전 속도

L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
ALPHAS = list(np.logspace(-4, 0, 20))


def candidates():
    """(이름, 피처세트, 파이프라인, 탐색 그리드)"""
    sc = StandardScaler
    return [
        ("Baseline (Linear, F1)", F1, make_pipeline(sc(), LinearRegression()), {}),
        ("ElasticNet (F2)", F2, make_pipeline(sc(), ElasticNet(max_iter=50000, random_state=RANDOM_SEED)),
         {"elasticnet__alpha": ALPHAS, "elasticnet__l1_ratio": L1_RATIOS}),
        ("ElasticNet (F3)", F3, make_pipeline(sc(), ElasticNet(max_iter=50000, random_state=RANDOM_SEED)),
         {"elasticnet__alpha": ALPHAS, "elasticnet__l1_ratio": L1_RATIOS}),
        ("Ridge (F2)", F2, make_pipeline(sc(), Ridge()), {"ridge__alpha": list(np.logspace(-3, 2, 20))}),
        ("Ridge (F3)", F3, make_pipeline(sc(), Ridge()), {"ridge__alpha": list(np.logspace(-3, 2, 20))}),
        ("Huber (F2)", F2, make_pipeline(sc(), HuberRegressor(max_iter=5000)),
         {"huberregressor__epsilon": [1.35, 1.75, 2.5], "huberregressor__alpha": [1e-4, 1e-3, 1e-2, 1e-1]}),
        ("RandomForest (F3)", F3, make_pipeline(sc(), RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED)),
         {"randomforestregressor__max_depth": [2, 3, 4], "randomforestregressor__min_samples_leaf": [2, 3, 5]}),
        ("LightGBM (F3)", F3, make_pipeline(sc(), LGBMRegressor(random_state=RANDOM_SEED, verbose=-1, min_child_samples=3)),
         {"lgbmregressor__num_leaves": [3, 4, 7], "lgbmregressor__n_estimators": [100, 300],
          "lgbmregressor__learning_rate": [0.03, 0.1]}),
    ]
