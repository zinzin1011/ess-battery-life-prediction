"""평가 지표: MAPE(주), RMSE, MAE, 수명 구간별 MAPE"""
import numpy as np


def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(p - y) / y) * 100)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = dict(MAPE=mape(y, p), RMSE=float(np.sqrt(np.mean((p - y) ** 2))), MAE=float(np.mean(np.abs(p - y))),
               median_ratio=float(np.median(p / y)))
    for name, m in [("MAPE_lt550", y < 550), ("MAPE_ge550", y >= 550)]:
        out[name] = mape(y[m], p[m]) if m.any() else np.nan
    return out


def log_mape_scorer(estimator, X, y_log):
    """GridSearchCV용: log10 예측을 원 단위로 환산해 MAPE 계산 (높을수록 좋게 부호 반전)"""
    return -mape(10 ** y_log, 10 ** estimator.predict(X))
