"""프로젝트 공통 설정: 경로, 난수 시드, 배치 파일, 셀 제외 기준"""
from pathlib import Path

RANDOM_SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"          # .mat 파일 위치 (--data-dir 로 변경 가능)
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

BATCH_FILES = {
    "b1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",   # 학습
    "b2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",   # 평가 (mandatory)
    "b3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",   # 추가 평가 (optional)
}

EOL_AH = 0.88               # 공칭용량 1.1 Ah의 80% (SOH 80%)
CENSOR_AH = 0.885           # 마지막 용량이 이 값보다 크면 EOL 미도달 (공식 LoadData.m 기준)
CYCLE_LATE, CYCLE_EARLY = 100, 10   # ΔQ = Q(100) - Q(10)
TARGET_PAPER_MAPE = 9.1     # 원논문 회귀 성능 (%)

# 원논문(LoadData.m)에서 데이터 품질 문제로 제외한 Batch 3 셀
PAPER_EXCLUDED = {"b3c2", "b3c37", "b3c42", "b3c43"}
