"""
내장 샘플 3종의 경로/생성 로직.
 - 정상 데이터(normal), 이상 데이터(outlier): 제출 코드의 원본 CSV를 그대로 사용.
 - 센서 오류 예시: 정상 데이터 일부에 결측·고착·물리적으로 불가능한 고립 스파이크·시각 역행을
   인위로 넣은 "합성 예시" (화면에 반드시 그렇게 표시).
"""
from pathlib import Path

import numpy as np
import pandas as pd

_VENDOR_DATA = Path(__file__).resolve().parent.parent / "vendor" / "data"
_ORIGINAL_DATA = Path(__file__).resolve().parent.parent.parent / "제출" / "소스코드" / "data"
DATA_DIR = _VENDOR_DATA if (_VENDOR_DATA / "press_data_normal.csv").exists() else _ORIGINAL_DATA
NORMAL_PATH = DATA_DIR / "press_data_normal.csv"
OUTLIER_PATH = DATA_DIR / "outlier_data.csv"

SENSOR_ERROR_N_ROWS = 2000
SENSOR_ERROR_SEED = 42


def make_sensor_error_sample() -> pd.DataFrame:
    """정상 데이터 앞부분 SENSOR_ERROR_N_ROWS행에 인위 결함을 주입한 합성 CSV(데이터프레임)를 만든다.
    결정적(고정 시드)이라 매번 같은 결과를 낸다.

    물리적으로 불가능한 스파이크(800, 1200행)는 같은 행의 다른 센서 컬럼도 함께 결측으로 만들어
    모델 입력(사이클 추출)에서는 그 행 전체가 제외되게 한다 — 품질 점검표에는 원본 그대로 남아
    "물리적으로 불가능한 값"으로 잡히지만, 하이브리드 모델의 사이클 특징에는 섞이지 않게 해서
    "센서 오류"와 "모델이 탐지하는 이상 신호"를 명확히 분리한다 (안 그러면 이 둘이 같은 사이클에
    섞여 모델도 함께 경보를 내고, 그러면 '이상 발생'이 메인이 되어 센서 오류 예시의 취지가 흐려진다).
    """
    base = pd.read_csv(NORMAL_PATH, index_col=0)
    d = base.iloc[:SENSOR_ERROR_N_ROWS].copy().reset_index(drop=True)
    rng = np.random.default_rng(SENSOR_ERROR_SEED)

    # 1) 결측: 임의의 18개 행, AI1_Vibration 값을 비움 (800, 1200행과는 겹치지 않게)
    candidates = np.setdiff1d(np.arange(len(d)), [800, 1200])
    miss_idx = rng.choice(candidates, size=18, replace=False)
    d.loc[miss_idx, "AI1_Vibration"] = np.nan

    # 2) 고착/포화: AI0_Vibration 500~508행을 같은 값으로 고정 (9행 연속, 원래 값 그대로 고정해서
    #    모델의 크기 특징(RMS 등)을 크게 바꾸지 않음 — 이 샘플의 목적은 "센서가 멈췄다"는 신호 자체)
    stuck_val = d.loc[500, "AI0_Vibration"]
    d.loc[500:508, "AI0_Vibration"] = stuck_val

    # 3) 물리적으로 불가능한 고립 스파이크: AI2_Current 두 지점을 극단값으로 (|값|>2000, data_quality.
    #    IMPOSSIBLE_BOUND 기준의 4~4.5배). 같은 행의 다른 센서 컬럼도 결측으로 만들어 그 행 자체가
    #    모델 입력에서는 제외되도록 한다.
    d.loc[800, "AI2_Current"] = 9000.0
    d.loc[800, "AI0_Vibration"] = np.nan
    d.loc[1200, "AI2_Current"] = -8500.0
    d.loc[1200, "AI1_Vibration"] = np.nan

    # 4) 시각 역행: 1500행의 시각을 바로 앞 행보다 0.05초 이르게 바꿔서 1곳 역행 발생
    #    (이웃 1행만 순서가 바뀌는 수준으로 작게 둬서, 사이클 추출·파형 자체는 크게 망가지지 않게 함)
    ts = pd.to_datetime(d["TimeStamp"])
    ts.iloc[1500] = ts.iloc[1499] - pd.Timedelta(seconds=0.05)
    d["TimeStamp"] = ts.astype(str)

    return d
