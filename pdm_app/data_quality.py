"""
센서·기록 자체를 의심할 근거가 있는지 점검한다 (data_check.py의 점검 방식을 참고했지만,
"정상 범위 밖"이나 값/차분 z-점수만으로는 내장 이상 샘플(연속적으로 변하는 진짜 이상 신호)까지
센서 오류로 오판하게 되므로 그 기준은 쓰지 않는다 — 실제로 outlier_data.csv에
값 이상치(|z|>6) 153+21건, 튀는 값(차분 |z|>6) 65+2건이 나오는 것을 확인했다
(web_app 개발 중 data_check.py 실행 결과). 대신 아래 기준만 "센서 점검 필요"로 쓴다:

  1) 결측·비숫자 값
  2) 같은 값이 RUN행 이상 이어지는 고착/포화
  3) 물리적으로 불가능한 수준의 값 (정상 파일과 이상 파일 모두에서 한 번도 나타나지 않는
     범위 밖의 고립된 값). 두 파일 관찰 범위:
       AI0_Vibration: 정상 [-0.315, 0.352], 이상 [-1.503, 1.803]
       AI1_Vibration: 정상 [-0.366, 0.400], 이상 [-0.751, 0.509]
       AI2_Current  : 정상 [-271.6, 273.2], 이상 [-399.4, 538.8]
     → 두 파일 중 더 넓은 쪽(이상 파일) 최댓값의 약 3~5배를 "물리적으로 불가능" 경계로 잡았다.
       AI0/AI1: |값| > 5.0 (이상 파일 최댓값 1.803의 약 2.8배, 정상 파일 최댓값의 약 14배)
       AI2    : |값| > 2000 (이상 파일 최댓값 538.8의 약 3.7배)
     + 이 값이 "고립"되어 있을 것(연속 3행 이하)도 함께 요구해, 사이클 전체가 지속적으로
       바뀌는 진짜 이상 신호와 구분한다.
  4) 시각 역행·중복
  5) 샘플링 간격이 10Hz(0.1초)와 크게 다름 (burst 안 간격 중 0.09~0.11초 비율이 절반 미만)
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SENSOR_COLS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
STUCK_RUN = 5                    # 같은 값이 이 행 수 이상 이어지면 고착으로 간주
IMPOSSIBLE_BOUND = {             # "물리적으로 불가능" 경계값 (근거: 위 docstring)
    "AI0_Vibration": 5.0,
    "AI1_Vibration": 5.0,
    "AI2_Current": 2000.0,
}
ISOLATED_MAX_RUN = 3             # 이 값 이하로 연속되면 "고립"으로 간주
SAMPLING_OK_RATIO = 0.5          # burst 안 간격이 0.09~0.11초인 비율이 이보다 낮으면 경보
# 정상 파일에 시각·값이 완전히 같은 중복행이 1개 있다는 것이 알려진 사실이라
# (README 3절 "결측·중복"), 중복 1건만으로는 센서 점검 필요로 잡지 않는다.
DUPLICATE_MIN_COUNT = 2


@dataclass
class QualityReport:
    issues: list = field(default_factory=list)       # 사용자에게 보여줄 한국어 문장들
    detail_rows: list = field(default_factory=list)   # 표로 보여줄 세부 행 (dict)
    n_missing: int = 0
    n_non_numeric: int = 0
    n_duplicate_rows: int = 0
    n_time_regressions: int = 0
    n_stuck: dict = field(default_factory=dict)
    n_impossible_isolated: dict = field(default_factory=dict)
    sampling_ok_ratio: float = 1.0
    has_issue: bool = False


def _max_run_lengths(mask: np.ndarray):
    """True가 연속되는 길이들의 배열. mask는 bool numpy array."""
    if mask.sum() == 0:
        return np.array([], dtype=int)
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    runs = idx[1::2] - idx[0::2]
    return runs


def check_quality(df_quality: pd.DataFrame, n_missing: int, n_non_numeric: int,
                   n_duplicate_rows: int, n_time_regressions: int) -> QualityReport:
    """df_quality: 원본 행 순서 그대로, 결측/비숫자가 NaN으로 남아있는 데이터프레임.
    loader.load_uploaded_csv()가 이미 계산한 결측/비숫자/중복/시각역행 개수를 그대로 받는다."""
    r = QualityReport(
        n_missing=n_missing, n_non_numeric=n_non_numeric,
        n_duplicate_rows=n_duplicate_rows, n_time_regressions=n_time_regressions,
    )

    if n_missing > 0:
        r.issues.append(f"결측값이 {n_missing}개 있습니다.")
    if n_non_numeric > 0:
        r.issues.append(f"숫자로 읽을 수 없는 센서 값이 {n_non_numeric}개 있습니다.")
    if n_duplicate_rows >= DUPLICATE_MIN_COUNT:
        r.issues.append(f"같은 시각이 중복된 행이 {n_duplicate_rows}개 있습니다.")
    if n_time_regressions > 0:
        r.issues.append(f"시각이 거꾸로 기록된 지점이 {n_time_regressions}곳 있습니다.")

    for c in SENSOR_COLS:
        v = df_quality[c].to_numpy(dtype=float)

        # 고착/포화: 같은 값이 STUCK_RUN행 이상 연속
        same = np.r_[False, (np.diff(v) == 0) & ~np.isnan(np.diff(v))]
        runs = _max_run_lengths(same)
        n_stuck_runs = int((runs >= STUCK_RUN - 1).sum())  # same은 "직전과 같다"이므로 run+1이 실제 길이
        r.n_stuck[c] = n_stuck_runs
        if n_stuck_runs > 0:
            r.issues.append(f"{c}: 같은 값이 {STUCK_RUN}행 이상 반복되는 고착 구간이 {n_stuck_runs}곳 있습니다.")

        # 물리적으로 불가능 + 고립된 값
        bound = IMPOSSIBLE_BOUND[c]
        impossible = np.abs(v) > bound
        n_impossible_total = int(np.nansum(impossible))
        if n_impossible_total > 0:
            run_lens = _max_run_lengths(np.nan_to_num(impossible, nan=0).astype(bool))
            n_isolated = int((run_lens <= ISOLATED_MAX_RUN).sum())
            r.n_impossible_isolated[c] = n_isolated
            if n_isolated > 0:
                r.issues.append(
                    f"{c}: 물리적으로 불가능한 수준(|값|>{bound:g})의 고립된 값이 {n_isolated}곳 있습니다."
                )
        else:
            r.n_impossible_isolated[c] = 0

    r.has_issue = len(r.issues) > 0
    return r


def check_sampling_rate(df_clean: pd.DataFrame) -> tuple[float, bool]:
    """burst(연속 기록) 안의 간격이 10Hz(0.1초)에 맞는지 확인. df_clean은 'seg' 컬럼 포함.
    반환: (0.09~0.11초 비율, 이상 여부)"""
    dt = df_clean["TimeStamp"].diff().dt.total_seconds()
    same_seg = df_clean["seg"].diff().fillna(0) == 0
    inner = dt[same_seg]
    if len(inner) == 0:
        return 1.0, False
    ratio = float(inner.between(0.09, 0.11).mean())
    return ratio, ratio < SAMPLING_OK_RATIO
