"""
업로드된 CSV를 제출 코드(pdm_hybrid.load → extract_cycles)에 넣을 수 있는 형태로
검증·전처리한다. 제출 코드 파일은 수정하지 않고, 같은 규칙(중복 제거, burst 분할)을
이 모듈에서 그대로 재현해 업로드 데이터에도 적용한다.
"""
from dataclasses import dataclass, field

import pandas as pd

REQUIRED_COLS = ["TimeStamp", "AI0_Vibration", "AI1_Vibration", "AI2_Current"]
OPTIONAL_LABEL_COL = "Equipment_state"
MIN_ROWS_FOR_CYCLE = 10  # pdm_hybrid.extract_cycles의 min_rows와 동일


@dataclass
class LoadResult:
    ok: bool
    df: pd.DataFrame = None            # 모델 입력용: 유효 행만, 중복 제거 + seg 포함
    df_quality: pd.DataFrame = None    # 품질 점검용: 원본 순서 그대로, 결측/비숫자 포함
    has_label: bool = False
    errors: list = field(default_factory=list)       # 치명적 오류 (진행 불가)
    notices: list = field(default_factory=list)       # 참고용 안내 (진행은 가능)
    n_duplicate_rows: int = 0
    n_time_regressions: int = 0
    n_missing: int = 0
    n_non_numeric: int = 0
    n_dropped_invalid: int = 0


def _strip_unnamed_index(raw: pd.DataFrame) -> pd.DataFrame:
    """맨 앞의 이름 없는 인덱스 열(Unnamed: 0 등)이 있으면 제거."""
    if len(raw.columns) == 0:
        return raw
    first = raw.columns[0]
    if str(first).startswith("Unnamed") or str(first).strip() == "":
        return raw.drop(columns=[first])
    return raw


def load_uploaded_csv(file_like) -> LoadResult:
    """업로드된 CSV 파일(file-like 또는 경로)을 읽고 검증한다.
    실패해도 예외를 던지지 않고 LoadResult.errors에 한국어 안내를 담아 반환한다."""
    try:
        raw = pd.read_csv(file_like)
    except Exception as e:
        return LoadResult(ok=False, errors=[f"CSV 파일을 읽을 수 없습니다 (형식 오류): {e}"])

    if raw.shape[0] == 0 or raw.shape[1] == 0:
        return LoadResult(ok=False, errors=["빈 CSV 파일입니다. 데이터가 있는 파일을 올려주세요."])

    raw = _strip_unnamed_index(raw)

    missing_cols = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing_cols:
        return LoadResult(
            ok=False,
            errors=[
                "필수 컬럼이 없습니다: " + ", ".join(missing_cols)
                + f" (필요한 컬럼: {', '.join(REQUIRED_COLS)})"
            ],
        )

    has_label = OPTIONAL_LABEL_COL in raw.columns
    d = raw[REQUIRED_COLS + ([OPTIONAL_LABEL_COL] if has_label else [])].copy()

    notices = []

    # ---- 시각 컬럼 ----
    ts_before_na = d["TimeStamp"].isna().sum()
    ts = pd.to_datetime(d["TimeStamp"], errors="coerce")
    n_bad_ts = int(ts.isna().sum() - ts_before_na)
    if n_bad_ts > 0:
        notices.append(f"TimeStamp 값 중 {n_bad_ts}개를 시각으로 해석할 수 없습니다.")
    d["TimeStamp"] = ts

    # ---- 센서 컬럼: 숫자로 변환 (실패하면 결측으로) ----
    n_non_numeric = 0
    for c in ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]:
        before_na = int(d[c].isna().sum())
        d[c] = pd.to_numeric(d[c], errors="coerce")
        after_na = int(d[c].isna().sum())
        n_non_numeric += after_na - before_na

    n_missing = int(d[REQUIRED_COLS].isna().sum().sum())
    df_quality = d.copy()  # 원본 순서 그대로 (품질 점검표: 결측·비숫자 그대로 보존)

    valid_mask = d["TimeStamp"].notna() & d[["AI0_Vibration", "AI1_Vibration", "AI2_Current"]].notna().all(axis=1)
    if valid_mask.sum() == 0:
        return LoadResult(
            ok=False,
            errors=["TimeStamp 또는 센서 값이 모두 비어 있거나 숫자가 아니어서 분석할 수 없습니다."],
            df_quality=df_quality, n_missing=n_missing, n_non_numeric=n_non_numeric,
        )

    n_dropped_invalid = int((~valid_mask).sum())
    if n_dropped_invalid > 0:
        notices.append(f"결측·비숫자 값이 있는 {n_dropped_invalid}행은 모델 입력에서 제외했습니다 (데이터 품질 점검표에는 표시됩니다).")

    d_clean = d[valid_mask].reset_index(drop=True)

    if len(d_clean) < MIN_ROWS_FOR_CYCLE:
        return LoadResult(
            ok=False,
            errors=[
                f"유효한 행이 {len(d_clean)}개뿐이라 사이클을 추출할 수 없습니다 "
                f"(burst 당 최소 {MIN_ROWS_FOR_CYCLE}행 필요). 더 많은 데이터를 포함한 CSV를 올려주세요."
            ],
            df_quality=df_quality, n_missing=n_missing, n_non_numeric=n_non_numeric,
        )

    # ---- 시각 역행 / 중복 (pdm_hybrid.load와 동일 규칙: 중복 제거 후 burst 분할) ----
    n_time_regressions = int((d_clean["TimeStamp"].diff().dt.total_seconds() < 0).sum())
    n_duplicate_rows = int(d_clean.duplicated(subset="TimeStamp").sum())

    d_clean = d_clean.sort_values("TimeStamp", kind="stable")
    d_clean = d_clean.drop_duplicates(subset="TimeStamp").reset_index(drop=True)
    gap = d_clean["TimeStamp"].diff().dt.total_seconds().fillna(0.1)
    d_clean["seg"] = (gap > 0.2).cumsum()

    usable_bursts = d_clean.groupby("seg").size()
    if (usable_bursts >= MIN_ROWS_FOR_CYCLE).sum() == 0:
        return LoadResult(
            ok=False,
            errors=[
                f"모든 구간(burst)의 길이가 {MIN_ROWS_FOR_CYCLE}행 미만이라 사이클을 하나도 만들 수 없습니다. "
                "연속된 기록이 더 긴 데이터를 올려주세요."
            ],
            df_quality=df_quality, n_missing=n_missing, n_non_numeric=n_non_numeric,
        )

    return LoadResult(
        ok=True,
        df=d_clean,
        df_quality=df_quality,
        has_label=has_label,
        errors=[],
        notices=notices,
        n_duplicate_rows=n_duplicate_rows,
        n_time_regressions=n_time_regressions,
        n_missing=n_missing,
        n_non_numeric=n_non_numeric,
        n_dropped_invalid=n_dropped_invalid,
    )


def load_builtin_path(path) -> LoadResult:
    """내장 샘플(정상/이상 원본 CSV)은 pdm_hybrid.load와 완전히 같은 결과를 내도록
    load_uploaded_csv를 그대로 통과시킨다 (파일 경로를 그대로 read_csv에 넘김)."""
    return load_uploaded_csv(path)
