"""
제출 코드(pdm_hybrid.py, experiment.py)를 그대로 import해서
앱 시작 시 한 번 학습하고 결과를 캐시하는 모듈.
제출 코드 파일은 절대 수정하지 않는다.

경로 우선순위:
  1) web_app/vendor/ — 제출/소스코드의 pdm_hybrid.py·experiment.py·data/를 그대로 복사해 둔
     배포용 사본(클라우드 배포 시 저장소가 web_app 폴더만 담아도 되도록 자기완결적으로 구성).
     내용은 제출 코드와 바이트 단위로 동일하고, 독자적으로 수정하지 않는다.
  2) 1)이 없을 때만 새 폴더/제출/소스코드/ 를 직접 sys.path로 추가 (로컬 전체 프로젝트 구조에서 실행할 때).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

_VENDORED = Path(__file__).resolve().parent.parent / "vendor"
_ORIGINAL = Path(__file__).resolve().parent.parent.parent / "제출" / "소스코드"
SUBMISSION_SRC = _VENDORED if (_VENDORED / "pdm_hybrid.py").exists() else _ORIGINAL
if str(SUBMISSION_SRC) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_SRC))

import pdm_hybrid as ph          # noqa: E402  (제출 코드, 수정 금지)
import experiment as exp         # noqa: E402  (제출 코드, 수정 금지)

S = ph.S  # ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']


@st.cache_resource(show_spinner="모델 학습 중 (정상 데이터 기반, 최초 1회만 실행)...")
def get_pipeline():
    """제출 코드와 동일한 절차로 학습하고, Platt 보정 로지스틱까지 앱 시작 시 한 번만 학습해서 캐시.

    - P['m'] : HybridDetector (정상 학습 사이클로 학습, 정상 검증 99.5 백분위수로 임곗값 - experiment.prepare() 그대로)
    - calibrator : risk_index -> anomaly_prob 로지스틱 보정기.
      학습 데이터 = 정상 검증 사이클(label 0) + 이상 파일 전체 사이클(label 1).
      ※ 이 보정기는 내장 이상 샘플을 "본" 상태로 학습되므로, 화면에 반드시
        "보정용으로 내장 이상 샘플 사용"이라고 표시해야 한다 (app.py에서 처리).
    """
    P = exp.prepare()
    m = P["m"]

    va_tbl = exp.table(P, P["va"], "val", 0)
    co_tbl = exp.table(P, P["co"], "outlier_calib", 1)
    X = pd.concat([va_tbl[["risk_index"]], co_tbl[["risk_index"]]], ignore_index=True)
    y = np.r_[np.zeros(len(va_tbl)), np.ones(len(co_tbl))]
    calibrator = LogisticRegression(C=1.0).fit(X, y)

    # ---- 파일 단위 "이상" 판정 기준 산출에 쓰는 참고 통계 ----
    # 정상 파일 전체(학습에 쓴 구간 포함) 기준 사이클/burst 경보 비율.
    fs, fa, f = m.predict(P["cn"])
    seg_f = pd.DataFrame({"seg": [c["seg"] for c in P["cn"]], "f": f}).groupby("seg").f.max()
    normal_full_cycle_alarm_rate = float(f.mean())
    normal_full_burst_alarm_rate = float(seg_f.mean())

    # 이상 파일 전체 기준 사이클/burst 경보 비율 (이상 탐지 대상 그 자체).
    gs, ga, g = m.predict(P["co"])
    seg_g = pd.DataFrame({"seg": [c["seg"] for c in P["co"]], "f": g}).groupby("seg").f.max()
    outlier_full_cycle_alarm_rate = float(g.mean())
    outlier_full_burst_alarm_rate = float(seg_g.mean())

    baseline = dict(
        normal_full_cycle_alarm_rate=normal_full_cycle_alarm_rate,
        normal_full_burst_alarm_rate=normal_full_burst_alarm_rate,
        outlier_full_cycle_alarm_rate=outlier_full_cycle_alarm_rate,
        outlier_full_burst_alarm_rate=outlier_full_burst_alarm_rate,
    )
    return dict(P=P, m=m, calibrator=calibrator, baseline=baseline)


def extract_cycles_with_times(d):
    """업로드/내장 데이터프레임(d: TimeStamp, S..., seg 포함)에서 사이클을 추출하고
    experiment.table()이 요구하는 t_start/n_rows를 붙인다 (experiment.with_times 그대로 재사용)."""
    cycles = ph.extract_cycles(d)
    return exp.with_times(cycles, d)


def score_cycles(pipeline, cycles):
    """임의의 사이클 목록(이미 with_times로 보강된 것)에 대해 하이브리드 탐지기 점수표를 만든다
    (제출 코드 table()/risk() 재사용). label은 알 수 없으므로 전부 0으로 채워 넣고(표 형식만 맞추기 위함),
    실제 정답(Equipment_state) 비교는 앱에서 별도로 처리."""
    P = pipeline["P"]
    D = exp.table(P, cycles, split="input", label=0)
    proba = pipeline["calibrator"].predict_proba(D[["risk_index"]].values)[:, 1]
    D["anomaly_prob"] = proba
    return D
