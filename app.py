"""
프레스 유압펌프 이상 탐지 모델(하이브리드: 모양 탐지기 + 크기 탐지기) 시연용 로컬 웹앱.
제출 코드(새 폴더/제출/소스코드/)는 수정하지 않고 import해서 사용한다.
실행: python -m streamlit run app.py

화면 구성은 실제 공정관리(SPC) 대시보드 스타일을 참고해, 현장 작업자가 3초 안에
"정상/센서점검/이상" 중 무엇인지, 어느 쪽이 문제인지 한눈에 알 수 있도록 설계했다.
"""
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pdm_app import model_service as ms
from pdm_app import data_quality as dq
from pdm_app import judge
from pdm_app import loader
from pdm_app import samples

st.set_page_config(page_title="프레스 유압펌프 이상 탐지", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# 색상 / 상태 팔레트
# ---------------------------------------------------------------------------
BG = "#0b0d12"
PANEL = "#141821"
PANEL_BORDER = "#262b36"
TEXT = "#e6e8eb"
TEXT_DIM = "#9aa4b2"
GREEN = "#34c759"
GREEN_BG = "#10251a"
AMBER = "#f5a623"
AMBER_BG = "#2b2109"
RED = "#ef4444"
RED_BG = "#2a1414"
BLUE = "#5b8def"
LAVENDER = "#8ab4f8"

STATUS_COLOR = {judge.STATUS_NORMAL: GREEN, judge.STATUS_SENSOR: AMBER, judge.STATUS_ANOMALY: RED}
STATUS_BG = {judge.STATUS_NORMAL: GREEN_BG, judge.STATUS_SENSOR: AMBER_BG, judge.STATUS_ANOMALY: RED_BG}
STATUS_ICON = {judge.STATUS_NORMAL: "✓", judge.STATUS_SENSOR: "⚠", judge.STATUS_ANOMALY: "✕"}
STATUS_CLIP = {
    judge.STATUS_NORMAL: "circle(50% at 50% 50%)",
    judge.STATUS_SENSOR: "polygon(50% 0%, 0% 100%, 100% 100%)",
    judge.STATUS_ANOMALY: "polygon(30% 0%, 70% 0%, 100% 30%, 100% 70%, 70% 100%, 30% 100%, 0% 70%, 0% 30%)",
}
STATUS_BADGE_LABEL = {judge.STATUS_NORMAL: "정상", judge.STATUS_SENSOR: "주의 필요", judge.STATUS_ANOMALY: "위험"}

# ---------------------------------------------------------------------------
# 전역 스타일
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
.block-container {{ padding-top: 2.8rem; padding-bottom: 2rem; max-width: 1500px; }}
header[data-testid="stHeader"] {{ background: {BG}; }}
div[data-testid="stMetric"] {{ display: none; }}

.topbar {{
    display:flex; align-items:center; gap:12px; margin-bottom: 4px;
}}
.topbar .title {{ font-size: 26px; font-weight: 800; color: {TEXT}; }}
.badge-pill {{
    display:inline-flex; align-items:center; gap:6px; padding: 3px 12px; border-radius: 999px;
    font-size: 13px; font-weight: 700;
}}
.src-caption {{ color: {TEXT_DIM}; font-size: 13px; margin-bottom: 10px; }}

.alertbar {{
    border-radius: 10px; padding: 12px 18px; margin-bottom: 16px;
    display:flex; align-items:center; gap: 14px; border: 1px solid;
}}
.alertbar .icon {{ font-size: 20px; }}
.alertbar .msg {{ font-size: 14px; line-height:1.5; }}
.alertbar .msg b {{ font-weight: 800; }}

.statuscard {{
    border-radius: 14px; padding: 20px 26px; display:flex; align-items:center; gap:22px;
    border: 2px solid; margin-bottom: 14px;
}}
.statuscard .shape {{
    width:68px; height:68px; min-width:68px; display:flex; align-items:center; justify-content:center;
    color:white; font-size:32px; font-weight:bold;
}}
.statuscard .status-title {{ font-size: 30px; font-weight: 900; }}
.statuscard .status-reason {{ font-size: 15px; color:{TEXT}; margin-top:3px; }}

.kpi-row {{ display:flex; gap:14px; margin-bottom: 16px; flex-wrap: wrap; }}
.kpi-card {{
    flex: 1 1 220px; background:{PANEL}; border:1px solid {PANEL_BORDER}; border-radius:12px;
    padding: 16px 18px;
}}
.kpi-card .kpi-label {{ color:{TEXT_DIM}; font-size:13px; margin-bottom:6px; }}
.kpi-card .kpi-value {{ font-size:28px; font-weight:800; }}
.kpi-card .kpi-sub {{ font-size:12px; margin-top:4px; font-weight:600; }}

.panel {{
    background:{PANEL}; border:1px solid {PANEL_BORDER}; border-radius:12px; padding:14px 16px 6px 16px;
    margin-bottom: 16px;
}}
.panel .panel-title {{ font-size:14px; font-weight:700; color:{TEXT}; margin-bottom:6px; }}

.status-chip {{ padding:2px 10px; border-radius:999px; font-size:12px; font-weight:700; }}
</style>
""", unsafe_allow_html=True)


def badge_pill(status):
    return (f'<span class="badge-pill" style="background:{STATUS_BG[status]};'
            f'color:{STATUS_COLOR[status]};border:1px solid {STATUS_COLOR[status]};">'
            f'{STATUS_ICON[status]} {STATUS_BADGE_LABEL[status]}</span>')


def kpi_band(value, warn_at, danger_at, higher_is_worse=True):
    """값에 따라 (색, 라벨) 반환 — 카드 하단 상태 표시용."""
    if higher_is_worse:
        if value >= danger_at:
            return RED, "✕ 위험"
        if value >= warn_at:
            return AMBER, "⚠ 주의"
        return GREEN, "✓ 양호"
    else:
        if value <= danger_at:
            return RED, "✕ 위험"
        if value <= warn_at:
            return AMBER, "⚠ 주의"
        return GREEN, "✓ 양호"


# ---------------------------------------------------------------------------
# matplotlib 다크 테마
# ---------------------------------------------------------------------------
with st.spinner("모델을 준비하는 중입니다..."):
    pipeline = ms.get_pipeline()
baseline = pipeline["baseline"]
KOREAN_FONT = ms.exp.korean_font()
plt.rcParams.update({
    "font.family": KOREAN_FONT,
    "axes.unicode_minus": False,
    "figure.facecolor": PANEL,
    "axes.facecolor": PANEL,
    "axes.edgecolor": PANEL_BORDER,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": TEXT_DIM,
    "ytick.color": TEXT_DIM,
    "grid.color": PANEL_BORDER,
    "legend.facecolor": PANEL,
    "legend.edgecolor": PANEL_BORDER,
    "legend.labelcolor": TEXT,
})


# ---------------------------------------------------------------------------
# 사이드바: 입력 선택 + 임곗값/데이터 현황
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙ 입력 데이터")
    source = st.radio("데이터 소스", ["내장 샘플", "CSV 업로드"], horizontal=True, label_visibility="collapsed")

    load_result = None
    sample_label = None
    sample_choice = None

    if source == "내장 샘플":
        sample_choice = st.selectbox(
            "샘플 선택",
            ["정상 데이터 (normal)", "이상 데이터 (outlier)", "센서 오류 예시 (합성)"],
        )
        if sample_choice.startswith("정상"):
            load_result = loader.load_builtin_path(samples.NORMAL_PATH)
            sample_label = "정상 데이터 전체 — ⚠ 학습 데이터 포함, 성능 평가용 아님"
        elif sample_choice.startswith("이상"):
            load_result = loader.load_builtin_path(samples.OUTLIER_PATH)
            sample_label = "이상 데이터 전체 (30 사이클, 약 2.7분)"
        else:
            synth_df = samples.make_sensor_error_sample()
            buf = io.StringIO()
            synth_df.to_csv(buf, index=False)
            buf.seek(0)
            load_result = loader.load_uploaded_csv(buf)
            sample_label = "⚗ 합성 예시 — 결측·고착·이상값·시각역행 인위 주입"
    else:
        up = st.file_uploader("CSV 업로드", type=["csv"],
                               help="필수 컬럼: TimeStamp, AI0_Vibration, AI1_Vibration, AI2_Current")
        if up is not None:
            load_result = loader.load_uploaded_csv(up)
            sample_label = f"업로드 파일: {up.name}"

    if st.button("🔄 재분석 (모델 다시 학습)", width="stretch"):
        ms.get_pipeline.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### 📐 판정 기준")
    st.markdown(f"""
<div style="font-size:13px; color:{TEXT_DIM}; line-height:1.9;">
사이클 위험지수 임곗값 : <b style="color:{TEXT}">1.0</b> (정상 검증 99.5백분위수)<br/>
파일 단위 이상 기준 : <b style="color:{TEXT}">경보 비율 &gt; 5%</b><br/>
&nbsp;&nbsp;(정상 전체 파일 기준 {baseline['normal_full_cycle_alarm_rate']:.2%} / 이상 전체 파일 기준 {baseline['outlier_full_cycle_alarm_rate']:.0%})<br/>
센서 점검 기준 : 결측·고착(≥5행)·물리적 불가능 고립값·시각역행·샘플링 이상
</div>
""", unsafe_allow_html=True)

if sample_label:
    st.sidebar.markdown(f'<div class="src-caption">{sample_label}</div>', unsafe_allow_html=True)

if load_result is None:
    st.info("왼쪽 사이드바에서 내장 샘플을 고르거나 CSV 파일을 업로드하세요.")
    st.stop()

if not load_result.ok:
    st.error("⚠ 데이터를 분석할 수 없습니다.\n\n" + "\n".join(load_result.errors))
    st.stop()

for notice in load_result.notices:
    st.warning(notice)


# ---------------------------------------------------------------------------
# 사이클 추출 + 모델 채점 + 품질 점검 + 판정
# ---------------------------------------------------------------------------
cycles = ms.extract_cycles_with_times(load_result.df)
if len(cycles) == 0:
    st.error("⚠ 추출된 사이클이 없어 분석할 수 없습니다. 연속된 기록이 더 긴 데이터를 올려주세요.")
    st.stop()

table_df = ms.score_cycles(pipeline, cycles)

quality = dq.check_quality(
    load_result.df_quality, load_result.n_missing, load_result.n_non_numeric,
    load_result.n_duplicate_rows, load_result.n_time_regressions,
)
sampling_ratio, sampling_bad = dq.check_sampling_rate(load_result.df)
verdict = judge.decide(table_df, quality.has_issue, quality.issues, sampling_bad, sampling_ratio)

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 데이터 현황")
    d_full = load_result.df
    period = f"{d_full['TimeStamp'].min():%Y-%m-%d %H:%M} ~ {d_full['TimeStamp'].max():%H:%M:%S}"
    st.markdown(f"""
<div style="font-size:13px; color:{TEXT_DIM}; line-height:1.9;">
사이클 수 : <b style="color:{TEXT}">{verdict.n_cycles:,}</b> &nbsp;|&nbsp;
burst 수 : <b style="color:{TEXT}">{verdict.n_bursts:,}</b><br/>
평균 위험지수 : <b style="color:{TEXT}">{table_df['risk_index'].mean():.3f}</b> &nbsp;|&nbsp;
표준편차 : <b style="color:{TEXT}">{table_df['risk_index'].std():.3f}</b><br/>
기간 : <b style="color:{TEXT}">{period}</b>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 상단 바: 제목 + 상태 배지
# ---------------------------------------------------------------------------
st.markdown(
    f'<div class="topbar"><span class="title">🏭 프레스 유압펌프 이상 탐지</span>{badge_pill(verdict.status)}</div>',
    unsafe_allow_html=True,
)

if sample_choice and sample_choice.startswith("센서"):
    st.caption("※ 이 샘플은 합성 예시입니다. 실제 설비에서 관찰된 데이터가 아닙니다.")
if sample_choice and sample_choice.startswith("이상"):
    st.caption("※ 이 이상 데이터는 값이 정상 범위를 넓게 벗어나지만, 고착·포화·시각역행 없이 연속적으로 변하는 "
               "진짜 이상 신호입니다 (센서 오류가 아닌 탐지 대상).")

# ---------------------------------------------------------------------------
# 탭 구성
# ---------------------------------------------------------------------------
tab_names = ["📊 대시보드", "📈 원신호 분석", "🔍 데이터 품질", "📋 사이클별 결과"]
if load_result.has_label:
    tab_names.append("🎯 정답 비교")
tabs = st.tabs(tab_names)


# =============================== 탭 1: 대시보드 ===============================
with tabs[0]:
    n_shape = int(table_df["shape_alarm"].sum())
    n_amp = int(table_df["amp_alarm"].sum())
    n_quality_items = len(quality.issues) + (1 if sampling_bad else 0)

    color = STATUS_COLOR[verdict.status]
    bg = STATUS_BG[verdict.status]

    # ---- 상태 카드 ----
    st.markdown(
        f"""
<div class="statuscard" style="background:{bg};border-color:{color};">
  <div class="shape" style="background:{color};clip-path:{STATUS_CLIP[verdict.status]};">{STATUS_ICON[verdict.status]}</div>
  <div>
    <div class="status-title" style="color:{color};">{verdict.status}</div>
    <div class="status-reason">{verdict.reason}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ---- 알림 바: 넬슨 룰 스타일 위반 내역 ----
    if verdict.status == judge.STATUS_ANOMALY:
        parts = [f"모양 경보 <b>{n_shape}건</b>", f"크기 경보(2사이클 연속) <b>{n_amp}건</b>",
                 f"경보 사이클 <b>{verdict.n_alarm_cycles}/{verdict.n_cycles}건 ({verdict.alarm_cycle_ratio:.1%})</b>",
                 f"경보 burst <b>{verdict.n_alarm_bursts}/{verdict.n_bursts}건 ({verdict.alarm_burst_ratio:.1%})</b>"]
        if n_quality_items > 0:
            parts.append(f"데이터 품질 이슈 <b>{n_quality_items}건 동반</b>")
        alert_msg = " · ".join(parts)
        alert_icon = "✕"
    elif verdict.status == judge.STATUS_SENSOR:
        all_issues = list(quality.issues)
        if sampling_bad:
            all_issues.append(f"샘플링 간격 이상 (0.09~0.11초 비율 {sampling_ratio:.0%})")
        alert_msg = " · ".join(f"<b>{s}</b>" for s in all_issues)
        alert_icon = "⚠"
    else:
        alert_msg = (f"모양·크기 경보 없음 · 경보 사이클 <b>{verdict.n_alarm_cycles}/{verdict.n_cycles}건 "
                     f"({verdict.alarm_cycle_ratio:.1%})</b> · 데이터 품질 문제 없음")
        alert_icon = "✓"

    st.markdown(
        f"""<div class="alertbar" style="background:{bg};border-color:{color};">
<span class="icon" style="color:{color};">{alert_icon}</span>
<span class="msg" style="color:{TEXT};">{alert_msg}</span>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown(f"**권고 조치:** {verdict.advice}")
    if verdict.secondary:
        st.markdown(f":warning: {verdict.secondary}")
    if sample_choice and sample_choice.startswith("정상"):
        st.caption(f"참고(baseline): 정상 전체 파일 경보 사이클 비율 {baseline['normal_full_cycle_alarm_rate']:.2%}, "
                   f"경보 burst 비율 {baseline['normal_full_burst_alarm_rate']:.2%} "
                   f"(학습에 쓴 사이클이 포함돼 있어 실제보다 낮게 나올 수 있음)")

    st.write("")

    # ---- KPI 카드 4개 ----
    max_risk = verdict.max_risk_index
    mean_prob = float(table_df["anomaly_prob"].mean())
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "최대 위험지수", f"{max_risk:.2f}", "임곗값 1.0 기준", kpi_band(max_risk, 1.0, 3.0)),
        (c2, "경보 사이클 비율", f"{verdict.alarm_cycle_ratio:.1%}",
         f"{verdict.n_alarm_cycles}/{verdict.n_cycles}건", kpi_band(verdict.alarm_cycle_ratio, 0.05, 0.20)),
        (c3, "경보 burst 비율", f"{verdict.alarm_burst_ratio:.1%}",
         f"{verdict.n_alarm_bursts}/{verdict.n_bursts}건", kpi_band(verdict.alarm_burst_ratio, 0.05, 0.20)),
        (c4, "평균 이상확률", f"{mean_prob:.1%}", "anomaly_prob 평균", kpi_band(mean_prob, 0.10, 0.50)),
    ]
    for col, label, value, sub, (bandc, bandl) in cards:
        with col:
            st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value" style="color:{bandc};">{value}</div>
  <div class="kpi-sub" style="color:{TEXT_DIM};">{sub}</div>
  <div class="kpi-sub" style="color:{bandc};">{bandl}</div>
</div>
""", unsafe_allow_html=True)

    st.write("")

    # ---- 히스토그램 + 관리도 ----
    col_hist, col_ctrl = st.columns(2)
    with col_hist:
        st.markdown(f'<div class="panel"><div class="panel-title">위험지수 분포</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.4))
        vals = table_df["risk_index"].dropna().values
        bins = np.linspace(min(0, vals.min()), max(vals.max(), 1.2), 30)
        colors_bin = []
        n, bin_edges, patches = ax.hist(vals, bins=bins, edgecolor=PANEL, linewidth=0.5)
        for patch, edge in zip(patches, bin_edges[:-1]):
            patch.set_facecolor(RED if edge >= 1.0 else LAVENDER)
        ax.axvline(1.0, color=RED, linestyle="--", linewidth=1.3, label="임곗값(=1)")
        ax.set_xlabel("위험지수")
        ax.set_ylabel("사이클 수")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_ctrl:
        st.markdown(f'<div class="panel"><div class="panel-title">사이클 관리도 (위험지수 추이)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.4))
        x = np.arange(len(table_df))
        y = table_df["risk_index"].values
        shape_only = table_df["shape_alarm"].astype(bool).values & ~table_df["amp_alarm"].astype(bool).values
        amp_only = table_df["amp_alarm"].astype(bool).values & ~table_df["shape_alarm"].astype(bool).values
        both = table_df["shape_alarm"].astype(bool).values & table_df["amp_alarm"].astype(bool).values
        normal = ~(shape_only | amp_only | both)
        ax.axhspan(0, 1.0, color=GREEN, alpha=0.06, zorder=0)
        ax.plot(x, y, color=TEXT_DIM, linewidth=0.8, zorder=1, alpha=0.6)
        ax.scatter(x[normal], y[normal], color=LAVENDER, s=16, zorder=2, label="정상")
        ax.scatter(x[shape_only], y[shape_only], color=AMBER, s=30, zorder=3, label="모양 경보")
        ax.scatter(x[amp_only], y[amp_only], color=RED, s=30, zorder=3, label="크기 경보")
        ax.scatter(x[both], y[both], color="#b00020", s=36, zorder=4, label="모양+크기")
        ax.axhline(1.0, color=RED, linestyle="--", linewidth=1.3, label="UCL(임곗값=1)")
        ax.set_xlabel("사이클 순서")
        ax.set_ylabel("위험지수")
        ax.legend(loc="upper left", fontsize=7, ncol=2)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- burst별 요약 표 (문제 있는 구간이 맨 위로) ----
    st.markdown(f'<div class="panel"><div class="panel-title">burst(연속 기록 구간)별 요약 — 경보 발생 구간이 맨 위</div>', unsafe_allow_html=True)
    bsum = table_df.groupby("burst_id").agg(
        사이클수=("pred", "size"), 경보수=("pred", "sum"), 최대위험지수=("risk_index", "max"),
        시작=("t_start", "min"), 끝=("t_end", "max"),
    ).reset_index().rename(columns={"burst_id": "burst"})
    bsum["상태"] = np.where(bsum["경보수"] > 0, "경보", "정상")
    bsum["최대위험지수"] = bsum["최대위험지수"].round(2)
    bsum = bsum.sort_values(["경보수", "최대위험지수"], ascending=[False, False]).reset_index(drop=True)

    def _style_status(s):
        return [f"background-color:{RED_BG};color:{RED};font-weight:700;" if v == "경보"
                else f"color:{GREEN};font-weight:700;" for v in s]

    styled = bsum.style.apply(_style_status, subset=["상태"])
    st.dataframe(styled, width="stretch", hide_index=True, height=280)
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(
        "이상 확률(anomaly_prob)은 정상 검증 사이클 + 내장 이상 샘플로 보정한 로지스틱 회귀 결과입니다. "
        "⚠ 이 보정은 내장 이상 샘플을 학습에 사용했으므로, 처음 보는 새 이상 유형에는 그대로 일반화되지 않을 수 있습니다."
    )


# =============================== 탭 2: 원신호 분석 ===============================
with tabs[1]:
    st.markdown(f'<div class="panel"><div class="panel-title">원신호 3채널 (빨간 음영 = 경보 사이클 구간)</div>', unsafe_allow_html=True)
    d = load_result.df.reset_index(drop=True)
    alarm_rows = np.zeros(len(d), dtype=bool)
    alarm_cycles = table_df[table_df["pred"] == 1]
    for _, row in alarm_cycles.iterrows():
        m_rows = (d["TimeStamp"] >= row["t_start"]) & (d["TimeStamp"] <= row["t_end"])
        alarm_rows |= m_rows.values

    fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
    labels = ["AI0_Vibration (상부 진동)", "AI1_Vibration (하부 진동)", "AI2_Current (전류)"]
    for ax, col, lab in zip(axes, ms.S, labels):
        ax.plot(np.arange(len(d)), d[col], color=BLUE, linewidth=0.6)
        ax.set_ylabel(lab, fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        if alarm_rows.any():
            runs = np.flatnonzero(np.diff(np.r_[0, alarm_rows.astype(int), 0]))
            for s, e in zip(runs[0::2], runs[1::2]):
                ax.axvspan(s, e, color=RED, alpha=0.18)
    axes[-1].set_xlabel("행 순서(row index)")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)


# =============================== 탭 3: 데이터 품질 ===============================
with tabs[2]:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown(f'<div class="panel"><div class="panel-title">점검 항목</div>', unsafe_allow_html=True)
        rows = [
            ("결측값", quality.n_missing),
            ("비숫자 값", quality.n_non_numeric),
            ("중복 시각 행", quality.n_duplicate_rows),
            ("시각 역행", quality.n_time_regressions),
            ("샘플링 간격 0.09~0.11초 비율", f"{sampling_ratio:.1%}"),
        ]
        for c in dq.SENSOR_COLS:
            rows.append((f"{c} 고착 구간(≥{dq.STUCK_RUN}행)", quality.n_stuck.get(c, 0)))
            rows.append((f"{c} 물리적으로 불가능한 고립값", quality.n_impossible_isolated.get(c, 0)))
        qdf = pd.DataFrame([(k, str(v)) for k, v in rows], columns=["점검 항목", "값"])
        st.dataframe(qdf, width="stretch", hide_index=True, height=360)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_b:
        st.markdown(f'<div class="panel"><div class="panel-title">발견된 문제</div>', unsafe_allow_html=True)
        if quality.issues:
            for s in quality.issues:
                st.markdown(f'<div style="color:{AMBER};padding:4px 0;">⚠ {s}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="color:{GREEN};padding:4px 0;">✓ 발견된 문제가 없습니다.</div>', unsafe_allow_html=True)
        if sampling_bad:
            st.markdown(f'<div style="color:{AMBER};padding:4px 0;">⚠ 샘플링 간격이 10Hz와 다릅니다 '
                        f'(0.09~0.11초 비율 {sampling_ratio:.0%})</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# =============================== 탭 4: 사이클별 결과 ===============================
with tabs[3]:
    st.markdown(f'<div class="panel"><div class="panel-title">사이클별 결과 표</div>', unsafe_allow_html=True)
    show_cols = ["burst_id", "cycle_in_burst", "full_cycle", "t_start", "t_end", "n_rows",
                 "shape_dist", "amp_score", "risk_index", "anomaly_prob",
                 "shape_alarm", "amp_alarm", "pred"]
    disp = table_df[show_cols].copy()

    def _style_pred(s):
        return [f"background-color:{RED_BG};color:{RED};font-weight:700;" if v == 1 else "" for v in s]

    st.dataframe(disp.style.apply(_style_pred, subset=["pred"]), width="stretch", hide_index=True, height=420)
    csv_bytes = table_df[show_cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇ CSV로 다운로드", data=csv_bytes, file_name="cycle_results.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)


# =============================== 탭 5: 정답 비교 ===============================
if load_result.has_label:
    with tabs[4]:
        st.markdown(f'<div class="panel"><div class="panel-title">정답(Equipment_state) 비교 — 혼동행렬</div>', unsafe_allow_html=True)
        d = load_result.df.reset_index(drop=True)
        cyc_true = []
        for _, row in table_df.iterrows():
            m_rows = (d["TimeStamp"] >= row["t_start"]) & (d["TimeStamp"] <= row["t_end"])
            lab = d.loc[m_rows, "Equipment_state"]
            cyc_true.append(int(round(lab.mean())) if len(lab) else 0)
        y_true = np.array(cyc_true)
        y_pred = table_df["pred"].values
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        cm = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=pd.Index(["실제 정상(0)", "실제 이상(1)"], name=""),
            columns=["예측 정상(0)", "예측 이상(1)"],
        )
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(cm, width="stretch")
            st.caption(f"정확도 {((tn+tp)/max(1,len(y_true))):.1%} · "
                       f"정밀도 {(tp/max(1,tp+fp)):.1%} · 재현율 {(tp/max(1,tp+fn)):.1%}")
        st.markdown("</div>", unsafe_allow_html=True)
