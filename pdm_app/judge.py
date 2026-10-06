"""
파일(세션) 단위 설비 상태 판정: 정상 작동 / 센서 점검 필요 / 이상 발생.
우선순위: 이상 > 센서 오류 > 정상.

파일 단위 "이상" 판정 기준 (경보 사이클 비율, 경보 burst 비율 중 하나라도 넘으면 이상):
  - 근거 수치 (web_app 개발 중 model_service.get_pipeline()의 baseline으로 측정):
      정상 파일 전체(학습 구간 포함, 756사이클/530burst): 경보 사이클 비율 0.40%(3/756),
                                                        경보 burst 비율 0.57%(3/530)
      이상 파일 전체(30사이클/17burst):                경보 사이클 비율 100%(30/30),
                                                        경보 burst 비율 100%(17/17)
  - 두 값 사이에 극단적인 차이가 있어, 안전 마진을 크게 두고 5%를 기준으로 잡았다
    (정상 파일 기준치의 약 9~12배, 이상 파일 기준치의 1/20 수준).
"""
from dataclasses import dataclass, field

ANOMALY_CYCLE_RATIO_THRESH = 0.05
ANOMALY_BURST_RATIO_THRESH = 0.05

STATUS_NORMAL = "정상 작동"
STATUS_SENSOR = "센서 점검 필요"
STATUS_ANOMALY = "이상 발생"


@dataclass
class Verdict:
    status: str
    reason: str                      # 카드 아래 한 줄 근거
    advice: str                      # 권고 조치 한 줄
    secondary: str = ""              # 이상인데 센서 문제도 있을 때 보조 경고
    alarm_cycle_ratio: float = 0.0
    alarm_burst_ratio: float = 0.0
    n_cycles: int = 0
    n_alarm_cycles: int = 0
    n_bursts: int = 0
    n_alarm_bursts: int = 0
    max_risk_index: float = 0.0
    shape_dominant: bool = False      # True면 모양 경보 우세, False면 크기 경보 우세


def decide(table_df, quality_has_issue: bool, quality_issues: list, sampling_bad: bool,
           sampling_ratio: float) -> Verdict:
    """table_df: model_service.score_cycles()가 만든 사이클별 결과표 (pred, shape_alarm, amp_alarm, risk_index, burst_id 포함)."""
    n_cycles = len(table_df)
    n_alarm_cycles = int(table_df["pred"].sum())
    alarm_cycle_ratio = n_alarm_cycles / n_cycles if n_cycles else 0.0

    burst_alarm = table_df.groupby("burst_id")["pred"].max()
    n_bursts = len(burst_alarm)
    n_alarm_bursts = int(burst_alarm.sum())
    alarm_burst_ratio = n_alarm_bursts / n_bursts if n_bursts else 0.0

    max_risk = float(table_df["risk_index"].max()) if n_cycles else 0.0
    n_shape = int(table_df["shape_alarm"].sum())
    n_amp = int(table_df["amp_alarm"].sum())

    is_anomaly = (alarm_cycle_ratio > ANOMALY_CYCLE_RATIO_THRESH) or (alarm_burst_ratio > ANOMALY_BURST_RATIO_THRESH)
    sensor_issue = quality_has_issue or sampling_bad

    v = Verdict(
        status=STATUS_NORMAL, reason="", advice="",
        alarm_cycle_ratio=alarm_cycle_ratio, alarm_burst_ratio=alarm_burst_ratio,
        n_cycles=n_cycles, n_alarm_cycles=n_alarm_cycles, n_bursts=n_bursts, n_alarm_bursts=n_alarm_bursts,
        max_risk_index=max_risk, shape_dominant=n_shape >= n_amp,
    )

    if is_anomaly:
        v.status = STATUS_ANOMALY
        v.reason = f"전체 {n_cycles} 사이클 중 {n_alarm_cycles}개 경보 · 최대 위험지수 {max_risk:.1f}"
        if v.shape_dominant:
            v.advice = "모양 경보 우세 → 전기·구동·부하 계통 우선 점검"
        else:
            v.advice = "크기 경보 우세 → 펌프 체결·베어링 등 기계 계통 우선 점검"
        if sensor_issue:
            v.secondary = "데이터 품질 문제도 함께 발견됨: " + " / ".join(quality_issues[:3])
            if sampling_bad:
                v.secondary += f" / 샘플링 간격이 10Hz와 다름(0.09~0.11초 비율 {sampling_ratio:.0%})"
        return v

    if sensor_issue:
        v.status = STATUS_SENSOR
        all_issues = list(quality_issues)
        if sampling_bad:
            all_issues.append(f"샘플링 간격이 10Hz(0.1초)와 다름 (0.09~0.11초 비율 {sampling_ratio:.0%})")
        v.reason = " / ".join(all_issues)
        v.advice = "원인 센서·기록을 확인한 뒤 재점검: " + (all_issues[0] if all_issues else "")
        return v

    v.status = STATUS_NORMAL
    v.reason = f"전체 {n_cycles} 사이클 중 경보 {n_alarm_cycles}개 · 최대 위험지수 {max_risk:.1f} (임곗값 1)"
    v.advice = "특별한 조치가 필요하지 않습니다."
    return v
