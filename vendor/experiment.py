"""
공통 준비 모듈: 데이터 로드 → 사이클 추출(시각 포함) → 시간 블록 분할 → 하이브리드 학습 → 사이클 특징표
predict.py, compare_models.py, error_analysis.py가 모두 이 모듈을 통해 같은 데이터·분할·모델을 쓴다.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import find_peaks
from sklearn.model_selection import GroupKFold
from pdm_hybrid import S, Q, load, extract_cycles, blocked_split, HybridDetector, amp_features

ROOT = Path(__file__).resolve().parent
DATA, OUT, FIG = ROOT / 'data', ROOT / 'results', ROOT / 'results' / 'figures'
for p in (OUT, FIG): p.mkdir(parents=True, exist_ok=True)

LOW_ROWS = (15900, 18400)      # 정상 파일의 저부하 구간(행 번호, EDA로 확인)
N_FOLDS = 5                    # 지도학습·확률보정용 그룹 교차검증 fold 수(이상 burst 단위)
AMP_COLS = ['v0_rms', 'v1_rms', 'v0_p2p', 'v1_p2p', 'cur_rms', 'cur_p2p', 'cur_diffstd', 'corr01']


def cycle_times(d, min_rows=10):
    """extract_cycles와 같은 규칙으로 각 사이클의 시작 시각과 행 수를 구한다."""
    out = []
    for _, g in d.groupby('seg'):
        if len(g) < min_rows: continue
        ts = g['TimeStamp'].to_numpy()
        pk, _ = find_peaks(g[S[2]].values, distance=10)
        if len(pk) > 1:
            out += [(ts[a], ts[b], b - a + 1) for a, b in zip(pk[:-1], pk[1:])]
        else:
            out.append((ts[0], ts[-1], len(g)))
    return out


def with_times(cycles, d):
    tt = cycle_times(d)
    assert len(tt) == len(cycles) and all(pd.Timestamp(e) == c['t'] for (_, e, _), c in zip(tt, cycles))
    return [{**c, 't_start': pd.Timestamp(s), 'n_rows': n} for (s, _, n), c in zip(tt, cycles)]


def prepare():
    n, o = load(DATA / 'press_data_normal.csv'), load(DATA / 'outlier_data.csv')
    cn, co = with_times(extract_cycles(n), n), with_times(extract_cycles(o), o)
    tr, va, te = blocked_split(cn)
    m = HybridDetector().fit(tr, va)                      # 정상 학습 데이터로 학습, 정상 검증 데이터로 임곗값
    sv, av = m.scores(va)
    ref = dict(med_s=np.nanmedian(sv), med_a=np.median(av))
    low = (n.TimeStamp.iloc[LOW_ROWS[0]], n.TimeStamp.iloc[LOW_ROWS[1]])
    # 진동-부하 관계: 정상 학습 사이클에서 전류 진폭 → 진동 RMS 회귀, 잔차 최댓값을 '진동 과다' 기준으로
    A = pd.DataFrame([amp_features(c['x']) for c in tr])
    vib = {c: (np.polyfit(A.cur_rms, A[c], 1), None) for c in ['v0_rms', 'v1_rms']}
    vib = {c: (b, (A[c] - np.polyval(b, A.cur_rms)).max()) for c, (b, _) in vib.items()}
    return dict(n=n, o=o, cn=cn, co=co, tr=tr, va=va, te=te, m=m, ref=ref, low=low, vib=vib)


def risk(P, m, cycles, th_shape=None, th_amp=None):
    """임곗값=1, 정상 검증 중앙값=0이 되도록 두 점수를 정규화하고 큰 쪽을 위험지수로 쓴다."""
    th_shape = m.th_shape if th_shape is None else th_shape
    th_amp = m.th_amp if th_amp is None else th_amp
    sh, am = m.scores(cycles)
    rs = (sh - P['ref']['med_s']) / (th_shape - P['ref']['med_s'])
    ra = (am - P['ref']['med_a']) / (th_amp - P['ref']['med_a'])
    return sh, am, rs, ra, np.fmax(rs, ra)                 # 부분 사이클(rs=NaN)은 ra만 사용


def table(P, cycles, split, label):
    """사이클 1개 = 1행. 메타정보, 원특징, 두 점수, 정규화 위험지수, 하이브리드 경보."""
    m = P['m']
    fs, fa, f = m.predict(cycles)
    sh, am, rs, ra, rk = risk(P, m, cycles)
    A = pd.DataFrame([amp_features(c['x']) for c in cycles])
    D = pd.DataFrame({
        'split': split, 'burst_id': [c['seg'] for c in cycles], 'cycle_in_burst': [c['k'] for c in cycles],
        'full_cycle': [c['full'] for c in cycles], 't_start': [c['t_start'] for c in cycles],
        't_end': [c['t'] for c in cycles], 'n_rows': [c['n_rows'] for c in cycles]})
    D = pd.concat([D, A], axis=1)
    D['shape_dist'], D['amp_score'], D['shape_ratio'], D['amp_ratio'], D['risk_index'] = sh, am, rs, ra, rk
    D['shape_alarm'], D['amp_alarm'], D['pred'] = fs.astype(int), fa.astype(int), f.astype(int)
    D['label'] = label
    D['low_load'] = D.t_start.between(*P['low']) if label == 0 else False
    for c, (b, rmax) in P['vib'].items():
        D[c + '_excess'] = (D[c] - np.polyval(b, D.cur_rms)) > rmax
    D['vib_excess'] = D.v0_rms_excess | D.v1_rms_excess
    return D


def model_X(D, fill_shape):
    """지도학습·IsolationForest 입력: 크기 특징(log 변환) + 모양 거리(부분 사이클은 정상 중앙값으로 채움) + 완전 사이클 여부"""
    A = D[AMP_COLS]
    X = np.log1p(A.abs()) * np.sign(A)
    X['shape_dist'] = D.shape_dist.fillna(fill_shape)
    X['full_cycle'] = D.full_cycle.astype(int)
    return X


def grouped_cv_proba(make, Xn_train, Xn_test, Xa, groups):
    """이상 burst 단위 GroupKFold. 매 fold: 정상 학습 + 이상(학습 fold)으로 적합 →
    이상(검증 fold)은 out-of-fold 확률, 정상 테스트는 fold 평균 확률."""
    Xn_train, Xn_test, Xa = map(np.asarray, (Xn_train, Xn_test, Xa))
    pa, pn = np.zeros(len(Xa)), np.zeros(len(Xn_test))
    for itr, ite in GroupKFold(N_FOLDS).split(Xa, groups=groups):
        X = np.vstack([Xn_train, Xa[itr]]); y = np.r_[np.zeros(len(Xn_train)), np.ones(len(itr))]
        mdl = make().fit(X, y)
        pa[ite] = mdl.predict_proba(Xa[ite])[:, 1]
        pn += mdl.predict_proba(Xn_test)[:, 1] / N_FOLDS
    return pn, pa


def korean_font():
    """그래프용 한글 폰트: 운영체제별로 설치된 것을 찾고, 없으면 기본 폰트(한글은 깨지지만 실행은 됨)"""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ['Malgun Gothic', 'AppleGothic', 'Apple SD Gothic Neo', 'NanumGothic', 'Noto Sans CJK KR', 'Noto Sans KR']:
        if name in have: return name
    print('[알림] 한글 폰트를 찾지 못해 기본 폰트를 사용합니다. 그래프의 한글이 깨질 수 있습니다(결과 수치에는 영향 없음).')
    return 'DejaVu Sans'


def wilson(k, n, z=1.96):
    """이항 비율 95% Wilson 신뢰구간"""
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)
