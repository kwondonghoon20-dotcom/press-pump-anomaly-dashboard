"""
프레스 유압펌프 하이브리드 이상 탐지 모델
 - 모양 탐지기 : 전류 사이클 파형 모양 거리(shape_dist)  ← 종합보고서 방식
 - 크기 탐지기 : 사이클 단위 진폭·변동 특징 + One-Class SVM ← 우리 분석 방식
 - 경보 규칙   : ① 모양 경보 = shape_dist 임곗값 초과 시 1사이클 만에 즉시
                ② 크기 경보 = 크기 점수 임곗값 초과 + '정상보다 큰 신호가 하나 이상' 인 사이클이
                   2사이클 연속일 때 (저부하·조용한 구간은 경보하지 않음)
 사이클 = 전류 봉우리~다음 봉우리 (정상 약 1.67초, 0.6Hz)
 사용법: python pdm_hybrid.py  (data/press_data_normal.csv, data/outlier_data.csv)
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

S = ['AI0_Vibration', 'AI1_Vibration', 'AI2_Current']
N_PTS = 20          # 사이클 리샘플 점 수
Q = 99.5            # 임곗값 백분위수 (정상 검증 데이터 기준)
LOUD = ['v0_rms', 'v1_rms', 'cur_p2p', 'cur_diffstd']   # '정상보다 큰가' 판단 특징

# ---------------- 1. 데이터 로드 · burst 분할 ----------------
def load(path):
    d = pd.read_csv(path, index_col=0)
    d['TimeStamp'] = pd.to_datetime(d['TimeStamp'])
    d = d.drop_duplicates(subset='TimeStamp').reset_index(drop=True)
    gap = d['TimeStamp'].diff().dt.total_seconds().fillna(0.1)
    d['seg'] = (gap > 0.2).cumsum()          # 0.1초 연속 구간 = burst
    return d

# ---------------- 2. 사이클 추출 ----------------
def extract_cycles(d, min_rows=10):
    """burst 안에서 전류 봉우리 사이를 1사이클로 자른다.
    완전한 사이클이 없는 burst는 burst 전체를 '부분 사이클'로 남긴다(크기 탐지기 전용)."""
    out = []
    for s, g in d.groupby('seg'):
        if len(g) < min_rows: continue
        X = g[S].values; t0 = g['TimeStamp'].iloc[0]
        pk, _ = find_peaks(X[:, 2], distance=10)
        cyc = [(a, b) for a, b in zip(pk[:-1], pk[1:])]
        if cyc:
            for k, (a, b) in enumerate(cyc):
                out.append(dict(seg=s, k=k, full=True, t=g['TimeStamp'].iloc[b], t_seg=t0, x=X[a:b + 1]))
        else:
            out.append(dict(seg=s, k=0, full=False, t=g['TimeStamp'].iloc[-1], t_seg=t0, x=X))
    return out

def resample_z(c):
    r = np.interp(np.linspace(0, 1, N_PTS), np.linspace(0, 1, len(c)), c)
    return (r - r.mean()) / (r.std() + 1e-9)

def amp_features(x):
    v0, v1, c = x[:, 0], x[:, 1], x[:, 2]
    rms = lambda z: np.sqrt(np.mean(z ** 2))
    f = dict(v0_rms=rms(v0), v1_rms=rms(v1), v0_p2p=np.ptp(v0), v1_p2p=np.ptp(v1),
             cur_rms=rms(c - c.mean()), cur_p2p=np.ptp(c), cur_diffstd=np.diff(c).std(),
             corr01=np.corrcoef(v0, v1)[0, 1])
    return {k: (0.0 if not np.isfinite(v) else v) for k, v in f.items()}

# ---------------- 3. 모델 ----------------
class HybridDetector:
    def fit(self, train_cycles, val_cycles, q=Q):
        full = [c['x'] for c in train_cycles if c['full']]
        self.template = np.mean([resample_z(x[:, 2]) for x in full], axis=0)      # 정상 평균 파형
        A = pd.DataFrame([amp_features(c['x']) for c in train_cycles])
        self.scaler = StandardScaler().fit(np.log1p(A.abs()) * np.sign(A))
        self.ocsvm = OneClassSVM(nu=0.01, gamma='scale').fit(self._amp_X(train_cycles))
        sv, av = self.scores(val_cycles)
        self.th_shape = np.nanpercentile(sv, q)
        self.th_amp = np.percentile(av, q)
        self.med = pd.DataFrame([amp_features(c['x']) for c in train_cycles])[LOUD].median()
        return self

    def _amp_X(self, cycles):
        A = pd.DataFrame([amp_features(c['x']) for c in cycles])
        return self.scaler.transform(np.log1p(A.abs()) * np.sign(A))

    def shape_dist(self, x):
        return np.sqrt(np.mean((resample_z(x[:, 2]) - self.template) ** 2))

    def scores(self, cycles):
        sh = np.array([self.shape_dist(c['x']) if c['full'] else np.nan for c in cycles])
        am = -self.ocsvm.score_samples(self._amp_X(cycles))
        return sh, am

    def predict(self, cycles):
        """cycles는 시간순 스트림으로 간주. 반환: 모양경보, 크기경보(규칙 적용), 최종경보"""
        order = np.argsort([c['t'] for c in cycles])
        sh, am = self.scores(cycles)
        f_shape = np.nan_to_num(sh, nan=-1) > self.th_shape
        A = pd.DataFrame([amp_features(c['x']) for c in cycles])
        louder = (A[LOUD] > self.med).any(axis=1).values          # 정상 중앙값보다 큰 신호가 하나라도
        raw = (am > self.th_amp) & louder
        f_amp = np.zeros(len(cycles), bool)
        r = raw[order]; f_amp[order] = r & np.r_[False, r[:-1]]     # 2사이클 연속
        return f_shape, f_amp, f_shape | f_amp

    def predict_single(self, cycles):
        """규칙 없이 사이클 하나씩 본 원점수 판정 (비교용)"""
        sh, am = self.scores(cycles)
        return np.nan_to_num(sh, nan=-1) > self.th_shape, am > self.th_amp

# ---------------- 4. 분할 (시간 블록, 누수 없음) ----------------
def blocked_split(cycles, n_blocks=10):
    segs = np.array(sorted({c['seg'] for c in cycles})); tr, va, te = set(), set(), set()
    for bl in np.array_split(segs, n_blocks):
        a, b = int(.6 * len(bl)), int(.8 * len(bl))
        tr |= set(bl[:a]); va |= set(bl[a:b]); te |= set(bl[b:])
    pick = lambda S_: [c for c in cycles if c['seg'] in S_]
    return pick(tr), pick(va), pick(te)

# ---------------- 5. 평가 ----------------
def evaluate(model, normal_test, anomaly):
    fs, fa, f = model.predict(normal_test)
    gs, ga, g = model.predict(anomaly)
    seg_det = pd.DataFrame(dict(seg=[c['seg'] for c in anomaly], k=[c['k'] for c in anomaly], f=g))
    first = seg_det[seg_det.k == 0].f
    per_seg = seg_det.groupby('seg').f.max()
    nseg = pd.DataFrame(dict(seg=[c['seg'] for c in normal_test], f=f)).groupby('seg').f.max()
    ss, sa = model.predict_single(anomaly)
    return {
        '정상 사이클 오경보율': f.mean(), '  └ 모양 경보': fs.mean(), '  └ 크기 경보': fa.mean(),
        '정상 burst 중 경보 발생 비율': nseg.mean(),
        '이상 사이클 탐지율': g.mean(), '  └ 모양 경보로 잡힘': gs.mean(), '  └ 크기 경보로 잡힘': ga.mean(),
        '  (원점수) 모양 단독 탐지율': ss.mean(), '  (원점수) 크기 단독 탐지율': sa.mean(),
        '이상 burst 탐지': f"{int(per_seg.sum())}/{len(per_seg)}",
        '이상 burst 첫 사이클 탐지': f"{int(first.sum())}/{len(first)}",
    }

# ---------------- 6. 결함 주입 시험 ----------------
def inject(cycles, kind, a, rng):
    out = []
    for c in cycles:
        x = c['x'].copy()
        if kind == '진동 진폭 증가':
            x[:, 0] *= 1 + 3 * a; x[:, 1] *= 1 + a
        elif kind == '전류 진폭 증가':
            m = x[:, 2].mean(); x[:, 2] = m + (x[:, 2] - m) * (1 + a)
        elif kind == '전류 파형 왜곡':    # 크기(평균·RMS)는 그대로, 모양만 망가뜨림
            cur = x[:, 2]; m, s = cur.mean(), cur.std()
            z = (cur - m) / s + a * 1.5 * rng.standard_normal(len(cur))
            x[:, 2] = m + (z - z.mean()) / z.std() * s
        out.append({**c, 'x': x})
    return out

if __name__ == '__main__':
    from pathlib import Path
    DATA = Path(__file__).resolve().parent / 'data'
    n = load(DATA / 'press_data_normal.csv'); o = load(DATA / 'outlier_data.csv')
    cn, co = extract_cycles(n), extract_cycles(o)
    tr, va, te = blocked_split(cn)
    m = HybridDetector().fit(tr, va)
    print(f"사이클 수: 정상 {len(cn)} (완전 {sum(c['full'] for c in cn)}), 이상 {len(co)} (완전 {sum(c['full'] for c in co)})")
    print(f"학습/검증/테스트 정상 사이클: {len(tr)}/{len(va)}/{len(te)}")
    print(f"임곗값(p{Q}): shape_dist {m.th_shape:.3f}, 크기 점수 {m.th_amp:.3f}\n")
    for k, v in evaluate(m, te, co).items():
        print(f"{k:28s} {v:.3f}" if isinstance(v, float) else f"{k:28s} {v}")
    sh_n, _ = m.scores([c for c in cn if c['full']]); sh_o, _ = m.scores([c for c in co if c['full']])
    print(f"\nshape_dist 분포: 정상 {np.nanmin(sh_n):.3f}~{np.nanmax(sh_n):.3f} / 이상 {np.nanmin(sh_o):.3f}~{np.nanmax(sh_o):.3f}")
    rng = np.random.default_rng(0); rows = []
    for kind in ['진동 진폭 증가', '전류 진폭 증가', '전류 파형 왜곡']:
        for a in [0, .1, .2, .3, .5, 1.0]:
            fs, fa, f = m.predict(inject(te, kind, a, rng))
            rows.append(dict(주입=kind, α=a, 모양=fs.mean(), 크기=fa.mean(), 하이브리드=f.mean()))
    print('\n[결함 주입 시험 - 정상 테스트 사이클 탐지율]')
    print(pd.DataFrame(rows).round(3).to_string(index=False))
