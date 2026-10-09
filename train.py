import os, glob, json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

def detect_season():
    now = datetime.now()
    start = now.year if now.month >= 8 else now.year - 1
    return f"{start % 100:02d}{(start+1) % 100:02d}"

def get_seasons(n=3):
    now = datetime.now()
    s0 = now.year if now.month >= 8 else now.year - 1
    return [f"{(s0-i) % 100:02d}{(s0-i+1) % 100:02d}" for i in range(n)]

def cw(mp):
    return 1.0 if mp >= 30 else 0.15 + (mp / 30) * 0.85

def bavg(cur, prev, key):
    def a(lst):
        v = [d.get(key) for d in lst if d.get(key) is not None]
        return float(np.mean(v)) if v else None
    c, p = a(cur), a(prev)
    if c is None and p is None: return 0.0
    if c is None: return p
    if p is None: return c
    w = cw(len(cur))
    return w * c + (1 - w) * p

CUR = detect_season()
SEASONS = get_seasons(3)
print("Current:", CUR)
print("Seasons:", SEASONS)

D = "/content/data"
os.makedirs(D, exist_ok=True)
BASE = "https://www.football-data.co.uk/mmz4281"

LOWER = {"E1":"Champ","E2":"L1","E3":"L2","D2":"B2","SP2":"Seg","I2":"SerB","F2":"L2F","N1":"Ere"}
TOP = {"E0":"PL","SP1":"LL","I1":"SA","D1":"BL","F1":"L1F"}
ALL = {**LOWER, **TOP}
LOWKEYS = set(LOWER.keys())

for season in SEASONS:
    for code in ALL:
        try:
            df = pd.read_csv(f"{BASE}/{season}/{code}.csv")
            df.to_csv(f"{D}/{season}_{code}.csv", index=False)
        except Exception:
            pass

fl = glob.glob(f"{D}/*.csv")
parts = []
for f in fl:
    t = pd.read_csv(f)
    fn = os.path.basename(f).replace(".csv", "")
    season, league = fn.split("_")
    t["Season"] = season
    t["League"] = league
    t["is_lower"] = 1 if league in LOWKEYS else 0
    parts.append(t)
df = pd.concat(parts, ignore_index=True)

df = df.dropna(subset=['HC','AC','HY','AY','HR','AR','HS','AS','HST','AST','HF','AF'])
df['TotalCards'] = df['HY'] + df['AY']
df['Target_Cards_Over4'] = (df['TotalCards'] > 4.5).astype(int)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)

df['Home_Rest_Days'] = 7.0
df['Away_Rest_Days'] = 7.0
lm = {}
for idx, row in df.iterrows():
    date = row['Date']
    for tc, rc in [('HomeTeam','Home_Rest_Days'), ('AwayTeam','Away_Rest_Days')]:
        team = row[tc]
        if team in lm:
            g = (date - lm[team]).days
            if 0 < g <= 30:
                df.at[idx, rc] = float(g)
        lm[team] = date

STATS = ['HC','AC','HY','AY','HS','AS','HST','AST','HF','AF']
hist = {}
for _, row in df.iterrows():
    s = row['Season']
    for side, tc in [('home','HomeTeam'), ('away','AwayTeam')]:
        k = (side, s, row[tc])
        hist.setdefault(k, []).append({st: row[st] for st in STATS})

def get_hist(side, season, team):
    cur = hist.get((side, season, team), [])
    y = int(season[:2])
    prev_s = f"{(y-1) % 100:02d}{y % 100:02d}"
    prev = hist.get((side, prev_s, team), [])
    return cur[-3:], prev[-3:]

rows_out = []
for _, row in df.iterrows():
    hc, hp = get_hist('home', row['Season'], row['HomeTeam'])
    ac, ap = get_hist('away', row['Season'], row['AwayTeam'])
    if not (hc or hp) or not (ac or ap):
        continue
    feat = {}
    for s in STATS:
        feat[f"Home_{s}_Avg"] = bavg(hc, hp, s)
        feat[f"Away_{s}_Avg"] = bavg(ac, ap, s)
    feat['Home_Rest_Days'] = row['Home_Rest_Days']
    feat['Away_Rest_Days'] = row['Away_Rest_Days']
    feat['Target_Cards_Over4'] = row['Target_Cards_Over4']
    feat['Season'] = row['Season']
    feat['is_lower'] = row['is_lower']
    rows_out.append(feat)

tdf = pd.DataFrame(rows_out)
print("Features built:", len(tdf))

FEATS = [c for c in tdf.columns if '_Avg' in c] + ['Home_Rest_Days','Away_Rest_Days']
TARGET = 'Target_Cards_Over4'

tr = tdf[tdf['Season'] != CUR]
te = tdf[(tdf['Season'] == CUR) & (tdf['is_lower'] == 1)]
Xtr = tr[FEATS].values
ytr = tr[TARGET].values
Xte = te[FEATS].values
yte = te[TARGET].values
print("Train:", Xtr.shape, "Test:", Xte.shape)

base = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.04,
                     subsample=0.8, colsample_bytree=0.8,
                     eval_metric='logloss', random_state=42)
model = CalibratedClassifierCV(base, method='isotonic', cv=5)
model.fit(Xtr, ytr)

if len(Xte) > 0:
    yp = model.predict_proba(Xte)[:, 1]
    b = brier_score_loss(yte, yp)
    bb = np.mean((yte - yte.mean()) ** 2)
    a = roc_auc_score(yte, yp)
    print("Brier:", round(b, 4), "Baseline:", round(bb, 4), "Improvement:", round((bb-b)/bb*100, 2), "%")
    print("AUC:", round(a, 4))
    print("Pred min:", round(yp.min(), 3), "max:", round(yp.max(), 3), "mean:", round(yp.mean(), 3))

raw = model.calibrated_classifiers_[0].estimator
raw.save_model("/content/xgb_model.json")

rpt = raw.predict_proba(Xtr)[:, 1]
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(rpt, ytr)
xg = np.linspace(0, 1, 100)
yg = iso.predict(xg)

cal = {"raw_probabilities": xg.tolist(), "calibrated_probabilities": yg.tolist()}
with open("/content/calibration.json", "w") as f:
    json.dump(cal, f, indent=2)

fo = {"feature_order": FEATS, "num_features": len(FEATS),
      "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
      "method": "blended_v3"}
with open("/content/feature_order.json", "w") as f:
    json.dump(fo, f, indent=2)

print("Exports ready, features:", len(FEATS))
