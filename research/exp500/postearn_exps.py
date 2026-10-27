"""Post-earnings reaction (label: positive >+1% / negative <-1% / flat abnormal return) ~45 experiments on 3,795 large-cap events (2010-2022-06).
Features use only information BEFORE each announcement: the ticker's previous events (lagged abn, lagged reported eps growth, lagged revenue growth). The event's own eps_yoy (the reported number) is the outcome side and is never used."""
import sys, os, itertools
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression, Ridge
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Family
ROOT = os.environ.get("EXP500_DATA", "D:/AGENTHON")
E = pd.read_parquet(os.environ.get("POSTEARN", f"{ROOT}/t4-local/dev/postearn_events.parquet")).sort_values(["ticker", "day"]).reset_index(drop=True)
E["lab"] = np.where(E.abn > 1, 2, np.where(E.abn < -1, 0, 1))                  # 2 positive, 1 flat, 0 negative
g = E.groupby("ticker")
E["abn_l1"] = g.abn.shift(1); E["abn_l2"] = g.abn.shift(2); E["abn_m4"] = g.abn.transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean()); E["abn_m8"] = g.abn.transform(lambda s: s.shift(1).rolling(8, min_periods=3).mean())
E["abs_m8"] = g.abn.transform(lambda s: s.abs().shift(1).rolling(8, min_periods=3).mean())
E["eps_l1"] = g.eps_yoy.shift(1).clip(-3, 3); E["eps_l2"] = g.eps_yoy.shift(2).clip(-3, 3)
E["rev_g"] = (E.rev / g.rev.shift(4) - 1).clip(-1, 2)                              # revenue vs the event four quarters earlier; both reported BEFORE this announcement? -> use the previous event's growth
E["rev_g_l1"] = g.rev_g.shift(1) if False else E.groupby("ticker").rev_g.shift(1)
E["rev_ratio_l1"] = (g.rev.shift(1) / g.rev.shift(5) - 1).clip(-1, 2)
E["pre"] = (E.session == "pre_market").astype(int); E["lab_l1"] = g.lab.shift(1)
E["lab_mode8"] = g.lab.transform(lambda s: s.shift(1).rolling(8, min_periods=3).apply(lambda x: pd.Series(x).mode().iloc[0], raw=True))
E["split"] = np.where(E.day <= "2015-12-31", "train", np.where(E.day <= "2018-12-31", "val", "final")); E["block"] = E.day.dt.year * 10 + E.day.dt.quarter
E["logrev"] = np.log(E.rev)
FE = {"abn": ["abn_l1", "abn_l2", "abn_m4", "abn_m8", "abs_m8"], "eps": ["eps_l1", "eps_l2"], "rev": ["rev_ratio_l1", "logrev"], "all": ["abn_l1", "abn_l2", "abn_m4", "abn_m8", "abs_m8", "eps_l1", "eps_l2", "rev_ratio_l1", "logrev", "pre"]}
print("events", len(E), E.split.value_counts().to_dict(), "label share", E.lab.value_counts(normalize=True).round(3).to_dict())
def rows(df, w): return df[df.split == w]
def sp(df, w):
    tr = df[df.split == "train"] if w == "val" else df[df.split.isin(["train", "val"])]; return tr, df[df.split == w]
fam = Family("postearn_label"); scorers = {}
base = lambda df, w: (rows(df, w).lab.values == 2).astype(float)                    # shipped: always positive
def add(eid, desc, params, pred):
    sc = lambda df, w: (pred(df, w) == rows(df, w).lab.values).astype(float); scorers[eid] = sc; fam.run(eid, desc, params, sc, base, E)
for lv, nm in ((0, "negative"), (1, "flat"), (2, "positive")): add(f"pe_always_{nm}", f"always {nm}", dict(), lambda df, w, lv=lv: np.full(len(rows(df, w)), lv))
add("pe_train_majority", "train majority label", dict(), lambda df, w: np.full(len(rows(df, w)), int(sp(df, w)[0].lab.mode().iloc[0])))
add("pe_last_label", "same label as the ticker's previous reaction", dict(), lambda df, w: rows(df, w).lab_l1.fillna(2).astype(int).values)
add("pe_opposite_last", "opposite of previous reaction (pos<->neg)", dict(), lambda df, w: (2 - rows(df, w).lab_l1.fillna(0)).astype(int).values)
add("pe_mode8", "ticker's modal label over its last 8 reactions", dict(), lambda df, w: rows(df, w).lab_mode8.fillna(2).astype(int).values)
for th in (-1.0, 0.0, 1.0):
    add(f"pe_absm8_thr{th}", f"positive if mean abn last 8 > {th} else negative", dict(th=th), lambda df, w, th=th: np.where(rows(df, w).abn_m8.fillna(0) > th, 2, 0))
for col in ("eps_l1", "rev_ratio_l1"):
    for th in (0.0, 0.1):
        add(f"pe_{col}_thr{th}", f"positive if {col} > {th} else negative", dict(col=col, th=th), lambda df, w, col=col, th=th: np.where(rows(df, w)[col].fillna(0) > th, 2, 0))
        add(f"pe_{col}_thr{th}_flat", f"positive if {col} > {th}, negative if < -{th}, else flat", dict(col=col, th=th), lambda df, w, col=col, th=th: np.where(rows(df, w)[col].fillna(0) > th, 2, np.where(rows(df, w)[col].fillna(0) < -th, 0, 1)))
for C, fs in itertools.product((0.01, 0.1, 1.0), FE):
    def lg(df, w, C=C, fs=fs):
        tr, te = sp(df, w); cols = FE[fs]; Xtr, Xte = tr[cols].fillna(0).values, te[cols].fillna(0).values; mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        return LogisticRegression(C=C, max_iter=400).fit((Xtr - mu) / sd, tr.lab.values).predict((Xte - mu) / sd)
    add(f"pe_logit_{fs}_C{C}", f"multinomial logit {fs} C={C}", dict(C=C, fs=fs), lg)
for lv, ne, fs in itertools.product((4, 8), (100, 300), ("abn", "all")):
    def gb(df, w, lv=lv, ne=ne, fs=fs):
        tr, te = sp(df, w); cols = FE[fs]
        m = lgb.LGBMClassifier(n_estimators=ne, learning_rate=0.03, num_leaves=lv, min_child_samples=60, subsample=0.7, subsample_freq=1, colsample_bytree=0.8, reg_lambda=20.0, verbose=-1, random_state=0)
        return m.fit(tr[cols].values, tr.lab.values).predict(te[cols].values)
    add(f"pe_gbm_{fs}_l{lv}_n{ne}", f"lgbm {fs} leaves={lv} n={ne}", dict(leaves=lv, n_est=ne, fs=fs), gb)
for a, fs, thr in itertools.product((100.0, 3000.0), ("abn", "all"), (0.75, 1.0)):
    def rg(df, w, a=a, fs=fs, thr=thr):
        tr, te = sp(df, w); cols = FE[fs]; Xtr, Xte = tr[cols].fillna(0).values, te[cols].fillna(0).values; mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        p = Ridge(alpha=a).fit((Xtr - mu) / sd, tr.abn.clip(-15, 15).values).predict((Xte - mu) / sd); return np.where(p > thr, 2, np.where(p < -thr, 0, 1))
    add(f"pe_ridge_{fs}_a{a}_t{thr}", f"ridge on abn, thresholds +-{thr} ({fs})", dict(alpha=a, fs=fs, thr=thr), rg)
fam.finalize(E, scorers, base, top_k=3)
print("done")
