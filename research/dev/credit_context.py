"""Credit: do context-specific distress signals add to the 16 shipped phrase flags + (negative equity, loss)? Runtime-feasible features only (no form-type control).
Metrics: AUC, sensitivity at the shipped rule's specificity, company-grouped bootstrap, shuffled-feature control, overlap with existing flags."""
import glob, json, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, "t4-local/agent")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
SP = r"[\s ​]"
NEW = {
 "n01 refinancing failure/risk": r"(?i)(unable|inability|not be able|failure|difficult\w*)" + SP + r"+to" + SP + r"+(refinance|extend|renew)|refinanc\w*.{0,40}(risk|uncertain|no assurance)",
 "n02 maturities within 12 months": r"(?i)(matur\w+|due).{0,60}(within|in the next|over the next|during the next).{0,15}(twelve|12)" + SP + r"+months",
 "n03 debt acceleration": r"(?i)accelerat\w+.{0,40}(debt|indebtedness|obligations|borrowings|notes|amounts" + SP + r"+outstanding)|lenders?" + SP + r"+(may|could|have" + SP + r"+the" + SP + r"+right" + SP + r"+to)" + SP + r"+(accelerate|demand)",
 "n04 default / acceleration notice": r"(?i)notice" + SP + r"+of" + SP + r"+(default|acceleration|event" + SP + r"+of" + SP + r"+default)|received" + SP + r"+a" + SP + r"+(default|reservation" + SP + r"+of" + SP + r"+rights)",
 "n05 covenant amendment/waiver (credit agreement)": r"(?i)(amend\w*|waiver).{0,60}(credit" + SP + r"+(agreement|facility)|covenant)",
 "n06 PROTECTIVE: liquidity sufficient for 12 months": r"(?i)(sufficient|adequate|enough).{0,80}(next|following|coming)" + SP + r"+(twelve|12)" + SP + r"+months",
 "n07 PROTECTIVE: no substantial doubt": r"(?i)(no|not|did" + SP + r"+not" + SP + r"+raise).{0,25}substantial" + SP + r"+doubt|substantial" + SP + r"+doubt.{0,120}(alleviat|no" + SP + r"+longer)",
 "n08 PROTECTIVE: in compliance with covenants": r"(?i)(in" + SP + r"+compliance|complied)" + SP + r"+with.{0,50}covenants",
 "n09 management plans to alleviate doubt": r"(?i)management.{0,10}s" + SP + r"+plans.{0,100}(alleviate|mitigate)",
 "n10 not sufficient cash/liquidity": r"(?i)(do|does|may|will)?" + SP + r"*not.{0,15}(have|be" + SP + r"+able" + SP + r"+to" + SP + r"+generate).{0,30}(sufficient|adequate).{0,30}(cash|liquidity|funds)",
 "n11 late / delayed filing": r"(?i)(unable|failed|delay\w*)" + SP + r"+to" + SP + r"+(timely" + SP + r"+)?file|notification" + SP + r"+of" + SP + r"+late" + SP + r"+filing|NT" + SP + r"+10-[KQ]",
 "n12 dividend suspended/eliminated": r"(?i)(suspend\w*|eliminat\w*|discontinu\w*)" + SP + r"+(our" + SP + r"+|the" + SP + r"+)?(quarterly" + SP + r"+)?dividend",
 "n13 reverse split / rights offering / ATM": r"(?i)reverse" + SP + r"+(stock" + SP + r"+)?split|rights" + SP + r"+offering|at-the-market",
 "n14 impairment of goodwill/long-lived assets": r"(?i)impairment" + SP + r"+(charge|of" + SP + r"+goodwill|of" + SP + r"+long-lived)",
}
idx = json.load(open("data_sec_credit/index.json")); ph = pd.read_csv("data_sec_credit/phrases.csv")
flag = [c for c in ph.columns if c[:2].isdigit()]; ph = ph.rename(columns={c: "p" + c.split(" ")[0] for c in flag}); flag = [c for c in ph.columns if c.startswith("p") and c[1:].isdigit()]
for c in flag: ph[c] = ph[c].astype(int)
rows = []
for a in ph.acc:
    t = open(f"data_sec_credit/txt/{a}.txt", encoding="utf-8").read()
    rows.append({k: int(bool(re.search(rx, t))) for k, rx in NEW.items()})
N = pd.DataFrame(rows); D = pd.concat([ph.reset_index(drop=True), N], axis=1); D["cik"] = D.acc.map(lambda a: idx[a]["cik"])
pit = pd.concat([pd.read_parquet(f, columns=["accession_number", "total_equity", "net_income"]) for f in sorted(glob.glob("data_ziplime/pit/**/*.parquet", recursive=True))]).drop_duplicates("accession_number").set_index("accession_number")
D["neg_eq"] = D.acc.map(lambda a: pit.total_equity.get(a, np.nan) < 0).astype(int); D["loss"] = D.acc.map(lambda a: pit.net_income.get(a, np.nan) < 0).astype(int)
y = D.label.values.astype(int); g = D.cik.values; NC = list(NEW)
print("flag prevalence in failing vs control filings, and the odds ratio (+0.5 smoothing):")
for k in NC:
    a, b = D[D.label == 1][k].mean(), D[D.label == 0][k].mean(); orr = ((D[(D.label == 1)][k].sum() + .5) / (D[(D.label == 1)][k].eq(0).sum() + .5)) / ((D[(D.label == 0)][k].sum() + .5) / (D[(D.label == 0)][k].eq(0).sum() + .5))
    ov = max(abs(np.corrcoef(D[k], D[c])[0, 1]) for c in flag if D[c].std() > 0)
    print(f"   {k:52s} failing {a:.2f} control {b:.2f} OR {orr:5.1f} | max |corr| with an existing flag {ov:.2f}")
BASE = flag + ["neg_eq", "loss"]
def cv(cols, seed=0):
    X = ((D[cols] - D[cols].mean()) / (D[cols].std() + 1e-9)).values; o = np.zeros(len(D))
    for a, b in GroupKFold(5).split(X, y, g): o[b] = LogisticRegression(C=0.3, max_iter=3000).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    return o
rule = ((D.p63 == 1) | (D[["p65", "p68", "p69", "p70", "p71", "p73", "p76", "p77"]].sum(axis=1) >= 3) | ((D.neg_eq == 1) & (D.loss == 1))).astype(int).values
spec = 1 - rule[y == 0].mean(); print(f"\nn={len(D)} positives={int(y.sum())} | shipped rule: sensitivity {rule[y == 1].mean():.3f}, specificity {spec:.3f}")
at = lambda p: np.mean(p[y == 1] > np.quantile(p[y == 0], spec))
p0 = cv(BASE); p1 = cv(BASE + NC)
print(f"baseline (16 flags + equity/loss) AUC {roc_auc_score(y, p0):.4f} sens@spec {at(p0):.3f} | + 14 context flags AUC {roc_auc_score(y, p1):.4f} sens@spec {at(p1):.3f}")
rng = np.random.default_rng(1); ug = np.unique(g); gi = {k: np.where(g == k)[0] for k in ug}; da, ds = [], []
for _ in range(1000):
    s = np.concatenate([gi[k] for k in rng.choice(ug, len(ug))])
    try:
        da.append(roc_auc_score(y[s], p1[s]) - roc_auc_score(y[s], p0[s]))
        ds.append(np.mean(p1[s][y[s] == 1] > np.quantile(p1[s][y[s] == 0], spec)) - np.mean(p0[s][y[s] == 1] > np.quantile(p0[s][y[s] == 0], spec)))
    except ValueError: pass
print(f"delta AUC {roc_auc_score(y, p1) - roc_auc_score(y, p0):+.4f} (90% CI {np.percentile(da, 5):+.4f}, {np.percentile(da, 95):+.4f}) | delta sensitivity at the rule's specificity {at(p1) - at(p0):+.3f} (90% CI {np.percentile(ds, 5):+.3f}, {np.percentile(ds, 95):+.3f})")
sh = []; rs = np.random.default_rng(9)
for _ in range(15):
    Dn = D.copy(); Dn[NC] = Dn[NC].values[rs.permutation(len(Dn))]
    X = ((Dn[BASE + NC] - Dn[BASE + NC].mean()) / (Dn[BASE + NC].std() + 1e-9)).values; o = np.zeros(len(D))
    for a, b in GroupKFold(5).split(X, y, g): o[b] = LogisticRegression(C=0.3, max_iter=3000).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    sh.append(roc_auc_score(y, o) - roc_auc_score(y, p0))
print(f"shuffled-feature control: mean delta AUC {np.mean(sh):+.4f} (sd {np.std(sh):.4f})")
# protective-only and risk-only subsets
for nm, cs in (("risk flags only (n01-n05, n10-n13)", [k for k in NC if "PROTECTIVE" not in k and not k.startswith(("n09", "n14"))]), ("protective flags only (n06-n08)", [k for k in NC if "PROTECTIVE" in k])):
    p2 = cv(BASE + cs); print(f"   + {nm:40s} AUC {roc_auc_score(y, p2):.4f} (delta {roc_auc_score(y, p2) - roc_auc_score(y, p0):+.4f}) sens@spec {at(p2):.3f}")
