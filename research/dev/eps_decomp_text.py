"""Step 3: do large one-off items named in the 10-Q text (gains/charges, in $ million/billion) explain the shipped rule's error on the NEXT quarter?
Joins data_sec 10-Q texts (cik, period_end) to the decomposition table D (origin = that quarter as the 'last' quarter)."""
import json, re, sys
import numpy as np, pandas as pd
sys.argv = ["x"]
exec(open("t4-local/dev/eps_decomp.py").read().split("SETS = {")[0])
idx = json.load(open("data_sec/index.json"))
rows = [dict(acc=a, cik=v["cik"], period_end=pd.Timestamp(v["period"])) for a, v in idx.items()]
S = pd.DataFrame(rows)
S["period_end"] = S.period_end.astype("datetime64[ns]")
J = S.merge(D[["cik", "period_end", "resid", "y", "base", "fin", "rev_last", "abs_prior", "pe_next", "eps_last"]], on=["cik", "period_end"], how="inner")
J = J.merge(E[["cik", "period_end", "net_income"]], on=["cik", "period_end"], how="left")
print("filings", len(S), "joined to an outcome", len(J), "companies", J.cik.nunique(), "fin", int(J.fin.sum()))
POS = r"(?:net\s+)?(?:gains?|benefits?|releases?)"
NEG = r"(?:charges?|impairments?|write-?downs?|write-?offs?|settlements?|restructuring|special assessment|litigation|losses)"
AMT = r"\$\s?([\d][\d,]*\.?\d*)\s*(million|billion)"
def scan(txt):
    gain = loss = 0.0; n = 0
    for m in re.finditer(AMT, txt):
        a = float(m.group(1).replace(",", "")) * (1e9 if m.group(2) == "billion" else 1e6)
        w = txt[max(0, m.start() - 120): m.end() + 120].lower()
        if re.search(r"visa|one-?time|non-?recurring|pre-?tax|after-?tax|net of tax|discrete|special|unusual|extraordinary", w) or True:
            if re.search(POS, w) and not re.search(NEG, w): gain += a; n += 1
            elif re.search(NEG, w) and not re.search(POS, w) and re.search(r"(?:one-?time|non-?recurring|special|unusual|restructuring|impairment|settlement|write-?down|write-?off)", w): loss += a; n += 1
    return gain, loss, n
out = []
for r in J.itertuples():
    try: txt = open(f"data_sec/txt/{r.acc}.txt", encoding="utf-8", errors="ignore").read()
    except OSError: out.append((np.nan,)*3); continue
    out.append(scan(txt))
J[["gain", "loss", "nitems"]] = pd.DataFrame(out, index=J.index)
ni = J.net_income.abs().clip(lower=1e6)
J["gr"] = (J.gain / ni).clip(0, 20); J["lr"] = (J.loss / ni).clip(0, 20)
J["flag_g"] = (J.gr >= 0.25).astype(int); J["flag_l"] = (J.lr >= 0.25).astype(int)
J["yy"] = J.y.clip(-1000, 1000)
for nm, f in (("ALL", J), ("LARGE", J[(J.rev_last >= 1e9) & (J.abs_prior >= 0.5)]), ("FIN", J[J.fin])):
    print(f"\n{nm}: n={len(f)} flag_g={f.flag_g.mean():.2f} flag_l={f.flag_l.mean():.2f}")
    for col in ("flag_g", "flag_l"):
        a, b = f[f[col] == 1], f[f[col] == 0]
        if len(a) >= 5: print(f"  {col}: n_flag={len(a)} mean resid flagged {a.resid.clip(-300,300).mean():+.1f} (median {a.resid.median():+.1f}) vs unflagged {b.resid.clip(-300,300).mean():+.1f} (median {b.resid.median():+.1f})")
    ok = f.dropna(subset=["gr"]); 
    if len(ok) > 30: print("  spearman resid~gr", round(ok.resid.rank().corr(ok.gr.rank()), 3), " resid~lr", round(ok.resid.rank().corr(ok.lr.rank()), 3))
J.drop(columns=["acc"]).to_csv("t4-local/exp_out/eps_decomp_text.csv", index=False)
