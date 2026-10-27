"""Run the EPS parser on real 10-Q filings and compare with the official XBRL diluted EPS (pit data).
Usage: python -I sec_eval.py   (reads data_sec/raw, untrusted downloads: only parsed as text)"""
import glob, html, json, re, sys
from html.parser import HTMLParser
sys.path.insert(0, "t4-local/agent")
import pandas as pd
from t4_agent import predict as P

class T(HTMLParser):
    def __init__(self): super().__init__(convert_charrefs=True); self.o = []; self.skip = 0
    def handle_starttag(self, t, a):
        if t in ("script", "style") : self.skip += 1
        if t in ("ix:header",): self.skip += 1
    def handle_endtag(self, t):
        if t in ("script", "style", "ix:header") and self.skip: self.skip -= 1
    def handle_data(self, d):
        if not self.skip: self.o.append(d)
def to_text(raw):
    p = T(); p.feed(raw.decode("utf-8", "replace")); s = " ".join(x.strip(" \t\r\n") for x in p.o if x.strip(" \t\r\n"))
    return re.sub(r"[ \t\r\n]{2,}", " ", s)

idx = json.load(open("data_sec/index.json"))
import os
os.makedirs("data_sec/txt", exist_ok=True)
def text_of(a):
    f = f"data_sec/txt/{a}.txt"
    if os.path.exists(f): return open(f, encoding="utf-8").read()
    t = to_text(open(f"data_sec/raw/{a}.htm", "rb").read()); open(f, "w", encoding="utf-8").write(t); return t
pit = pd.concat([pd.read_parquet(f, columns=["cik", "accession_number", "period_end", "quarters", "form", "knowledge_date", "eps_diluted", "revision"]) for f in sorted(glob.glob("data_ziplime/pit/**/*.parquet", recursive=True))])
pit = pit[pit.quarters.isin([1]) & pit.form.astype(str).str.startswith("10-Q") & pit.eps_diluted.notna()]
pit["period_end"] = pd.to_datetime(pit.period_end)
pit = pit.sort_values(["cik", "period_end", "knowledge_date", "revision"]).groupby(["cik", "period_end"], as_index=False).first()
bycik = {}
for r in pit.itertuples(): bycik.setdefault(r.cik, []).append((r.period_end, r.eps_diluted))
def official(cik, pe):
    L = bycik.get(cik, [])
    cur = next((v for p, v in L if abs((p - pe).days) <= 3), None)
    pri = next((v for p, v in L if abs((p - (pe - pd.DateOffset(years=1))).days) <= 20), None)
    return cur, pri
CONFIGS = [("final majority + block", "none", True, True, False, "TIER")]
texts = {a: text_of(a) for a in idx if os.path.exists(f"data_sec/raw/{a}.htm")}
gt = {a: official(idx[a]["cik"], pd.Timestamp(idx[a]["period"])) for a in texts}
valid = [a for a in texts if gt[a][0] is not None and gt[a][1] is not None]
print("filings", len(texts), "with official pair", len(valid))
from t4_agent import eps_parse as E
for name, ytd, exc, sc, sd, which in CONFIGS:
    E.YTD_GUARD, E.EXCLUDE_CTX, E.SKIP_CONT, E.SKIP_DISC = ytd, exc, sc, sd
    E.MAJORITY = ('majority' in name and 'off' not in name.split('majority')[1][:4]) or False
    E.ATTR_LABELS = 'block' in name and 'block off' not in name

    n_ok = n_cur = n_sw = n_par = 0; wrong = []
    for a in valid:
        got = P._parse_eps_pair(texts[a])
        if got is None: continue
        n_par += 1; cur, pri = got[0], got[1]; co, po = gt[a]
        if abs(cur - co) < 0.006 and abs(pri - po) < 0.006: n_ok += 1
        elif abs(cur - co) < 0.006: n_cur += 1; wrong.append((a, "prior-only", cur, pri, co, po))
        elif abs(cur - po) < 0.006 and abs(pri - co) < 0.006: n_sw += 1; wrong.append((a, "swapped", cur, pri, co, po))
        else: wrong.append((a, "wrong", cur, pri, co, po))
    print(f"{name:34s} coverage {n_par / len(valid):.3f} ({n_par}) | pair correct {n_ok / n_par:.3f} | cur correct {(n_ok + n_cur) / n_par:.3f} | swapped {n_sw} prior-only-wrong {n_cur} wrong {len(wrong) - n_sw - n_cur}")
    if name.startswith("tiered broad"): json.dump(wrong, open("data_sec/wrong_best.json", "w"))
