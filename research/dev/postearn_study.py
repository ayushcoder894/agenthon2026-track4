"""Event study: one-day ABNORMAL return of large-cap stocks after earnings releases (2010 - 2022-06-28).

Data: ZipLime/earnings-calendar (Apache-2.0; release timestamps, session, reported figures) and
AYUSHKHAIRE/all-stock-market-data-daily-updates (CC0-1.0; daily split-adjusted prices). Benchmark = equal-weighted
mean return of the other large caps on the same day (SPY is not in the price dataset).
Reaction window as in the task: close of the last session BEFORE the first tradeable session -> close of that session.
"""
import io
import json
import os
import concurrent.futures as cf

import numpy as np
import pandas as pd
import requests

LIMIT = pd.Timestamp("2022-06-28")
ev = pd.read_parquet("data_earn/earnings.parquet")
ev = ev[(ev.form == "8-K") & ev.ticker.notna() & ev.first_tradeable_session.notna()]
ev["session_day"] = pd.to_datetime(ev.first_tradeable_session)
ev = ev[(ev.session_day >= "2010-03-01") & (ev.session_day <= LIMIT)]
big = ev[ev.revenue >= 10e9]
tickers = sorted(big.ticker.dropna().unique())
print("large-cap events:", len(big), "tickers:", len(tickers))

base = "https://huggingface.co/datasets/AYUSHKHAIRE/all-stock-market-data-daily-updates/resolve/main/"
os.makedirs("data_px", exist_ok=True)


def fetch(t):
    path = f"data_px/{t}.csv"
    if not os.path.exists(path):
        r = requests.get(base + f"{t[0]}/{t}.csv", timeout=120)
        if not r.ok:
            return t, None
        open(path, "wb").write(r.content)
    d = pd.read_csv(path)
    d["day"] = pd.to_datetime(d.timestamp, unit="s").dt.normalize()
    d = d[d.day <= LIMIT].drop_duplicates("day").set_index("day").sort_index()
    return t, d.close


with cf.ThreadPoolExecutor(8) as ex:
    px = {t: s for t, s in ex.map(fetch, tickers) if s is not None and len(s) > 250}
print("price series:", len(px))
P = pd.DataFrame(px).sort_index()
R = P.pct_change(fill_method=None)
R = R.where(R.abs() < 0.6)  # drop split/glitch days the adjustment missed
mkt_sum, mkt_n = R.sum(axis=1), R.notna().sum(axis=1)

rows = []
for _, e in big[big.ticker.isin(px)].iterrows():
    d = e.session_day.normalize()
    if d not in R.index:
        continue
    r = R.at[d, e.ticker]
    if not np.isfinite(r) or mkt_n.at[d] < 20:
        continue
    m = (mkt_sum.at[d] - r) / (mkt_n.at[d] - 1)
    rows.append({"ticker": e.ticker, "day": d, "abn": 100 * (r - m), "rev": e.revenue, "session": e.session,
                 "eps_yoy": e.eps_yoy_change, "year": d.year})
S = pd.DataFrame(rows)
print("events with prices:", len(S), "| tickers", S.ticker.nunique(), "| years", S.year.min(), "-", S.year.max())


def summary(df, name):
    a = df.abn
    print(f"{name:42s} n={len(a):5d} | P(abn>+1%)={np.mean(a > 1):.3f} P(abn<-1%)={np.mean(a < -1):.3f} P(flat)={np.mean(a.abs() <= 1):.3f}"
          f" | q05={a.quantile(.05):.2f} q50={a.quantile(.5):.2f} q95={a.quantile(.95):.2f} | median|abn|={a.abs().median():.2f}")


summary(S, "all large caps (quarterly revenue >= $10B)")
summary(S[S.rev >= 2e10], "quarterly revenue >= $20B")
summary(S[S.session == "after_close"], "after-close releases")
summary(S[(S.rev >= 2e10) & (S.session == "after_close")], ">= $20B and after close")
for y, d in S.groupby("year"):
    summary(d, f"  year {y}")
for flag in (True, False):
    summary(S[(S.eps_yoy > 0) == flag], f"last-known eps_yoy>0 = {flag} (NOT known pre-release; diagnostic)")
S.to_parquet("t4-local/dev/postearn_events.parquet")
json.dump({"n": int(len(S)), "p_pos": float(np.mean(S.abn > 1)), "p_neg": float(np.mean(S.abn < -1)), "p_flat": float(np.mean(S.abn.abs() <= 1)),
           "q05": float(S.abn.quantile(.05)), "q50": float(S.abn.quantile(.5)), "q95": float(S.abn.quantile(.95))},
          open("t4-local/dev/postearn_fit.json", "w"), indent=1)
