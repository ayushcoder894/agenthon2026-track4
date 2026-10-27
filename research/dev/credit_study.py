"""Which balance-sheet signals visible in a company's own filing predict a bankruptcy filing (8-K item 1.03) within 12 months?
Data: ZipLime/company-fundamentals pit (SEC XBRL as reported; Apache-2.0) and ZipLime/sec-8k-events items (Apache-2.0).
Origins (statement knowledge dates) up to 2021-06-28 so that every 12-month label window closes by 2022-06-28."""
import glob

import numpy as np
import pandas as pd

ORIGIN_END = pd.Timestamp("2021-06-28", tz="UTC")
it = pd.concat([pd.read_parquet(f, columns=["cik", "item_code", "form", "is_amendment", "accepted_at"]) for f in glob.glob("data_8k/*.parquet")])
bk = it[(it.item_code == "1.03") & ~it.is_amendment].groupby("cik").accepted_at.apply(sorted).to_dict()
print("companies with an item-1.03 filing:", len(bk))

COLS = ["cik", "period_end", "period_kind", "form", "knowledge_date", "revision", "total_equity", "net_income", "total_assets",
        "total_liabilities", "cash_and_equivalents", "current_liabilities", "revenue", "long_term_debt"]
f = pd.concat([pd.read_parquet(p, columns=COLS) for p in glob.glob("data_ziplime/pit/**/*.parquet", recursive=True)])
f = f[f.period_kind.isin(["quarter", "year"]) & f.form.astype(str).str.match(r"10-[QK]") & f.total_equity.notna() & f.total_assets.notna()]
f = f.sort_values(["cik", "period_end", "knowledge_date", "revision"]).groupby(["cik", "period_end", "period_kind"], as_index=False).first()
f = f[(f.knowledge_date <= ORIGIN_END) & (f.total_assets >= 5e8)]
f["cik"] = f.cik.astype(str).str.zfill(10)


def label(row):
    dates = bk.get(row.cik)
    if not dates:
        return 0, False
    prior = any(d <= row.knowledge_date for d in dates)
    nxt = any(row.knowledge_date < d <= row.knowledge_date + pd.Timedelta(days=365) for d in dates)
    return int(nxt), prior


lab = f.apply(label, axis=1, result_type="expand")
f["bk12"], f["already"] = lab[0], lab[1]
f = f[~f.already]
f["neg_eq"] = f.total_equity < 0
f["loss"] = f.net_income < 0
f["lev"] = f.total_liabilities / f.total_assets
f["cash_cl"] = f.cash_and_equivalents / f.current_liabilities
print(f"statements: {len(f)} | companies {f.cik.nunique()} | bankruptcies within 12m: {int(f.bk12.sum())} | base rate {f.bk12.mean():.4f}")


def show(mask, name):
    s = f[mask]
    if len(s) == 0:
        return
    rate = s.bk12.mean()
    print(f"{name:52s} n={len(s):7d} events={int(s.bk12.sum()):4d} P(bk12)={rate:.4f}  relative risk x{rate / f.bk12.mean():.1f}")


show(f.neg_eq, "negative equity")
show(~f.neg_eq, "positive equity")
show(f.loss, "net loss")
show(f.neg_eq & f.loss, "negative equity AND net loss")
show(f.neg_eq & f.loss & (f.cash_cl < 0.2), "negative equity AND loss AND cash < 20% of current liab.")
show(f.loss & (f.lev > 0.9), "loss AND liabilities > 90% of assets")
show(f.loss & (f.lev > 1.0), "loss AND liabilities > assets")
show((f.lev > 1.0) & (f.cash_cl < 0.2), "liabilities > assets AND cash < 20% of current liab.")
show(f.neg_eq & ~f.loss, "negative equity but profitable")
for y0, y1 in ((2010, 2015), (2016, 2021)):
    m = f.knowledge_date.dt.year.between(y0, y1)
    s1, s0 = f[m & f.neg_eq & f.loss], f[m]
    print(f"  {y0}-{y1}: base {s0.bk12.mean():.4f} | neg equity & loss {s1.bk12.mean():.4f} (x{s1.bk12.mean() / max(s0.bk12.mean(), 1e-9):.1f}, n={len(s1)})")
