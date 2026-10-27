"""Build the portable EPS panel (parquet) from the PIT/ZipLime history. Run on the laptop; the panel (not the 131MB PIT data) is what the A100 needs."""
import numpy as np, pandas as pd
src = open("t4-local/dev/eps_decomp.py").read().split("SETS = {")[0]
exec(src)
lag = F[["cik", "period_end", "eps_diluted_l1", "eps_diluted_l2", "eps_diluted_l3", "eps_diluted_l4", "f_ni_yoy", "f_ni_rev", "f_int_rev", "f_nonop", "f_pm", "f_etr", "f_roa", "f_eq_assets", "f_trend"]]
X = D.merge(lag, on=["cik", "period_end"], how="left", suffixes=("", "_dup"))
X = X.loc[:, ~X.columns.str.endswith("_dup")]
X["split"] = np.where(X.pe_next <= "2016-12-31", "train", np.where(X.pe_next <= "2019-12-31", "val", "final"))
X["block"] = X.pe_next.dt.year
X["kd_next"] = X.kd_next.dt.tz_localize(None) if getattr(X.kd_next.dt, "tz", None) is not None else X.kd_next
keep = ["cik", "period_end", "pe_next", "kd_next", "split", "block", "fin", "rev_last", "abs_prior", "eps_last", "eps_prior_next", "eps_diluted_l1", "eps_diluted_l2", "eps_diluted_l3", "eps_diluted_l4",
        "g_last", "rr", "y", "up_next", "base", "b"] + [c for c in X.columns if c.startswith("f_")]
X = X[[c for c in keep if c in X.columns]].copy(); X["fin"] = X.fin.astype(bool)
X.to_parquet("t4-local/dev/exp500/panels/eps_panel.parquet"); print(X.split.value_counts().to_dict(), X.shape)
