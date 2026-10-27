"""Prototype: read (short-term debt, total current liabilities, cash) from a 10-Q/10-K balance sheet in the flattened text.
Validate against ZipLime pit XBRL values on the SEC sample (data_sec/txt). Ratio rule: STdebt/CL > 0.6 and cash/CL < 0.2 (and a net loss)."""
import glob, re, sys, json
import numpy as np, pandas as pd

SP = r"[\s ​]"
NUM = SP + r"*\$?" + SP + r"*(\()?" + SP + r"*(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?" + SP + r"*(\))?"
CL_ROW = re.compile(r"(?i)total" + SP + r"+current" + SP + r"+liabilities" + NUM)
CASH_ROW = re.compile(r"(?i)cash" + SP + r"+and" + SP + r"+(?:cash" + SP + r"+)?equivalents(?!" + SP + r"+(?:at|of|were|was|are|is|increased|decreased))" + NUM)
STD_ROWS = [  # (class, regex)
    ("borrow", re.compile(r"(?i)(?:short" + SP + r"*-?" + SP + r"*term" + SP + r"+(?:debt|borrowings|borrowing|bank" + SP + r"+debt|notes" + SP + r"+payable|loans)|notes" + SP + r"+payable|commercial" + SP + r"+paper|(?:debt|borrowings)," + SP + r"+current|current" + SP + r"+debt)" + NUM)),
    ("curlt", re.compile(r"(?i)current" + SP + r"+(?:portion|maturities|installments)" + SP + r"+of" + SP + r"+(?:long" + SP + r"*-?" + SP + r"*term" + SP + r"+)?(?:debt|borrowings|notes|obligations|convertible" + SP + r"+notes|senior" + SP + r"+notes)(?:" + SP + r"+and" + SP + r"+(?:finance|capital)" + SP + r"+lease" + SP + r"+obligations)?" + NUM)),
]


def val(m):
    v = float(m.group(2).replace(",", ""))
    return -v if (m.group(1) and m.group(3)) else v


def read_balance(text):
    """-> dict(cl, cash, std) in the statement's own unit, or None."""
    for m in CL_ROW.finditer(text):
        cl = val(m)
        if cl <= 0:
            continue
        win_start = max(0, m.start() - 5000)
        win = text[win_start:m.start()]
        if not re.search(r"(?i)total" + SP + r"+current" + SP + r"+assets", win):
            continue  # not a balance sheet liabilities block
        cash = None
        for cm in CASH_ROW.finditer(text[max(0, win_start - 4000):m.start()]):
            cash = val(cm); break
        std, seen = 0.0, set()
        for cls, rx in STD_ROWS:
            for sm in rx.finditer(win):
                if cls in seen:
                    break
                seen.add(cls); std += abs(val(sm)); break
        return {"cl": cl, "cash": cash, "std": std, "pos": m.start()}
    return None


if __name__ == "__main__":
    idx = json.load(open("data_sec/index.json"))
    pit = pd.concat([pd.read_parquet(f, columns=["cik", "period_end", "quarters", "form", "knowledge_date", "revision", "current_liabilities", "short_term_debt", "cash_and_equivalents"]) for f in sorted(glob.glob("data_ziplime/pit/**/*.parquet", recursive=True))])
    pit = pit[pit.form.astype(str).str.startswith("10-") & pit.current_liabilities.notna()]
    pit["period_end"] = pd.to_datetime(pit.period_end)
    pit = pit.sort_values(["cik", "period_end", "knowledge_date", "revision"]).groupby(["cik", "period_end"], as_index=False).first()
    by = {(r.cik, r.period_end): r for r in pit.itertuples()}
    n = ok_cl = ok_cash = ok_std = both = std_wrong = 0; ratios = []
    flag_p = flag_t = 0
    for a, m in idx.items():
        try: t = open(f"data_sec/txt/{a}.txt", encoding="utf-8").read()
        except FileNotFoundError: continue
        g = read_balance(t); r = by.get((m["cik"], pd.Timestamp(m["period"])))
        if r is None: continue
        n += 1
        if g is None: continue
        scale = None
        for s_ in (1, 1e3, 1e6):
            if abs(g["cl"] * s_ - r.current_liabilities) / abs(r.current_liabilities) < 0.005: scale = s_; break
        if scale is None: continue
        ok_cl += 1
        if g["cash"] is not None and r.cash_and_equivalents == r.cash_and_equivalents and abs(g["cash"] * scale - r.cash_and_equivalents) / max(abs(r.cash_and_equivalents), 1) < 0.005: ok_cash += 1
        tstd = r.short_term_debt if r.short_term_debt == r.short_term_debt else 0.0
        if abs(g["std"] * scale - tstd) <= 0.01 * max(r.current_liabilities, 1) : ok_std += 1
        else: std_wrong += 1
        gr = g["std"] / g["cl"]; tr = tstd / r.current_liabilities
        ratios.append((gr, tr))
        flag_p += gr > 0.6; flag_t += tr > 0.6
    print(f"filings with pit values {n} | CL read & matches pit {ok_cl} ({ok_cl / n:.3f}) | cash matches {ok_cash} ({ok_cash / max(ok_cl, 1):.3f} of CL-ok) | ST debt within 1% of CL {ok_std} ({ok_std / max(ok_cl, 1):.3f})")
    R = np.array(ratios)
    print(f"ratio flag >0.6: parsed {flag_p} vs pit {flag_t}; corr {np.corrcoef(R[:, 0], R[:, 1])[0, 1]:.3f}; mean abs diff {np.mean(np.abs(R[:, 0] - R[:, 1])):.4f}; |diff|>0.1 in {int(np.sum(np.abs(R[:, 0] - R[:, 1]) > 0.1))} filings")
