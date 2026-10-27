"""Experiments 51-54: agreement policies for the current-quarter EPS when rows conflict (on all sample filings with an official pair)."""
import sys, re, json, collections
sys.path.insert(0, "t4-local/agent")
from t4_agent import eps_parse as E
src = open("t4-local/dev/sec_eval.py").read().split("CONFIGS = ")[0]
exec(src)
texts = {a: text_of(a) for a in idx if os.path.exists(f"data_sec/raw/{a}.htm")}
gt = {a: official(idx[a]["cik"], pd.Timestamp(idx[a]["period"])) for a in texts}
valid = [a for a in texts if gt[a][0] is not None and gt[a][1] is not None]

def rows_for(t, excl):
    out = []
    for m in E._LABEL.finditer(t):
        if "continuing" in m.group(0).lower(): continue
        toks, end = E._tokens(t, m.end())
        if not toks: continue
        pair = E._row(toks)
        if pair is None: continue
        if excl is not None and excl.search(t[max(0, m.start() - 350): m.start()]): continue
        out.append((pair, m.start()))
    return out

def policy(rows, kind):
    if not rows: return None
    pairs = [p for p, _ in rows]
    if kind == "strict":  return pairs[0] if len(set(pairs)) == 1 else None
    if kind == "cur_agree": return pairs[0] if len({p[0] for p in pairs}) == 1 else None
    if kind == "first_row": return pairs[0]
    if kind == "majority":
        c = collections.Counter(pairs).most_common()
        return c[0][0] if len(c) == 1 or c[0][1] > c[1][1] else None
    if kind == "majority_cur":
        c = collections.Counter(p[0] for p in pairs).most_common()
        if len(c) == 1 or c[0][1] > c[1][1]:
            cur = c[0][0]; return next(p for p in pairs if p[0] == cur)
        return None
    if kind == "majority_cur_ge3": # require >=3 rows or unanimity
        c = collections.Counter(p[0] for p in pairs).most_common()
        if (len(c) == 1) or (c[0][1] >= 3 and c[0][1] > c[1][1]):
            cur = c[0][0]; return next(p for p in pairs if p[0] == cur)
        return None

print(f"valid {len(valid)}")
for kind in ("strict", "cur_agree", "majority_cur", "majority_cur_ge3", "majority", "first_row"):
    n = ok_cur = ok_pair = 0
    for a in valid:
        t = texts[a]
        r = None
        for excl in (E._EXCLUDE_BROAD, E._EXCLUDE_NARROW):
            r = policy(rows_for(t, excl), kind)
            if r is not None: break
        if r is None: continue
        cur, pri = r
        if E.column_order(t, 0) == "asc": pass
        n += 1; co, po = gt[a]
        ok_cur += abs(cur - co) < 0.006; ok_pair += (abs(cur - co) < 0.006 and abs(pri - po) < 0.006)
    print(f"{kind:18s} coverage {n / len(valid):.3f} ({n}) | current correct {ok_cur / n:.3f} | pair correct {ok_pair / n:.3f} | right currents: {ok_cur}")
