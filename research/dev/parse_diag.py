"""Categorise why the EPS parser refuses real filings (sample data_sec, filings with an official pair)."""
import sys, re, json, collections
sys.path.insert(0, "t4-local/agent")
from t4_agent import eps_parse as E
src = open("t4-local/dev/sec_eval.py").read().split("CONFIGS = ")[0]
exec(src)
texts = {a: text_of(a) for a in idx if os.path.exists(f"data_sec/raw/{a}.htm")}
gt = {a: official(idx[a]["cik"], pd.Timestamp(idx[a]["period"])) for a in texts}
valid = [a for a in texts if gt[a][0] is not None and gt[a][1] is not None]
cats = collections.defaultdict(list)
for a in valid:
    t = texts[a]
    if E.parse_pair(t) is not None: continue
    cur_o = gt[a][0]
    rows, cont, conflicts = [], 0, 0
    labels = list(E._LABEL.finditer(t))
    if not labels: cats["A no diluted label at all"].append(a); continue
    rawrows = []
    for m in labels:
        toks, end = E._tokens(t, m.end())
        pair = E._row(toks) if toks else None
        rawrows.append((m, toks, pair))
    withtok = [x for x in rawrows if x[1]]
    if not withtok: cats["B label found but no number follows"].append(a); continue
    paired = [x for x in withtok if x[2] is not None]
    if not paired:
        # tokens present but not a recognised layout
        n = collections.Counter(len(x[1]) for x in withtok).most_common(1)[0][0]
        cats[f"C numbers follow but layout rejected (most common token count {n})"].append(a); continue
    nocont = [x for x in paired if "continuing" not in x[0].group(0).lower()]
    if not nocont: cats["D only continuing-operations rows"].append(a); continue
    vals = {(x[2][0], x[2][1]) for x in nocont}
    if len(vals) > 1:
        hit = any(abs(v[0] - cur_o) < 0.006 for v in vals)
        cats["E conflicting rows (official current is among them)" if hit else "E2 conflicting rows (official current not among them)"].append(a); continue
    cats["F filtered by exclusion wording"].append(a)
tot = len(valid); nref = sum(len(v) for v in cats.values())
print(f"valid {tot}, refused {nref} ({nref / tot:.3f})")
for k, v in sorted(cats.items(), key=lambda kv: -len(kv[1])):
    print(f"  {len(v):4d}  {k}")
json.dump({k: v for k, v in cats.items()}, open("data_sec/refusal_cats.json", "w"))
