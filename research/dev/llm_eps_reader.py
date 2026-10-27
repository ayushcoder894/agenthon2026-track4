"""Feasibility test: LLM-assisted EPS reader on filings the regex parser refuses. The model sees up to 5 short passages around 'diluted' and must return the current and
prior-year diluted EPS plus the verbatim quote it read them from. Accepted only if the quote is an exact substring of the filing text and contains both numbers.
Scored against the official XBRL pair (current EPS is what the run-rate model needs)."""
import json, random, re, sys, time
import requests
sys.path.insert(0, "t4-local/agent")
from t4_agent import eps_parse as E
src = open("t4-local/dev/sec_eval.py").read().split("CONFIGS = ")[0]
exec(src)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
texts = {a: text_of(a) for a in idx if os.path.exists(f"data_sec/raw/{a}.htm")}
gt = {a: official(idx[a]["cik"], pd.Timestamp(idx[a]["period"])) for a in texts}
valid = [a for a in texts if gt[a][0] is not None and gt[a][1] is not None]
refused = [a for a in valid if E.parse_pair(texts[a]) is None]
random.Random(4).shuffle(refused); refused = refused[:N]
print("refused filings in the test:", len(refused), "of", sum(1 for a in valid if E.parse_pair(texts[a]) is None), flush=True)

def windows(t, k=4, w=170):
    cands = []
    for m in re.finditer(r"(?i)dilut", t):
        seg = t[max(0, m.start() - w): m.end() + w]
        nums = re.findall(r"(?<![\d,])\(?\d{1,3}\.\d{2}\)?(?![\d,])", seg)
        if len(nums) >= 2: cands.append((len(nums), m.start(), seg))
    cands.sort(key=lambda x: (-min(x[0], 4), x[1]))
    out, last = [], -10**9
    for _, pos, seg in cands:
        if pos - last < 400: continue
        out.append(seg); last = pos
        if len(out) >= k: break
    return out

def ask(wins):
    body = "\n\n".join(f"[{i + 1}] {w}" for i, w in enumerate(wins))
    prompt = ("These passages come from a company's quarterly report (10-Q). Find the income statement line for the quarter's GAAP diluted earnings per share (net, "
              "not from continuing operations, not adjusted). Return ONE JSON object only: {\"current\": <number for the most recent quarter>, \"prior\": <number for the same "
              "quarter a year earlier>, \"quote\": \"<the exact text from the passage that contains both numbers>\"}. If you cannot find it, return {\"current\": null}.\n\n" + body)
    r = requests.post("http://localhost:11434/api/chat", json={"model": "qwen2.5-coder:7b", "messages": [{"role": "user", "content": prompt}], "stream": False, "options": {"num_gpu": 0, "num_ctx": 4096, "temperature": 0, "num_predict": 160}}, timeout=900)
    return r.json()["message"]["content"]

def num(x):
    try: return float(str(x).replace("(", "-").replace(")", "").replace("$", "").replace(",", ""))
    except ValueError: return None

n_ans = n_ok = n_quote_ok = 0; rows = []
for i, a in enumerate(refused):
    t = texts[a]; co, po = gt[a]; wins = windows(t)
    if not wins: rows.append((a, "no windows")); continue
    try: reply = ask(wins)
    except Exception as exc: rows.append((a, f"error {type(exc).__name__}")); continue
    m = re.search(r"\{.*\}", reply, re.S)
    try: j = json.loads(m.group(0)) if m else {}
    except ValueError: j = {}
    cur, pri, quote = num(j.get("current")), num(j.get("prior")), j.get("quote") or ""
    if cur is None: rows.append((a, "null")); continue
    n_ans += 1
    verbatim = bool(quote) and " ".join(quote.split()) in " ".join(t.split()) and f"{abs(cur):.2f}".lstrip("0") in quote and (pri is None or f"{abs(pri):.2f}".lstrip("0") in quote)
    n_quote_ok += verbatim
    ok = abs(cur - co) < 0.006
    if verbatim: n_ok += ok
    rows.append((a, "answered", cur, pri, co, po, verbatim, ok))
    if (i + 1) % 10 == 0: print(f"  {i + 1}/{len(refused)} answered {n_ans} verified {n_quote_ok} verified+correct {n_ok}", flush=True)
print(f"RESULT on {len(refused)} refused filings: answered {n_ans}; quote verified verbatim {n_quote_ok}; of the verified, current EPS correct {n_ok} ({n_ok / max(n_quote_ok, 1):.3f}); "
      f"unverified answers correct {sum(1 for r in rows if len(r) > 7 and not r[6] and r[7])}/{sum(1 for r in rows if len(r) > 7 and not r[6])}")
json.dump(rows, open("data_sec/llm_reader_rows.json", "w"), default=str)
