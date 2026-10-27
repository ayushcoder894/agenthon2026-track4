"""Download quarterly estimated-vs-reported EPS history (Alpha Vantage EARNINGS endpoint) for large caps, resumable and rate-limited.
usage: python alpha_earnings.py <API_KEY> [max_calls]   Output: data_alpha/<TICKER>.json. The key is never written to disk or into any image.
Free tier = 25 requests/day (the script stops cleanly at max_calls and continues next run); a paid key allows 75/min (set PER_MIN)."""
import json, os, sys, time
import pandas as pd, requests

key = sys.argv[1]; max_calls = int(sys.argv[2]) if len(sys.argv) > 2 else 25
PER_MIN = 5  # free-tier burst limit
os.makedirs("data_alpha", exist_ok=True)
ev = pd.read_parquet("t4-local/dev/postearn_events.parquet")
order = ev.groupby("ticker").rev.median().sort_values(ascending=False).index.tolist()    # largest companies first
calls = 0
for t in order:
    f = f"data_alpha/{t}.json"
    if os.path.exists(f): continue
    if calls >= max_calls: break
    r = requests.get("https://www.alphavantage.co/query", params={"function": "EARNINGS", "symbol": t, "apikey": key}, timeout=60)
    calls += 1
    try: j = r.json()
    except ValueError: continue
    if "quarterlyEarnings" not in j:
        print(t, "->", str(j)[:120]);
        if "rate limit" in str(j).lower() or "premium" in str(j).lower() or "25 requests" in str(j): break
        continue
    json.dump(j, open(f, "w")); print(t, len(j["quarterlyEarnings"]), "quarters", flush=True)
    time.sleep(60.0 / PER_MIN + 0.2)
print("calls", calls, "| tickers done", len(os.listdir("data_alpha")))
