"""Fetch public FOMC statements (federalreserve.gov) 2000-2022 for OFFLINE training of the FOMC text model. Output: data_fomc/statements.jsonl {date, url, text}."""
import json, re, time, urllib.request, html, os
UA = {"User-Agent": "Mozilla/5.0"}
def get(u):
    for _ in range(3):
        try:
            return urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30).read().decode("utf-8", "ignore")
        except Exception:
            time.sleep(2)
    return ""
urls = {}
for y in range(2000, 2023):
    h = get(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm")
    for m in re.finditer(r'href="([^"]*?(?:monetary/?(\d{8})a\.htm|boarddocs/press/monetary/\d{4}/(\d{8})/[^"]*))"', h):
        d = m.group(2) or m.group(3); u = m.group(1)
        urls.setdefault(d, u if u.startswith("http") else "https://www.federalreserve.gov" + u)
    print(y, len([d for d in urls if d.startswith(str(y))]), flush=True); time.sleep(0.5)
h = get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
for m in re.finditer(r'href="([^"]*?monetary/?(\d{8})a\.htm)"', h):
    if m.group(2) <= "20221231": urls.setdefault(m.group(2), m.group(1) if m.group(1).startswith("http") else "https://www.federalreserve.gov" + m.group(1))
out = open("data_fomc/statements.jsonl", "w", encoding="utf-8")
for d in sorted(urls):
    h = get(urls[d])
    h = re.sub(r"(?is)<(script|style).*?</\1>", " ", h); t = html.unescape(re.sub(r"<[^>]+>", " ", h)); t = re.sub(r"\s+", " ", t)
    i = t.find("Release Date"); t = t[i:] if i >= 0 else t
    j = t.find("For media inquiries"); t = t[:j] if j > 0 else t
    out.write(json.dumps({"date": f"{d[:4]}-{d[4:6]}-{d[6:]}", "url": urls[d], "text": t}) + "\n"); out.flush(); time.sleep(0.4)
print("done", len(urls))
