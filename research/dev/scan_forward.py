"""Scan each public unit's corpus for forward-looking cues; count documents and matches and show up to 2 numeric examples per cue."""
import json, pathlib, re
U = pathlib.Path("track4-analysis-public/units")
CUES = {"guidance/outlook": r"(?i)\b(guidance|outlook)\b", "expects/anticipates": r"(?i)\b(we|company|management|committee|treasury)\s+(expect|expects|anticipate|anticipates|project|projects|forecast)\w*",
        "consensus/analyst": r"(?i)\b(consensus|analyst estimates?|street estimates?)\b", "projection/SEP": r"(?i)\b(summary of economic projections|median.{0,40}projection|projected path|dot plot)\b",
        "market-implied/priced": r"(?i)(market-implied|priced in|priced into|futures|implied)", "announced sizes/schedule": r"(?i)(tentative schedule|announced|will be announced|scheduled for)",
        "range of $ (guidance)": r"(?i)range of \$\s?\d|between \$\s?\d[\d.,]* (and|to) \$\s?\d"}
for u in sorted(p.name for p in U.iterdir() if p.is_dir()):
    docs = []
    for f in (U / u / "corpus").iterdir():
        if f.name == "manifest.json": continue
        try: j = json.load(open(f, encoding="utf-8")); docs.append((f.name, j.get("text", "")))
        except Exception: pass
    print(f"\n=== {u}: {len(docs)} docs, {sum(len(t) for _, t in docs):,} chars")
    for cue, rx in CUES.items():
        n = 0; ex = None
        for name, t in docs:
            for m in re.finditer(rx, t):
                n += 1
                if ex is None:
                    seg = t[max(0, m.start() - 60): m.end() + 110]
                    if re.search(r"\d", seg): ex = (name[:30], " ".join(seg.split()))
        if n: print(f"   {cue:26s} {n:5d}" + (f" | e.g. [{ex[0]}] ...{ascii(ex[1])[:170]}" if ex else ""))
