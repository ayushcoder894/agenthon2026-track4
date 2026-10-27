"""Conservative extraction of the latest quarter's diluted EPS pair (current, prior-year) from a company's own 10-Q text.

Returns a pair only when the evidence is unambiguous; anything else returns None and the caller keeps its base rate.

A diluted-EPS row is a label ("Diluted", "Diluted EPS", "Diluted earnings per common share", "Earnings per share
- assuming dilution", "Net income per share - diluted", optionally followed by footnote marks) and then a run of numbers:
  * 2 numbers: current, prior-year
  * 4 numbers: three-month current, three-month prior-year, year-to-date current, year-to-date prior-year
  * 6 numbers: current, prior-year, % change, YTD current, YTD prior-year, % change  (accepted only when BOTH percent
    columns match the values to rounding, which verifies the layout)
Every EPS number carries two decimals (share counts and other figures never match). Parentheses mean negative.
Rows that follow adjusted / non-GAAP / pro forma / reconciliation language are ignored (they are not the reported EPS).
All accepted rows in the document must agree on the first two numbers, otherwise the pair is refused.

Stress test on real 10-Q filings (SEC EDGAR, 2015-2022, official XBRL diluted EPS as truth): see ARTIFACT_PROVENANCE.md.
"""
from __future__ import annotations

import os
import re

_WS = r"[\s ​]"
_D = r"[—–−\-�]"          # dashes; the corpus turns non-ASCII punctuation into U+FFFD
_SLASH = r"(?:income|earnings)/\((?:loss)\)"   # "income/(loss)"
_WORDS = (_SLASH + r"|shareholders|stockholders|shareowners|earnings|eps|income|net|per|common|shares|share|stock|from|continuing|operations|"
          r"attributable|to|total|loss|\(loss\)|\([a-z0-9]{1,2}\)|[:—–\-�]")

# Test switches (read once; the shipped defaults are the validated ones)
YTD_GUARD = os.environ.get("T4_EPS_YTD_GUARD", "none")        # strict | soft | none
EXCLUDE_CTX = os.environ.get("T4_EPS_EXCLUDE_CTX", "1") == "1"
SKIP_CONT = os.environ.get("T4_EPS_SKIP_CONT", "1") == "1"      # ignore "... from continuing operations" rows (not the headline GAAP EPS)
SKIP_DISC = os.environ.get("T4_EPS_SKIP_DISC", "0") == "1"      # ignore rows that sit under a "discontinued operations" heading

_LABEL = re.compile(
    r"(?:(?:total" + _WS + r"+)?[Dd]iluted(?:" + _WS + r"+(?:" + _WORDS + r"))*"
    r"|[Ee]arnings" + _WS + r"+per" + _WS + r"+diluted" + _WS + r"+(?:common" + _WS + r"+)?share"
    r"|[Ee]arnings" + _WS + r"+per" + _WS + r"+(?:common" + _WS + r"+)?share(?:" + _WS + r"+of" + _WS + r"+common" + _WS + r"+stock)?"
    + _WS + r"*[,:]?" + _WS + r"*" + _D + r"?" + _WS + r"*(?:assuming" + _WS + r"+dilution|diluted)"
    r"|(?:[Nn]et" + _WS + r"+)?(?:income|earnings)(?:" + _WS + r"+\(loss\))?" + _WS + r"+per" + _WS + r"+(?:common" + _WS + r"+)?share"
    + _WS + r"*[,:]?" + _WS + r"*" + _D + r"+" + _WS + r"*diluted"
    r"|[Bb]asic" + _WS + r"+and" + _WS + r"+diluted(?:" + _WS + r"+(?:" + _WORDS + r"))*"
    r"|(?:[Nn]et" + _WS + r"+)?(?:income|loss|earnings)(?:" + _WS + r"+\(loss\))?" + _WS + r"+per" + _WS + r"+(?:common" + _WS + r"+)?share[,—–\-�]?" + _WS + r"*basic" + _WS + r"+and" + _WS + r"+diluted)"
    r"(?:" + _WS + r"+(?:attributable|available)" + _WS + r"+to" + _WS + r"+[^\d$()%:;]{1,80}?(?=" + _WS + r"*[$\d(]))?"
    r"(?:" + _WS + r"*\([a-z0-9]{1,2}\)|" + _WS + r"*\*)*",
    re.I,
)
_TOK = re.compile(
    r"[\s ​$]*(?P<open>\()?[\s ​$]*(?P<num>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\.\d+)"
    r"[\s ​]*(?P<close>\))?[\s ​]*(?P<pct>%)?"
)
_EXCLUDE_BROAD = re.compile(r"pro\s*forma|non-?gaap|adjusted|reconcil|excluding|as\s+reported\s+and", re.I)
_EXCLUDE_NARROW = re.compile(r"pro\s*forma|non-?gaap|adjusted\s+(?:diluted|earnings|eps|net\s+income\s+per|income\s+per)|as\s+adjusted|reconcil(?:e|iation)s?\s+(?:of|to)|excluding", re.I)
_EXCLUDE = _EXCLUDE_NARROW


def _tokens(text: str, pos: int):
    out = []
    while len(out) < 9:
        m = _TOK.match(text, pos)
        if not m:
            break
        raw = m.group("num")
        val = float(raw.replace(",", ""))
        out.append({"v": -val if m.group("open") else val, "dec": len(raw.split(".")[1]) if "." in raw else 0})
        pos = m.end()
    return out, pos


def _row(tokens):
    """-> (current, prior) or None."""
    tokens = list(tokens)
    if len(tokens) > 2 and tokens[-1]["v"] < 0 and tokens[-1]["dec"] == 0 and abs(tokens[-1]["v"]) <= 9:
        tokens.pop()  # a trailing footnote mark such as "(1)"
    n = len(tokens)

    def pct_ok(cur, pri, p):
        if pri["v"] == 0:
            return False
        calc = (cur["v"] - pri["v"]) / abs(pri["v"]) * 100.0
        return abs(calc - p["v"]) <= max(1.0, 0.03 * abs(calc))

    if n == 2 and all(t["dec"] == 2 for t in tokens):
        return tokens[0]["v"], tokens[1]["v"]
    if n == 3 and tokens[0]["dec"] == 2 and tokens[1]["dec"] == 2 and pct_ok(tokens[0], tokens[1], tokens[2]):
        return tokens[0]["v"], tokens[1]["v"]
    if n == 4 and all(t["dec"] == 2 for t in tokens):
        a, b, c, d = tokens
        if YTD_GUARD == "none":
            return a["v"], b["v"]
        # year-to-date includes the quarter: its magnitude cannot be much smaller (rejects adjustment tables)
        if abs(c["v"]) >= 0.8 * abs(a["v"]) and abs(d["v"]) >= 0.8 * abs(b["v"]):
            return a["v"], b["v"]
        if YTD_GUARD == "soft" and abs(c["v"]) >= 0.8 * abs(a["v"]) and abs(d["v"]) >= 0.8 * abs(b["v"]) - 1.0:
            return a["v"], b["v"]  # an earlier loss quarter may lower the prior-year year-to-date figure below the quarter's
        return None
    if n == 6:
        a, b, c, d, e, f = tokens
        if all(t["dec"] == 2 for t in (a, b, d, e)) and pct_ok(a, b, c) and pct_ok(d, e, f):
            return a["v"], b["v"]
    return None


_YEARS = re.compile(r"(20\d\d)(?:[\s,$]|[A-Za-z]+\.?\s+\d{1,2},)*?(20\d\d)")


def column_order(text: str, start: int):
    """'desc' (current year first, the usual layout), 'asc' (older year first) or None, from the nearest pair of
    adjacent years in the table header before the row."""
    win = text[max(0, start - 2500): start]
    last = None
    for m in _YEARS.finditer(win):
        y1, y2 = int(m.group(1)), int(m.group(2))
        if abs(y1 - y2) == 1:
            last = (y1, y2)
    if last is None:
        return None
    return "asc" if last[0] < last[1] else "desc"


_NARR = re.compile(
    r"[Dd]iluted\s+(?:earnings|EPS|earnings\s+per\s+(?:common\s+)?share)[^.$]{0,40}?\$\s?(\d{1,3}\.\d{2})[^.$]{0,30}?"
    r"compared\s+(?:to|with)[^.$]{0,40}?\$\s?(\d{1,3}\.\d{2})([^.]{0,140})")


def parse_narrative(text: str):
    """'Diluted earnings per share were $2.18, compared to $1.88 in the same quarter of the prior year.' -> (cur, prior, raw, start, end).
    Only when the sentence says it compares the quarter (never a year-to-date comparison)."""
    found = []
    for m in _NARR.finditer(text):
        tail = m.group(3).lower()
        window = text[max(0, m.start() - 160): m.end()].lower()
        if not re.search(r"(same\s+quarter|three\s+months|second\s+quarter|first\s+quarter|third\s+quarter|fourth\s+quarter)", tail + window[-200:]):
            continue
        if re.search(r"(nine\s+months|six\s+months|fiscal\s+year|year\s+ended|first\s+half|first\s+nine)", tail):
            continue
        found.append((float(m.group(1)), float(m.group(2)), " ".join(m.group(0).split()), m.start(), m.end()))
    if not found or len({(a, b) for a, b, _, _, _ in found}) != 1:
        return None
    return found[0]


ATTR_LABELS = os.environ.get("T4_EPS_BLOCK", "1") == "1"
MAJORITY = os.environ.get("T4_EPS_MAJORITY", "1") == "1"
_NET = re.compile(r"(?i)(?<![A-Za-z0-9_])(?:net" + _WS + r"+(?:income|earnings|loss|\(loss\)" + _WS + r"*(?:income|earnings)|income" + _WS + r"*\(loss\)|earnings" + _WS + r"*\(loss\))(?![A-Za-z0-9_])"
                  r"|(?:earnings|income)(?:" + _WS + r"*\(loss\))?" + _WS + r"+attributable" + _WS + r"+to(?![A-Za-z0-9_]))"
                  r"(?!" + _WS + r"*(?:per|from|continuing|attributable\s+to\s+non|margin|before|tax))[^\d$()%]{0,60}?(?=" + _WS + r"*[$\d(])")
_DIL_HDR = re.compile(r"(?i)(?<![A-Za-z0-9_])diluted(?![A-Za-z0-9_])" + _WS + r"*:?" + _WS + r"*(?=(?:​| |\s)*(?:income|earnings|continuing|net(?![A-Za-z0-9_])))")


def _block_rows(text: str):
    """'Diluted:' header followed by sub-rows (continuing operations / discontinued / Net income): the Net income row's pair."""
    out = []
    for h in _DIL_HDR.finditer(text):
        win = text[h.end(): h.end() + 700]
        if re.search(r"(?i)(?<![A-Za-z0-9_])basic" + _WS + r"*:|per" + _WS + r"+share" + _WS + r"+basic", win[:700]):
            cut = re.search(r"(?i)(?<![A-Za-z0-9_])basic", win)
            win = win[:cut.start()] if cut else win
        for n in _NET.finditer(win):
            toks, end = _tokens(win, n.end())
            pair = _row(toks) if toks else None
            if pair is not None:
                out.append((pair, h.start(), h.end() + end))
                break
    return out


def _parse(text: str, exclude):
    found = []
    for m in _LABEL.finditer(text):
        if SKIP_CONT and "continuing" in m.group(0).lower():
            continue
        if SKIP_DISC and re.search(r"discontinued" + _WS + r"+operations[^.$]{0,60}$", text[max(0, m.start() - 90): m.start()], re.I):
            continue
        toks, end = _tokens(text, m.end())
        if not toks:
            continue
        pair = _row(toks)
        if pair is not None:
            if exclude is not None and exclude.search(text[max(0, m.start() - 350): m.start()]):
                continue  # adjusted / non-GAAP / pro forma figures are not the reported diluted EPS
            found.append((pair[0], pair[1], " ".join(text[m.start(): end].split()), m.start(), end))
    if not found and ATTR_LABELS:
        for pair, st, en in _block_rows(text):
            if exclude is not None and exclude.search(text[max(0, st - 350): st]):
                continue
            found.append((pair[0], pair[1], " ".join(text[st: en].split()), st, en))
    if not found:
        return None
    if len({(a, b) for a, b, _, _, _ in found}) != 1:
        if not MAJORITY:
            return None
        import collections

        c = collections.Counter(a for a, _, _, _, _ in found).most_common()
        if len(c) > 1 and c[0][1] == c[1][1]:
            return None
        cur = c[0][0]
        found = [x for x in found if x[0] == cur]
        if len({(a, b) for a, b, _, _, _ in found}) != 1:
            found = found[:1]   # same current quarter, different prior (restated / split-adjusted): keep the first row
    a, b, raw, st, en = found[0]
    if column_order(text, st) == "asc":
        a, b = b, a  # the older year is printed first
    return a, b, raw, st, en


def parse_pair(text: str):
    """-> (current, prior, raw_line, start, end) or None (start/end: offsets of the first matching row in `text`).
    Tiered: rows after any adjusted / non-GAAP / reconciliation wording are ignored first (the cleanest reading); if that gives no
    unambiguous pair, only rows after clearly non-GAAP wording are ignored; then the quarter-comparison sentence is tried.
    When rows disagree, the majority current-quarter value wins (a tie refuses)."""
    if not isinstance(text, str) or not text:
        return None
    tiers = (_EXCLUDE_BROAD, _EXCLUDE_NARROW) if EXCLUDE_CTX else (None,)
    for exclude in tiers:
        got = _parse(text, exclude)
        if got is not None:
            return got
    return parse_narrative(text)
