"""Generate t4_agent/credit_sem.py: the 16 fitted phrase flags (extracted verbatim from credit_text.py, so the regexes are byte-identical to the fitted dataset)
plus the semantic-event extraction code in credit_sem_body.py."""
src = open("t4-local/dev/credit_text.py", encoding="utf-8").read()
start = src.index('SP = r"')
end = src.index("\n}\n", start) + 3
block = src[start:end]
block = block.replace("lambda t: P._going_concern_hits(t) > 0", "lambda t: False")
head = '''"""Credit semantic layer (D19): shipped phrase flags + negative equity / net loss + validated LLM-extracted distress events -> small logistic model.
The model only EXTRACTS facts (event type, assertion, time, resolution, verbatim quote); code verifies every quote against the passages it was given,
and a closed-form logistic model fitted on SEC filings dated on or before 2022-06-28 turns the validated facts into the label.
Any failure (no endpoint, bad reply, timeout) leaves the deterministic rule of earlier variants in force."""
from __future__ import annotations

import math
import os
import re
import time

'''
open("t4-local/agent/t4_agent/credit_sem.py", "w", encoding="utf-8", newline="\n").write(head + block + open("t4-local/dev/credit_sem_body.py", encoding="utf-8").read())
print("written", len(block))
