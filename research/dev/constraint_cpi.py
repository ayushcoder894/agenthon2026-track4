"""Constraint-based CPI: (a) core = sum of component forecasts + residual 'other core'; (b) energy = 0.5 x gasoline forecast + 0.5 x other-energy forecast.
Fixed approximate relative-importance weights (BLS, Dec 2023): within core: shelter .453, new vehicles .054, used cars .024, apparel .031, medical .084, transportation services .079, other = rest;
within energy: gasoline .5. Skill = MAE(last print)/(MAE(last) + MAE(model)), test 2016-2022H1."""
import json
import numpy as np, pandas as pd
exec(open("t4-local/dev/cpi_study.py").read().split('print("=== per-component')[0])
fit = json.load(open("t4-local/dev/cpi_pump_fit.json"))
def rule(c): s = M[c]; return 0.5 * s.shift(1) + 0.5 * s.shift(1).rolling(6).mean()
def pump_pred(c):
    f = fit[c]; d = pd.DataFrame({"pump": M.pump, "m6": M[c].shift(1).rolling(6).mean(), "core6": M.CPI_CORE.shift(1).rolling(6).mean(), "food6": M.CPI_FOOD.shift(1).rolling(6).mean()})
    return f["intercept"] + sum(cf * d[ft] for ft, cf in zip(f["feats"], f["coef"])) + M.month.map(lambda m: f["month"][m - 1])
te_i = M.index.year >= 2016
def skill(y, p, last): d = pd.DataFrame({"y": y, "p": p, "l": last}).dropna(); d = d[d.index.year >= 2016]; a, b = np.abs(d.y - d.l).mean(), np.abs(d.y - d.p).mean(); return a / (a + b), len(d)
# (a) core
W = {"CPI_SHELTER": .453, "CPI_NEWVEH": .054, "CPI_USEDCARS": .024, "CPI_APPAREL": .031, "CPI_MEDICAL": .084, "CPI_TRANSPSVC": .079}
w_o = 1 - sum(W.values())
other = (M.CPI_CORE - sum(w * M[c] for c, w in W.items())) / w_o
other_pred = 0.5 * other.shift(1) + 0.5 * other.shift(1).rolling(6).mean()
core_c = sum(w * rule(c) for c, w in W.items()) + w_o * other_pred
core_direct = rule("CPI_CORE")
for nm, p in (("direct rule (0.5 last + 0.5 mean6)", core_direct), ("constructed from components + other", core_c), ("average of both", 0.5 * core_direct + 0.5 * core_c)):
    s, n = skill(M.CPI_CORE, p, M.CPI_CORE.shift(1)); print(f"core   {nm:40s} skill {s:.3f} (n={n})")
# (b) energy
other_en = (M.CPI_ENERGY - 0.5 * M.CPI_GASOLINE) / 0.5
oe_pred = 0.5 * other_en.shift(1) + 0.5 * other_en.shift(1).rolling(6).mean()
en_c = 0.5 * pump_pred("CPI_GASOLINE") + 0.5 * oe_pred
for nm, p in (("pump model (shipped)", pump_pred("CPI_ENERGY")), ("constructed: .5 gasoline pump + .5 other", en_c), ("average of both", 0.5 * pump_pred("CPI_ENERGY") + 0.5 * en_c)):
    s, n = skill(M.CPI_ENERGY, p, M.CPI_ENERGY.shift(1)); print(f"energy {nm:40s} skill {s:.3f} (n={n})")
# (c) headline built from the constructed core / energy
food = rule("CPI_FOOD")
for nm, core_p, en_p in (("shipped (rule core, pump energy)", core_direct, pump_pred("CPI_ENERGY")), ("constructed core + constructed energy", core_c, en_c), ("constructed core + pump energy", core_c, pump_pred("CPI_ENERGY"))):
    head = 0.135 * food + 0.07 * en_p + 0.795 * core_p
    s, n = skill(M.CPI_ALLITEMS, head, M.CPI_ALLITEMS.shift(1)); print(f"headline {nm:40s} skill {s:.3f} (n={n})")
