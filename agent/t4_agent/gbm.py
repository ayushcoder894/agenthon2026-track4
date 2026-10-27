"""Pure-Python evaluator for exported LightGBM trees (numeric splits, '<=' decisions). No ML library is needed at run time; the learned trees live in artifacts.json."""
from __future__ import annotations

import math


def _walk(node: dict, x: list) -> float:
    while "leaf" not in node:
        v = x[node["f"]]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            node = node["l"] if node.get("dl", True) else node["r"]
        else:
            node = node["l"] if v <= node["t"] else node["r"]
    return node["leaf"]


def predict(model: dict, x: list) -> float:
    """Sum of the leaf values of all trees (the init score is folded into the first tree by LightGBM)."""
    return sum(_walk(t, x) for t in model["trees"])


def convert(dump: dict) -> dict:
    """lightgbm Booster.dump_model() -> compact {'trees': [...]} with feature indices, thresholds, default directions and leaves."""
    def rec(n):
        if "leaf_value" in n:
            return {"leaf": float(n["leaf_value"])}
        assert n["decision_type"] == "<=", n["decision_type"]
        return {"f": int(n["split_feature"]), "t": float(n["threshold"]), "dl": bool(n.get("default_left", True)), "l": rec(n["left_child"]), "r": rec(n["right_child"])}
    return {"trees": [rec(t["tree_structure"]) for t in dump["tree_info"]]}
