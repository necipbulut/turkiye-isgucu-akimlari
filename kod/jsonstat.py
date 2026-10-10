# -*- coding: utf-8 -*-
"""Minimal JSON-stat 2.0 reader for Eurostat API responses (values and status flags)."""
import json

import pandas as pd


def read(path):
    d = json.load(open(path, encoding="utf-8"))
    ids, size = d["id"], d["size"]
    cats = []
    for k in ids:
        idx = d["dimension"][k]["category"]["index"]
        cats.append(sorted(idx, key=lambda c: idx[c]))
    labels = {k: d["dimension"][k]["category"].get("label", {}) for k in ids}
    strides = [1] * len(size)
    for i in range(len(size) - 2, -1, -1):
        strides[i] = strides[i + 1] * size[i + 1]
    status = d.get("status", {}) or {}
    rows = []
    keys = set(d["value"].keys()) | set(status.keys())
    for key in keys:
        n = int(key)
        rec = {k: cats[i][n // strides[i] % size[i]] for i, k in enumerate(ids)}
        rec["value"] = d["value"].get(key)
        rec["flag"] = status.get(key, "")
        rows.append(rec)
    return pd.DataFrame(rows), labels
