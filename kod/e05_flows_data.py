# -*- coding: utf-8 -*-
"""e05_flows_data.py - data for the interactive labour-flow map (akim_haritasi.html).

Eurostat, Türkiye, 15-74, seasonally adjusted, 2014Q1-2026Q2:
  lfsi_long_q  quarterly transitions between employment (E), unemployment (U) and inactivity (N):
               thousand persons and percent of the origin state (2021Q1 missing)
  lfsi_emp_q   labour force and employment (thousand, percent of population)
  lfsi_sla_q   underemployed part-time workers and the potential labour force
Steady-state unemployment uses the same three-state chain as e02 (4-quarter averages of the rates).
Writes ../out/akim_data.json and fills it into ../page_akim/akim_template.html -> ../akim_haritasi.html.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jsonstat import read

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA, OUT = ROOT / "data" / "eurostat", ROOT / "out"
START, END = "2014Q1", "2026Q2"
K = ["EMP_UNE", "EMP_INAC", "UNE_EMP", "UNE_INAC", "INAC_EMP", "INAC_UNE"]
SHORT = {"EMP_UNE": "EU", "EMP_INAC": "EN", "UNE_EMP": "UE", "UNE_INAC": "UN", "INAC_EMP": "NE", "INAC_UNE": "NU"}
UNIT = {"EMP": "PC_EMP", "UNE": "PC_UNE", "INAC": "PC_INAC"}


def ser(df, **f):
    for k, v in f.items():
        df = df[df[k] == v]
    s = df.set_index("time")["value"].astype(float).sort_index()
    s.index = pd.PeriodIndex([x.replace("-", "") for x in s.index], freq="Q")
    return s


lg, _ = read(DATA / "lfsi_long_q.json")
em, _ = read(DATA / "lfsi_emp_q.json")
sl, _ = read(DATA / "lfsi_sla_q.json")
idx = pd.period_range(START, END, freq="Q")
cnt = pd.DataFrame({SHORT[k]: ser(lg, s_adj="SA", sex="T", unit="THS_PER", wstatus=k) for k in K}).reindex(idx)
rate = pd.DataFrame({SHORT[k]: ser(lg, s_adj="SA", sex="T", unit=UNIT[k.split("_")[0]], wstatus=k) for k in K}).reindex(idx)
act = ser(em, s_adj="SA", sex="T", age="Y15-74", unit="THS_PER", indic_em="ACT")
actp = ser(em, s_adj="SA", sex="T", age="Y15-74", unit="PC_POP", indic_em="ACT")
emp = ser(em, s_adj="SA", sex="T", age="Y15-74", unit="THS_PER", indic_em="EMP_LFS")
st = pd.DataFrame({"ACT": act, "E": emp, "POP": act / (actp / 100)}).reindex(idx)
st["U"] = st.ACT - st.E
st["N"] = st.POP - st.ACT
w = {w_: ser(sl, s_adj="SA", sex="T", age="Y15-74", unit="THS_PER", wstatus=w_).reindex(idx)
     for w_ in ["NSEEK_AVL", "SEEK_NAVL", "UEMP_PT"]}
st["PLF"] = w["NSEEK_AVL"] + w["SEEK_NAVL"]
st["PT"] = w["UEMP_PT"]
st["u"] = st.U / st.ACT * 100
st["lfpr"] = st.ACT / st.POP * 100
st["want"] = (st.ACT + st.PLF) / st.POP * 100
st["slack"] = (st.U + st.PT + st.PLF) / (st.ACT + st.PLF) * 100


def steady(p):
    eu, en, ue, un, ne, nu = (p[k] / 100 for k in ["EU", "EN", "UE", "UN", "NE", "NU"])
    M = np.array([[1 - eu - en, eu, en], [ue, 1 - ue - un, un], [ne, nu, 1 - ne - nu]])
    wv, v = np.linalg.eig(M.T)
    s = np.real(v[:, np.argmin(abs(wv - 1))])
    s = s / s.sum()
    return s[1] / (s[0] + s[1]) * 100


r4 = rate.rolling(4).mean()  # NaN where a window touches the missing 2021Q1
st["ustar"] = [steady(r4.loc[i]) if r4.loc[i].notna().all() else np.nan for i in idx]
base = rate.loc[pd.Period("2014Q1", "Q"):pd.Period("2019Q4", "Q")].mean()

rows = []
for i in idx:
    r = {"q": str(i)}
    for c in ["E", "U", "N", "PLF", "PT"]:
        r[c] = None if pd.isna(st.loc[i, c]) else round(st.loc[i, c])
    for c in ["u", "lfpr", "want", "slack", "ustar"]:
        r[c] = None if pd.isna(st.loc[i, c]) else round(st.loc[i, c], 2)
    r["f"] = None if cnt.loc[i].isna().any() else {k: round(v) for k, v in cnt.loc[i].items()}
    r["r"] = None if rate.loc[i].isna().any() else {k: round(v, 1) for k, v in rate.loc[i].items()}
    rows.append(r)
payload = {"quarters": rows, "base": {k: round(v, 2) for k, v in base.items()},
           "meta": {"source": "Eurostat lfsi_long_q, lfsi_emp_q, lfsi_sla_q (TÜİK HİA), 15–74 yaş, mevsimsellikten arındırılmış",
                    "retrieved": "2026-10-06"}}
(OUT / "akim_data.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
tpl = (ROOT / "page_akim" / "akim_template.html").read_text(encoding="utf-8")
html = tpl.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False))
(ROOT / "akim_haritasi.html").write_text(html, encoding="utf-8")
print("quarters", len(rows), "first", rows[0]["q"], "last", rows[-1]["q"])
print({k: rows[-1][k] for k in ["E", "U", "N", "PLF", "PT", "u", "slack", "ustar", "want"]}, rows[-1]["f"])
print("base", payload["base"])
