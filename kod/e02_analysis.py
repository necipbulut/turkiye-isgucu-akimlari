# -*- coding: utf-8 -*-
"""e02_analysis.py - stocks, rates, underutilization, flows, hours and sectors by policy phase.

Phases follow the Essay 3 V2 draft: base = last quarter before each phase; rate-cut phase
2021Q3-2023Q2, tightening phase 2023Q3-2024Q2, NRIR episode 2021Q3-2024Q2, normalization
2024Q3-2026Q1; earlier windows 2010Q1-2017Q2, 2017Q3-2019Q4, 2020Q1-2021Q2.
Outputs in ../out (CSV tables and numbers.json). Nothing is estimated beyond identities,
a three-state steady state of the flow rates, and simple ratios.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jsonstat import read

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE.parent / "data", HERE.parent / "out"
V2 = HERE.parents[1]
OUT.mkdir(exist_ok=True)
P = lambda s: pd.Period(s, freq="Q")
NUM = {}

# phases: (key, label, base quarter, end quarter)
PH = [("W1", "2010Q1–2017Q2", "2009Q4", "2017Q2"), ("W2", "2017Q3–2019Q4", "2017Q2", "2019Q4"),
      ("W3", "2020Q1–2021Q2", "2019Q4", "2021Q2"), ("Cut", "Rate-cut 2021Q3–2023Q2", "2021Q2", "2023Q2"),
      ("Tight", "Tightening 2023Q3–2024Q2", "2023Q2", "2024Q2"), ("NRIR", "NRIR 2021Q3–2024Q2", "2021Q2", "2024Q2"),
      ("Norm", "Normalization 2024Q3–2026Q1", "2024Q2", "2026Q1")]

# ------------------------------------------------------------------ 1. EVDS stocks and rates (15+, SA)
m = pd.read_csv(DATA / "evds_hia_monthly.csv")
m["month"] = pd.PeriodIndex(m["month"], freq="M")
m = m.set_index("month")
q = m.groupby(m.index.asfreq("Q")).mean()
q = q[m.groupby(m.index.asfreq("Q")).size() == 3]  # complete quarters only
q.index.name = "period"
q.to_csv(OUT / "evds_quarterly.csv", encoding="utf-8")
ann = m.groupby(m.index.year).mean()[m.groupby(m.index.year).size() == 12]
ann.to_csv(OUT / "evds_annual.csv", encoding="utf-8")

rows = []
for k, lab, b, e in PH:
    b, e = P(b), P(e)
    if b not in q.index or e not in q.index:
        continue
    r = {"phase": k, "label": lab, "base": str(b), "end": str(e)}
    for v in ["pop", "lf", "e", "u", "n"]:
        r[f"d_{v}"] = q.loc[e, f"{v}_sa"] - q.loc[b, f"{v}_sa"]
    for v in ["lfpr", "er", "ur"]:
        r[f"{v}_b"], r[f"{v}_e"] = q.loc[b, f"{v}_sa"], q.loc[e, f"{v}_sa"]
    # exact rates from stocks (SA rates are published to 1 decimal)
    LFb, LFe, Eb, Ee, POb, POe = (q.loc[b, "lf_sa"], q.loc[e, "lf_sa"], q.loc[b, "e_sa"], q.loc[e, "e_sa"],
                                  q.loc[b, "pop_sa"], q.loc[e, "pop_sa"])
    lfpr_b, lfpr_e, er_b, er_e = LFb / POb, LFe / POe, Eb / POb, Ee / POe
    ub, ue = 1 - er_b / lfpr_b, 1 - er_e / lfpr_e
    # decomposition of the change in u: ln(1-u) = ln ER - ln LFPR
    ubar = (ub + ue) / 2
    c_er = -(1 - ubar) * np.log(er_e / er_b) * 100
    c_lfpr = (1 - ubar) * np.log(lfpr_e / lfpr_b) * 100
    r.update(du=(ue - ub) * 100, du_from_er=c_er, du_from_lfpr=c_lfpr, du_resid=(ue - ub) * 100 - c_er - c_lfpr)
    # labour-force change: population growth at constant participation vs participation change
    r["dlf_pop"] = (POe - POb) * lfpr_b
    r["dlf_part"] = POe * (lfpr_e - lfpr_b)
    r["years"] = (e - b).n / 4
    rows.append(r)
dec = pd.DataFrame(rows)
dec.to_csv(OUT / "t1_stocks_rates_decomposition.csv", index=False, encoding="utf-8")
NUM["decomposition"] = dec.round(3).to_dict(orient="records")

# ------------------------------------------------------------------ 2. TurkStat annual underutilization (15+)
# Typed from TurkStat annual bulletins "Isgucu Istatistikleri, YYYY" (as reproduced in the sources below).
TUIK = pd.DataFrame([
    # year, U, E, LF (thousand), u, ER, LFPR, LU2, LU3, LU4 (percent)
    (2021, 3919, 28797, 32716, 12.0, 45.2, 51.4, 16.8, 20.0, 24.4),
    (2022, 3582, 30752, 34334, 10.4, 47.5, 53.1, 14.6, 17.4, 21.3),
    (2023, 3264, 31632, 34896, 9.4, 48.3, 53.3, 15.4, 17.3, 22.8),
    (2024, 3113, 32620, 35733, 8.7, 49.5, 54.2, 18.1, 18.4, 26.7),
    (2025, 2966, 32566, 35533, 8.3, 49.0, 53.5, 19.6, 19.9, 29.7),
], columns=["year", "U", "E", "LF", "u", "ER", "LFPR", "LU2", "LU3", "LU4"]).set_index("year")
SOURCES = {
    2021: "https://www.ekonomist.com.tr/haberler/son-dakika-2021-issizlik-rakami-aciklandi-30703",
    2022: "https://www.alomaliye.com/2023/03/23/2022-yili-issizlik-orani-yuzde-10-4-seviyesinde-gerceklesti/",
    2023: "https://www.alomaliye.com/2024/03/25/2023-yili-issizlik-orani/",
    2024: "https://www.alomaliye.com/2025/03/20/isgucu-istatistikleri-2024/",
    2025: "https://habersefi.com/isgucu-istatistikleri-2025/",
}
t = TUIK.copy()
t["TRU"] = (t.LU2 - t.u) / 100 * t.LF                       # LU2 = (U + TRU) / LF
t["PLF"] = (t.LU3 / 100 * t.LF - t.U) / (1 - t.LU3 / 100)   # LU3 = (U + PLF) / (LF + PLF)
t["ELF"] = t.LF + t.PLF
t["LU4_check"] = (t.U + t.TRU + t.PLF) / t.ELF * 100
t["gap"] = t.LU4 - t.u
t["gap_TRU"] = t.TRU / t.ELF * 100                           # gap = TRU/ELF + (1-u) PLF/ELF
t["gap_PLF"] = (1 - t.u / 100) * t.PLF / t.ELF * 100
t["gap_check"] = t.gap_TRU + t.gap_PLF
t["source"] = [SOURCES[y] for y in t.index]
t.round(2).to_csv(OUT / "t2_tuik_underutilization_annual.csv", encoding="utf-8")
d = t[["U", "TRU", "PLF", "E", "LF"]].diff()
d.round(0).to_csv(OUT / "t2b_tuik_components_changes.csv", encoding="utf-8")
NUM["tuik_annual"] = t.drop(columns="source").round(2).reset_index().to_dict(orient="records")
NUM["tuik_annual_changes"] = d.round(0).reset_index().to_dict(orient="records")
# check EVDS annual averages (NSA) against the bulletins
chk = ann.loc[[y for y in TUIK.index if y in ann.index], ["u_nsa", "e_nsa", "lf_nsa", "ur_nsa", "er_nsa", "lfpr_nsa"]].round(1)
NUM["evds_vs_bulletin"] = chk.reset_index().to_dict(orient="records")

# ------------------------------------------------------------------ 3. Eurostat: slack, flows, hours, sectors
def es(name):
    df, lab = read(DATA / "eurostat" / f"{name}.json")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df, lab


def qser(df, **f):
    s = df
    for k, v in f.items():
        s = s[s[k] == v]
    s = s.set_index("time")["value"].sort_index()
    s.index = pd.PeriodIndex([x.replace("-", "") for x in s.index], freq="Q")
    return s


sla, _ = es("lfsi_sla_q")
slack = pd.DataFrame({w: qser(sla, s_adj="SA", sex="T", age="Y15-74", unit="THS_PER", wstatus=w)
                      for w in ["UNE", "UEMP_PT", "NSEEK_AVL", "SEEK_NAVL", "SLACK"]})
slack["PLF"] = slack.NSEEK_AVL + slack.SEEK_NAVL
slack["SLACK_pc"] = qser(sla, s_adj="SA", sex="T", age="Y15-74", unit="PC_ELF", wstatus="SLACK")
for sx in ["M", "F"]:
    slack[f"NSEEK_AVL_{sx}"] = qser(sla, s_adj="SA", sex=sx, age="Y15-74", unit="THS_PER", wstatus="NSEEK_AVL")
    slack[f"UEMP_PT_{sx}"] = qser(sla, s_adj="SA", sex=sx, age="Y15-74", unit="THS_PER", wstatus="UEMP_PT")
slack["flag_break"] = sla[(sla.s_adj == "SA") & (sla.sex == "T") & (sla.age == "Y15-74") & (sla.unit == "THS_PER")
                          & (sla.wstatus == "SLACK")].set_index("time")["flag"].reindex(
    [str(p)[:4] + "-Q" + str(p)[-1] for p in slack.index]).values
slack.index.name = "period"
em0, _ = es("lfsi_emp_q")
slack["LF"] = qser(em0, s_adj="SA", sex="T", age="Y15-74", unit="THS_PER", indic_em="ACT")
slack["ur"] = slack.UNE / slack.LF * 100
slack["slack_rate"] = slack.SLACK / (slack.LF + slack.PLF) * 100   # EU analogue of LU4
slack["gap"] = slack.slack_rate - slack.ur
slack["gap_UEMP_PT"] = slack.UEMP_PT / (slack.LF + slack.PLF) * 100
slack["gap_PLF"] = (1 - slack.ur / 100) * slack.PLF / (slack.LF + slack.PLF) * 100
slack.to_csv(OUT / "t3_eurostat_slack_sa.csv", encoding="utf-8")
NUM["eurostat_gap_quarterly"] = slack.loc[P("2019Q4"):, ["ur", "slack_rate", "gap", "gap_UEMP_PT", "gap_PLF"]].round(2).reset_index().astype({"period": str}).to_dict(orient="records")
ya = slack.groupby(slack.index.year)[["UNE", "UEMP_PT", "NSEEK_AVL", "SEEK_NAVL", "PLF"]].mean()
NUM["eurostat_slack_annual"] = ya.loc[2019:].round(0).reset_index().to_dict(orient="records")

lng, _ = es("lfsi_long_q")
F = {}
for w, unit in [("EMP_UNE", "PC_EMP"), ("EMP_INAC", "PC_EMP"), ("UNE_EMP", "PC_UNE"), ("UNE_INAC", "PC_UNE"),
                ("INAC_EMP", "PC_INAC"), ("INAC_UNE", "PC_INAC"), ("EMP_EMP", "PC_EMP"), ("UNE_UNE", "PC_UNE"),
                ("INAC_INAC", "PC_INAC")]:
    F[w] = qser(lng, s_adj="SA", sex="T", unit=unit, wstatus=w)
flows = pd.DataFrame(F)
# the transition series has no 2021Q1 (questionnaire break); reindex so rolling windows never bridge the gap
flows = flows.reindex(pd.period_range(flows.index.min(), flows.index.max(), freq="Q"))
flows.index.name = "period"
flows.to_csv(OUT / "t4_eurostat_flows_sa.csv", encoding="utf-8")


KEYS0 = ["EMP_UNE", "EMP_INAC", "UNE_EMP", "UNE_INAC", "INAC_EMP", "INAC_UNE"]


def steady(p):
    """Stationary shares of a 3-state chain (E, U, N) from quarterly transition rates in percent."""
    eu, en, ue, un, ne, nu = (p[k] / 100 for k in ["EMP_UNE", "EMP_INAC", "UNE_EMP", "UNE_INAC", "INAC_EMP", "INAC_UNE"])
    M = np.array([[1 - eu - en, eu, en], [ue, 1 - ue - un, un], [ne, nu, 1 - ne - nu]])
    w, v = np.linalg.eig(M.T)
    s = np.real(v[:, np.argmin(abs(w - 1))])
    s = s / s.sum()
    return s[1] / (s[0] + s[1]) * 100, (s[0] + s[1]) * 100


fl4 = flows.rolling(4).mean()   # NaN where the window touches the missing 2021Q1
ss = pd.DataFrame([steady(fl4.loc[i]) if fl4.loc[i].notna().all() else (np.nan, np.nan) for i in fl4.index],
                  index=fl4.index, columns=["u_star", "lfpr_star"])
ss["u_star_q"] = [steady(flows.loc[i])[0] if flows.loc[i, KEYS0].notna().all() else np.nan for i in flows.index]
ss.to_csv(OUT / "t4b_steady_state.csv", encoding="utf-8")
KEYS = KEYS0
frows = []
# phase endpoints use single-quarter seasonally adjusted rates, as Table 1 does for stocks
for k, lab, b, e in PH:
    b, e = P(b), P(e)
    if b not in flows.index or e not in flows.index or flows.loc[b, KEYS].isna().any():
        continue
    pb, pe = flows.loc[b, KEYS], flows.loc[e, KEYS]
    r = {"phase": k, "u_star_b": steady(pb)[0], "u_star_e": steady(pe)[0]}
    r["du_star"] = r["u_star_e"] - r["u_star_b"]
    for key in KEYS:  # change one flow at a time, averaged over the two orders
        p1 = pb.copy(); p1[key] = pe[key]
        p2 = pe.copy(); p2[key] = pb[key]
        r[f"c_{key}"] = 0.5 * ((steady(p1)[0] - steady(pb)[0]) + (steady(pe)[0] - steady(p2)[0]))
        r[f"{key}_b"], r[f"{key}_e"] = pb[key], pe[key]
    frows.append(r)
fdec = pd.DataFrame(frows)
fdec.to_csv(OUT / "t4c_flow_decomposition.csv", index=False, encoding="utf-8")
NUM["flow_decomposition"] = fdec.round(3).to_dict(orient="records")

# hours: average actual weekly hours, main job, employed persons 15+, NSA -> 4-quarter averages
wh, _ = es("lfsq_ewhais")
hrs = pd.DataFrame({
    "h_emp": qser(wh, isco08="TOTAL", wstatus="EMP", worktime="TOTAL", age="Y_GE15", sex="T", unit="HR"),
    "h_sal": qser(wh, isco08="TOTAL", wstatus="SAL", worktime="TOTAL", age="Y_GE15", sex="T", unit="HR"),
    "h_ft": qser(wh, isco08="TOTAL", wstatus="EMP", worktime="FT", age="Y_GE15", sex="T", unit="HR"),
})
eg, _ = es("lfsq_egan2")
nace = {c: qser(eg, unit="THS_PER", sex="T", age="Y_GE15", nace_r2=c) for c in
        ["TOTAL", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U"]}
nace = pd.DataFrame(nace)
hrs["emp"] = nace["TOTAL"]
hrs["total_hours"] = hrs.h_emp * hrs.emp
na = pd.read_csv(V2 / "data" / "evds_na_q.csv")
na["period"] = pd.PeriodIndex(na["period"], freq="Q")
na = na.set_index("period")
hrs["gdp_cv"] = na["y_cv"]
h4 = hrs.rolling(4).mean()
h4["gdp_cv"] = hrs["gdp_cv"].rolling(4).sum()
h4["prod_worker"] = h4.gdp_cv / h4.emp
h4["prod_hour"] = h4.gdp_cv / h4.total_hours
h4.index.name = "period"
h4.to_csv(OUT / "t5_hours_productivity_4q.csv", encoding="utf-8")
hrows = []
for k, lab, b, e in PH:
    b, e = P(b), P(e)
    if b not in h4.index or e not in h4.index or pd.isna(h4.loc[b, "h_emp"]):
        continue
    yrs = (e - b).n / 4
    r = {"phase": k}
    for v in ["emp", "h_emp", "total_hours", "gdp_cv", "prod_worker", "prod_hour"]:
        r[f"g_{v}"] = (np.log(h4.loc[e, v]) - np.log(h4.loc[b, v])) * 100 / yrs if pd.notna(h4.loc[e, v]) else np.nan
    r["h_emp_b"], r["h_emp_e"] = h4.loc[b, "h_emp"], h4.loc[e, "h_emp"]
    r["h_ft_b"], r["h_ft_e"] = h4.loc[b, "h_ft"], h4.loc[e, "h_ft"]
    hrows.append(r)
hdec = pd.DataFrame(hrows)
hdec.to_csv(OUT / "t5b_hours_by_phase.csv", index=False, encoding="utf-8")
NUM["hours_by_phase"] = hdec.round(3).to_dict(orient="records")

# sectors: change in employment (thousand, 4-quarter averages) by phase
n4 = nace.rolling(4).mean()
GROUPS = {"Agriculture": ["A"], "Manufacturing": ["C"], "Other industry": ["B", "D", "E"], "Construction": ["F"],
          "Trade": ["G"], "Transport": ["H"], "Accommodation and food": ["I"],
          "Business services": ["J", "K", "L", "M", "N"], "Public, education, health": ["O", "P", "Q"],
          "Other services": ["R", "S", "T", "U"]}
srows = []
last_nace = n4["C"].last_valid_index()  # NACE detail ends earlier than the total
for k, lab, b, e in PH:
    b, e = P(b), min(P(e), last_nace)
    if b not in n4.index or e not in n4.index or pd.isna(n4.loc[b, "TOTAL"]):
        continue
    r = {"phase": k, "end_used": str(e), "TOTAL": n4.loc[e, "TOTAL"] - n4.loc[b, "TOTAL"]}
    for g, cs in GROUPS.items():
        r[g] = sum(n4.loc[e, c] - n4.loc[b, c] for c in cs if pd.notna(n4.loc[e, c]) and pd.notna(n4.loc[b, c]))
    srows.append(r)
sec = pd.DataFrame(srows)
sec.to_csv(OUT / "t6_sector_employment_changes.csv", index=False, encoding="utf-8")
NUM["sector_changes"] = sec.round(0).to_dict(orient="records")

# status in employment and part-time (4-quarter averages, 15+)
ft, _ = es("lfsq_eftpt")
st = pd.DataFrame({w: qser(ft, unit="THS_PER", sex="T", age="Y_GE15", wstatus=w, worktime="TOTAL")
                   for w in ["EMP", "SAL", "SELF_NS", "SELF_S", "CFAM"]})
st["PT"] = qser(ft, unit="THS_PER", sex="T", age="Y_GE15", wstatus="EMP", worktime="PT")
st4 = st.rolling(4).mean()
st4.index.name = "period"
st4.to_csv(OUT / "t7_status_parttime_4q.csv", encoding="utf-8")
strows = []
for k, lab, b, e in PH:
    b, e = P(b), P(e)
    if b not in st4.index or e not in st4.index or pd.isna(st4.loc[b, "EMP"]):
        continue
    r = {"phase": k}
    for v in st4.columns:
        r[f"d_{v}"] = st4.loc[e, v] - st4.loc[b, v]
    r["pt_share_b"], r["pt_share_e"] = st4.loc[b, "PT"] / st4.loc[b, "EMP"] * 100, st4.loc[e, "PT"] / st4.loc[e, "EMP"] * 100
    strows.append(r)
NUM["status_changes"] = pd.DataFrame(strows).round(1).to_dict(orient="records")

# participation by sex (SA, 15-74, % of population)
em, _ = es("lfsi_emp_q")
sexp = pd.DataFrame({f"act_{sx}": qser(em, s_adj="SA", sex=sx, age="Y15-74", unit="PC_POP", indic_em="ACT") for sx in "TMF"})
for sx in "TMF":
    sexp[f"emp_{sx}"] = qser(em, s_adj="SA", sex=sx, age="Y15-74", unit="PC_POP", indic_em="EMP_LFS")
sexp.index.name = "period"
sexp.to_csv(OUT / "t8_participation_by_sex_sa.csv", encoding="utf-8")
NUM["participation_by_sex"] = {str(i): sexp.loc[P(i)].round(1).to_dict() for i in
                               ["2019Q4", "2021Q2", "2023Q2", "2024Q2", "2026Q1"] if P(i) in sexp.index}

# inactivity by reason (annual): share of the inactive who want to work, by reason
ig, lab_ig = es("lfsa_igar")
igr = ig[(ig.unit == "PC_POP") & (ig.sex == "T") & (ig.age == "Y15-74")].pivot(index="time", columns="reason", values="value")
igr.to_csv(OUT / "t9_inactive_wanting_work_by_reason_pc_pop.csv", encoding="utf-8")
NUM["inactive_reason_pc_pop"] = igr.loc[[y for y in igr.index if int(y) >= 2019]].round(2).reset_index().to_dict(orient="records")
NUM["inactive_reason_labels"] = lab_ig["reason"]

# unemployment duration: long-term share (12 months or more)
ud, _ = es("lfsq_ugad")
dur = pd.DataFrame({c: qser(ud, unit="THS_PER", sex="T", age="Y15-74", duration=c) for c in
                    ["TOTAL", "M12-17", "M18-23", "M24-47", "M_GE48"]})
dur["lt_share"] = dur[["M12-17", "M18-23", "M24-47", "M_GE48"]].sum(axis=1) / dur.TOTAL * 100
dur4 = dur.rolling(4).mean()
NUM["long_term_unemployment_share_4q"] = {str(i): round(dur4.loc[P(i), "lt_share"], 1) for i in
                                          ["2019Q4", "2021Q2", "2023Q2", "2024Q2", "2026Q1"] if P(i) in dur4.index}

# latest monthly values (EVDS, SA)
NUM["latest_month"] = {"month": str(m.index[-1]), **m.iloc[-1][["ur_sa", "lfpr_sa", "er_sa", "e_sa", "u_sa", "lf_sa"]].to_dict()}
(OUT / "numbers.json").write_text(json.dumps(NUM, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print(dec.round(2).to_string())
print(t.drop(columns="source").round(1).to_string())
print(chk.to_string())
print(slack.loc[P("2019Q1"):].round(0).to_string())
print(fdec.round(2).T.to_string())
print(hdec.round(2).to_string())
print(sec.round(0).to_string())
print(pd.DataFrame(strows).round(1).to_string())
print(sexp.loc[P("2019Q1"):].round(1).iloc[::2].to_string())
print(igr.loc[[y for y in igr.index if int(y) >= 2017]].round(2).to_string())
print(dur4.loc[P("2019Q1"):, "lt_share"].round(1).to_string())
print(ss.loc[P("2019Q1"):].round(2).to_string())
