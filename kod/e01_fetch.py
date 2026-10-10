# -*- coding: utf-8 -*-
"""e01_fetch.py - labour market data for the employment and underutilization note (Essay 3 V2).

Sources
  CBRT EVDS (TurkStat household labour force survey, 15+, monthly):
      bie_tisguc  TP.TIG01-08  seasonally adjusted
      bie_yisgucu2 TP.YISGUCU2.G1-G8  not seasonally adjusted
  Eurostat (TurkStat LFS transmitted to Eurostat, EU concepts, mostly 15-74), geo=TR:
      lfsi_sla_q  labour market slack (unemployed, underemployed part-time, available not seeking,
                  seeking not available), NSA/SA
      lfsi_long_q quarterly transitions between employment, unemployment and inactivity, NSA/SA
      lfsi_emp_q, une_rt_q, lfsq_pganws, lfsq_egan2, lfsq_eftpt, lfsq_ewhais, lfsq_eppga,
      lfsq_ugad, lfsa_igar (annual, inactivity by reason)
The EVDS key is read from ESSAY 3/.env and is never printed or written.
TurkStat's own annual underutilization rates (LU2, LU3, LU4) are typed in e02 from the annual
bulletins (sources listed there); TurkStat's portal blocks automated downloads.
"""
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from evds import evdsAPI

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
E3 = HERE.parents[2]
EUROSTAT = ["lfsi_sla_q", "lfsi_long_q", "lfsi_emp_q", "une_rt_q", "lfsq_pganws", "lfsq_egan2", "lfsq_eftpt",
            "lfsq_ewhais", "lfsq_eppga", "lfsq_ugad", "lfsa_igar"]
EVDS = {"pop_sa": "TP.TIG01", "lf_sa": "TP.TIG02", "e_sa": "TP.TIG03", "u_sa": "TP.TIG04", "n_sa": "TP.TIG05",
        "lfpr_sa": "TP.TIG06", "er_sa": "TP.TIG07", "ur_sa": "TP.TIG08",
        "pop_nsa": "TP.YISGUCU2.G1", "lf_nsa": "TP.YISGUCU2.G2", "e_nsa": "TP.YISGUCU2.G3", "u_nsa": "TP.YISGUCU2.G4",
        "n_nsa": "TP.YISGUCU2.G5", "lfpr_nsa": "TP.YISGUCU2.G6", "er_nsa": "TP.YISGUCU2.G7", "ur_nsa": "TP.YISGUCU2.G8"}


def load_key():
    for line in (E3 / ".env").read_text(encoding="utf-8-sig").splitlines():
        if line.strip().startswith("EVDS_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("EVDS_KEY not found")


def fetch_evds(log):
    ev = evdsAPI(load_key())
    df = None
    for attempt in range(3):
        try:
            df = ev.get_data(list(EVDS.values()), startdate="01-01-2005", enddate=date.today().strftime("%d-%m-%Y"))
            break
        except Exception as e:
            log.append(f"EVDS retry {attempt + 1}: {type(e).__name__}")
            time.sleep(4)
    if df is None:
        sys.exit("EVDS fetch failed")
    out = pd.DataFrame({"month": pd.PeriodIndex(pd.to_datetime(df["Tarih"], format="%Y-%m"), freq="M")})
    for name, code in EVDS.items():
        out[name] = pd.to_numeric(df[code.replace(".", "_")], errors="coerce").values
    out = out.dropna(subset=["ur_sa", "ur_nsa"], how="all")
    out.to_csv(DATA / "evds_hia_monthly.csv", index=False, encoding="utf-8")
    log.append(f"EVDS: {len(out)} months, {out.month.min()} to {out.month.max()}")


def fetch_eurostat(log):
    d = DATA / "eurostat"
    d.mkdir(exist_ok=True)
    base = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{}?geo=TR&lang=EN"
    for ds in EUROSTAT:
        for attempt in range(6):
            r = requests.get(base.format(ds), timeout=300)
            if r.status_code == 200:
                (d / f"{ds}.json").write_bytes(r.content)
                log.append(f"Eurostat {ds}: {len(r.content)} bytes")
                break
            log.append(f"Eurostat {ds}: HTTP {r.status_code}, retry")
            time.sleep(20)


if __name__ == "__main__":
    log = [f"retrieved {date.today().isoformat()}"]
    fetch_evds(log)
    fetch_eurostat(log)
    (DATA / "fetch_log.txt").write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))
