# -*- coding: utf-8 -*-
"""e06_report_figs.py - figures for the PDF report and the GitHub showcase (Turkish labels).

Self-contained: reads the processed outputs in ../out (CSV and akim_data.json) and writes
../rapor/fig/R01..R12.{svg,png}. Also writes two helper tables used by the report:
out/t10_want_by_sex.csv (from the Eurostat JSON, if present) and out/t11_ustar_decomp.csv.
"""
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
if (HERE.parent / "out").exists():          # thesis folder layout
    OUT, FIG = HERE.parent / "out", HERE.parent / "rapor" / "fig"
else:                                       # GitHub repository layout: kod/, veri/, figurler/
    OUT, FIG = HERE.parent / "veri", HERE.parent / "figurler"
FIG.mkdir(parents=True, exist_ok=True)

C_E, C_U, C_N = "#008c84", "#cf5a2e", "#8256ab"
INK, GREY, BAND = "#1c1e21", "#6b7076", "#e4e1d8"
mpl.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.7, "axes.grid": True, "grid.color": "#e3e1dc", "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "svg.fonttype": "none", "svg.hashsalt": "rapor", "figure.dpi": 100, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})
W = 6.6  # inches, text width of an A4 page with 2.2 cm margins
P = lambda s: pd.Period(s, freq="Q")
EP0, EP1, TURN = P("2021Q3"), P("2024Q2"), P("2023Q2")


def tr(x, nd=1):
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


def save(fig, name):
    fig.savefig(FIG / f"{name}.svg")
    fig.savefig(FIG / f"{name}.png", dpi=220)
    plt.close(fig)


def shade(ax):
    ax.axvspan(EP0.start_time, EP1.end_time, color=BAND, lw=0, zorder=0)
    ax.axvline(TURN.end_time, color=GREY, lw=0.7, ls=(0, (2, 2)), zorder=1)


def years(ax, step=2):
    ax.xaxis.set_major_locator(mpl.dates.YearLocator(step))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))


def qcsv(name):
    d = pd.read_csv(OUT / name)
    d["period"] = pd.PeriodIndex(d["period"], freq="Q")
    return d.set_index("period")


def ts(s):
    return s.index.to_timestamp()


AK = json.loads((OUT / "akim_data.json").read_text(encoding="utf-8"))
QS = AK["quarters"]
BASE = AK["base"]
A = pd.DataFrame([{k: v for k, v in q.items() if k not in ("f", "r")} for q in QS])
A["period"] = pd.PeriodIndex(A["q"], freq="Q")
A = A.set_index("period")
for k in ["EU", "EN", "UE", "UN", "NE", "NU"]:
    A["f_" + k] = [q["f"][k] if q["f"] else np.nan for q in QS]
    A["r_" + k] = [q["r"][k] if q["r"] else np.nan for q in QS]

# ---------------------------------------------------------------- R01 unemployment vs underutilization
s = qcsv("t3_eurostat_slack_sa.csv").loc[P("2014Q1"):]
t = pd.read_csv(OUT / "t2_tuik_underutilization_annual.csv", index_col=0)
fig, axs = plt.subplots(1, 2, figsize=(W, W * 0.42))
ax = axs[0]
shade(ax)
ax.plot(ts(s), s["slack_rate"], color=C_N, lw=1.6, label="Atıl işgücü oranı (AB)")
ax.plot(ts(s), s["ur"], color=C_U, lw=1.6, label="İşsizlik oranı")
mid = [pd.Timestamp(f"{y}-07-01") for y in t.index]
ax.plot(mid, t["LU4"], ls="none", marker="o", ms=4, color=C_N, mfc="white", mew=1.2, label="TÜİK atıl işgücü (yıllık)")
ax.plot(mid, t["u"], ls="none", marker="s", ms=3.5, color=C_U, mfc="white", mew=1.2, label="TÜİK işsizlik (yıllık)")
ax.set_ylim(0, 34)
ax.set_ylabel("%")
years(ax)
ax.legend(loc="upper left", ncol=2, fontsize=7.5, columnspacing=0.8, handlelength=1.4)
ax.set_title("(a) Oranlar")
ax = axs[1]
yrs = [str(y) for y in t.index]
x = np.arange(len(yrs))
b1, b2, b3 = t["U"] / 1000, t["TRU"] / 1000, t["PLF"] / 1000
ax.bar(x, b1, 0.62, color=C_U, label="İşsiz")
ax.bar(x, b2, 0.62, bottom=b1, color="#e3a183", label="Zamana bağlı eksik istihdam")
ax.bar(x, b3, 0.62, bottom=b1 + b2, color="#c3afd8", label="Potansiyel işgücü")
for i, v in enumerate(b1 + b2 + b3):
    ax.text(i, v + 0.2, tr(v), ha="center", fontsize=8.5)
ax.set_xticks(x, yrs)
ax.set_ylim(0, 14)
ax.set_ylabel("milyon kişi")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", fontsize=7.5)
ax.set_title("(b) TÜİK atıl işgücünün bileşenleri")
fig.tight_layout()
save(fig, "R01_issizlik_atil")

# ---------------------------------------------------------------- R02 gap components (EU, quarterly)
s2 = s.loc[P("2021Q2"):]
fig, ax = plt.subplots(figsize=(W, W * 0.36))
shade(ax)
ax.stackplot(ts(s2), s2["gap_UEMP_PT"], s2["gap_PLF"], colors=["#e3a183", "#c3afd8"],
             labels=["Eksik istihdamdan (kısmi süreli)", "Potansiyel işgücünden"], lw=0)
ax.plot(ts(s2), s2["gap"], color=INK, lw=1.4, label="Fark: atıl işgücü − işsizlik oranı")
ax.set_ylabel("puan")
ax.set_ylim(0, 19)
years(ax, 1)
ax.legend(loc="upper left", fontsize=8)
ax.annotate(f"2022Q4: {tr(s2.loc[P('2022Q4'),'gap'])}", (P("2022Q4").to_timestamp(), s2.loc[P("2022Q4"), "gap"]),
            xytext=(0, -26), textcoords="offset points", ha="center", fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
ax.annotate(f"2026Q2: {tr(s2['gap'].iloc[-1])}", (ts(s2)[-1], s2["gap"].iloc[-1]), xytext=(-8, 8),
            textcoords="offset points", ha="right", fontsize=8.5)
fig.tight_layout()
save(fig, "R02_fark_bilesenleri")

# ---------------------------------------------------------------- R03 decomposition of the change in u
d = pd.read_csv(OUT / "t1_stocks_rates_decomposition.csv").set_index("phase")
PH = [("W2", "Kredi garantisi\nve kur krizi\n2017Q3–2019Q4"), ("Cut", "İndirim evresi\n2021Q3–2023Q2"),
      ("Tight", "Sıkılaştırma\n2023Q3–2024Q2"), ("Norm", "Normalleşme\n2024Q3–2026Q1")]
fig, ax = plt.subplots(figsize=(W, W * 0.4))
x = np.arange(len(PH))
er = [d.loc[k, "du_from_er"] for k, _ in PH]
lf = [d.loc[k, "du_from_lfpr"] for k, _ in PH]
du = [d.loc[k, "du"] for k, _ in PH]
ax.bar(x - 0.17, er, 0.32, color=C_E, label="İstihdam oranının katkısı")
ax.bar(x + 0.17, lf, 0.32, color=C_N, label="Katılım oranının katkısı")
ax.scatter(x, du, marker="D", s=34, color=INK, zorder=3, label="İşsizlik oranındaki değişim")
for i, v in enumerate(du):
    ax.text(i, v + (0.35 if v >= 0 else -0.75), tr(v), ha="center", fontsize=8.5, fontweight="bold")
ax.axhline(0, color="#555", lw=0.7)
ax.set_xticks(x, [l for _, l in PH], fontsize=8.5)
ax.set_ylabel("puan")
ax.grid(axis="x", visible=False)
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
save(fig, "R03_issizlik_ayristirma")

# ---------------------------------------------------------------- R04 participation and wanting to work
wbs = None
try:
    sys.path.insert(0, str(HERE))
    from jsonstat import read
    em, _ = read(HERE.parent / "data" / "eurostat" / "lfsi_emp_q.json")
    sl, _ = read(HERE.parent / "data" / "eurostat" / "lfsi_sla_q.json")

    def ser(df, **f):
        for k, v in f.items():
            df = df[df[k] == v]
        q = df.set_index("time")["value"].astype(float).sort_index()
        q.index = pd.PeriodIndex([x.replace("-", "") for x in q.index], freq="Q")
        return q
    rows = {}
    for sx in "TFM":
        a = ser(em, s_adj="SA", sex=sx, age="Y15-74", unit="THS_PER", indic_em="ACT")
        ap = ser(em, s_adj="SA", sex=sx, age="Y15-74", unit="PC_POP", indic_em="ACT")
        pl = ser(sl, s_adj="SA", sex=sx, age="Y15-74", unit="THS_PER", wstatus="NSEEK_AVL") + \
            ser(sl, s_adj="SA", sex=sx, age="Y15-74", unit="THS_PER", wstatus="SEEK_NAVL")
        pop = a / (ap / 100)
        rows[f"lfpr_{sx}"] = a / pop * 100
        rows[f"want_{sx}"] = (a + pl) / pop * 100
    wbs = pd.DataFrame(rows).loc[P("2014Q1"):]
    wbs.index.name = "period"
    wbs.round(2).to_csv(OUT / "t10_want_by_sex.csv", encoding="utf-8")
except Exception:  # repository copy without the raw Eurostat files
    wbs = qcsv("t10_want_by_sex.csv")
fig, axs = plt.subplots(1, 2, figsize=(W, W * 0.42))
for ax, sx, ttl, lims in [(axs[0], "T", "(a) Toplam, 15–74 yaş", (48, 66)), (axs[1], "F", "(b) Kadınlar", (30, 50))]:
    shade(ax)
    ax.fill_between(ts(wbs), wbs[f"lfpr_{sx}"], wbs[f"want_{sx}"], color="#c3afd8", alpha=0.6, lw=0,
                    label="Çalışmaya hazır ama iş aramayan")
    ax.plot(ts(wbs), wbs[f"want_{sx}"], color=C_N, lw=1.6, label="Çalışmak isteyenler")
    ax.plot(ts(wbs), wbs[f"lfpr_{sx}"], color=C_E, lw=1.6, label="Katılım oranı (işgücü)")
    ax.set_ylim(*lims)
    ax.set_ylabel("nüfusun %'si")
    years(ax)
    ax.set_title(ttl)
axs[0].legend(loc="upper left", fontsize=7.5)
fig.tight_layout()
save(fig, "R04_katilim_isteyenler")

# ---------------------------------------------------------------- R05 flow diagrams (small multiples)
POS = {"E": (-1.0, 0.0), "N": (1.0, 0.0), "U": (0.0, 1.18)}
maxS = max(max(q["E"], q["N"]) for q in QS)
RM = 0.56
rad = lambda v: RM * np.sqrt(v / maxS)
maxF = max(max(q["f"].values()) for q in QS if q["f"])
FLW = [("EU", "E", "U"), ("EN", "E", "N"), ("UE", "U", "E"), ("UN", "U", "N"), ("NE", "N", "E"), ("NU", "N", "U")]
COL = {"E": C_E, "U": C_U, "N": C_N}
NAME = {"E": "İstihdam", "U": "İşsiz", "N": "İşgücü dışı"}


def flowpanel(ax, qlab):
    q = next(x for x in QS if x["q"] == qlab)
    R = {k: rad(q[k]) for k in "EUN"}
    inner = {"E": rad(q["PT"]), "N": rad(q["PLF"])}
    for k in "EUN":
        cx, cy = POS[k]
        ax.add_patch(Circle((cx, cy), R[k], facecolor=mpl.colors.to_rgba(COL[k], 0.13), edgecolor=COL[k], lw=1.1, zorder=1))
        if k == "U":
            ax.add_patch(Circle((cx, cy), R[k], facecolor="none", edgecolor=COL[k], hatch="////", lw=0, zorder=2))
            ax.text(cx, cy + R[k] + 0.17, NAME[k], ha="center", fontsize=8.5, fontweight="bold")
            ax.text(cx, cy + R[k] + 0.05, f"{tr(q[k] / 1000)} m", ha="center", fontsize=8)
        else:
            ri = inner[k]
            ax.add_patch(Circle((cx, cy - R[k] + ri + 0.02), ri, facecolor="none", edgecolor=COL[k], hatch="////",
                                lw=0.6, ls="--", zorder=2))
            ax.text(cx, cy + 0.12, NAME[k], ha="center", fontsize=8.5, fontweight="bold")
            ax.text(cx, cy - 0.02, f"{tr(q[k] / 1000)} m", ha="center", fontsize=8)
    for key, a, b in FLW:
        v = q["f"][key]
        lw = 0.6 + v / maxF * 7.5
        (ax_, ay_), (bx_, by_) = POS[a], POS[b]
        dx, dy = bx_ - ax_, by_ - ay_
        dist = np.hypot(dx, dy)
        th = np.arctan2(dy, dx)
        dl = 0.33
        # start and end points on the right-hand side of the direction of travel, the side arc3 bends to
        sx, sy = ax_ + (R[a] + 0.03) * np.cos(th - dl), ay_ + (R[a] + 0.03) * np.sin(th - dl)
        ex, ey = bx_ + (R[b] + 0.04) * np.cos(th + np.pi + dl), by_ + (R[b] + 0.04) * np.sin(th + np.pi + dl)
        rad_c = 0.32
        arr = FancyArrowPatch((sx, sy), (ex, ey), connectionstyle=f"arc3,rad={rad_c}", color=COL[b], lw=lw,
                              arrowstyle=f"-|>,head_length={0.35 + lw * 0.06},head_width={0.22 + lw * 0.045}",
                              mutation_scale=10, zorder=3, alpha=0.9, shrinkA=0, shrinkB=0)
        ax.add_patch(arr)
        # label near the start of the arc (arc3: control point offset from the chord midpoint)
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        cx_, cy_ = mx + rad_c * (ey - sy), my - rad_c * (ex - sx)
        tl = 0.36
        px = (1 - tl) ** 2 * sx + 2 * tl * (1 - tl) * cx_ + tl ** 2 * ex
        py = (1 - tl) ** 2 * sy + 2 * tl * (1 - tl) * cy_ + tl ** 2 * ey
        nx, ny = (ey - sy), -(ex - sx)
        nn = np.hypot(nx, ny)
        off = 0.07 + lw * 0.006
        ax.text(px + nx / nn * off, py + ny / nn * off, f"{v:,}".replace(",", "."), ha="center", va="center",
                fontsize=7.2, color=INK, zorder=4,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-0.75, 1.62)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(qlab, loc="center", fontsize=10.5)


fig, axs = plt.subplots(2, 2, figsize=(W, W * 0.84))
for ax, ql in zip(axs.ravel(), ["2021Q2", "2022Q4", "2024Q2", "2026Q2"]):
    flowpanel(ax, ql)
fig.tight_layout(h_pad=0.4, w_pad=0.2)
save(fig, "R05_akim_diyagramlari")

# ---------------------------------------------------------------- R06 heatmap of the six transition rates
LAB = {"EU": "İstihdam → işsiz", "EN": "İstihdam → işgücü dışı", "UE": "İşsiz → istihdam", "UN": "İşsiz → işgücü dışı",
       "NE": "İşgücü dışı → istihdam", "NU": "İşgücü dışı → işsiz"}
keys = ["EU", "EN", "UE", "UN", "NE", "NU"]
M = np.array([[(A.loc[i, "r_" + k] / BASE[k] - 1) if pd.notna(A.loc[i, "r_" + k]) else np.nan for i in A.index] for k in keys])
fig, ax = plt.subplots(figsize=(W, W * 0.33))
cmap = mpl.colors.LinearSegmentedColormap.from_list("dv", ["#3f6fae", "#e9e6de", "#b8741a"])
cmap.set_bad("#ffffff")
im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, vmin=-0.5, vmax=0.5, interpolation="nearest")
n = len(A)
for j in range(len(keys)):
    ax.add_patch(mpl.patches.Rectangle((list(A.index).index(P("2021Q1")) - 0.5, j - 0.5), 1, 1, fill=False, hatch="////",
                                       edgecolor="#999999", lw=0))
i0, i1, it = list(A.index).index(EP0), list(A.index).index(EP1), list(A.index).index(TURN)
ax.add_patch(mpl.patches.Rectangle((i0 - 0.5, -0.5), i1 - i0 + 1, len(keys), fill=False, edgecolor=INK, lw=1.0))
ax.axvline(it + 0.5, color=INK, lw=0.7, ls=(0, (2, 2)))
ax.set_yticks(range(len(keys)), [LAB[k] for k in keys], fontsize=8.5)
xt = [i for i, p in enumerate(A.index) if p.quarter == 1]
ax.set_xticks(xt, [str(A.index[i].year) for i in xt], fontsize=8.5)
ax.grid(False)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, ticks=[-0.5, 0, 0.5])
cb.ax.set_yticklabels(["−%50", "0", "+%50"], fontsize=8)
cb.outline.set_visible(False)
fig.tight_layout()
save(fig, "R06_akim_isi_haritasi")

# ---------------------------------------------------------------- R07 where do separations and unemployment exits go
f4 = A[["f_EU", "f_EN", "f_UE", "f_UN", "f_NE", "f_NU"]].reindex(pd.period_range("2014Q1", "2026Q2", freq="Q")).rolling(4).mean()
sep_u = f4.f_EU / (f4.f_EU + f4.f_EN) * 100
ue_n = f4.f_UN / (f4.f_UE + f4.f_UN) * 100
ne_u = f4.f_NU / (f4.f_NE + f4.f_NU) * 100
fig, ax = plt.subplots(figsize=(W, W * 0.38))
shade(ax)
ax.plot(ts(sep_u), sep_u, color=C_U, lw=1.6, label="İşten ayrılanların işsizliğe gidenlerin payı")
ax.plot(ts(ue_n), ue_n, color=C_N, lw=1.6, label="İşsizlikten çıkanların işgücü dışına gidenlerin payı")
ax.plot(ts(ne_u), ne_u, color=GREY, lw=1.4, ls="--", label="İşgücü dışından çıkanların iş aramaya başlayanların payı")
for ser_, c in [(sep_u, C_U), (ue_n, C_N), (ne_u, GREY)]:
    ax.text(ts(ser_)[-1], ser_.iloc[-1] + 1.2, "%" + tr(ser_.iloc[-1], 0), color=c, fontsize=8.5, ha="right")
ax.set_ylim(15, 62)
ax.set_ylabel("%, dört çeyrek ortalaması")
years(ax)
ax.legend(loc="upper left", fontsize=7.8)
fig.tight_layout()
save(fig, "R07_ayrilanlar_nereye")
pd.DataFrame({"sep_to_u": sep_u, "uexit_to_n": ue_n, "nexit_to_u": ne_u}).round(2).to_csv(OUT / "t12_destination_shares_4q.csv")

# ---------------------------------------------------------------- R08 steady-state vs actual unemployment
fig, ax = plt.subplots(figsize=(W, W * 0.36))
shade(ax)
ax.plot(ts(A["slack"]), A["slack"], color=C_N, lw=1.5, label="Atıl işgücü oranı (AB)")
ax.plot(ts(A["u"]), A["u"], color=C_U, lw=1.6, label="İşsizlik oranı")
ax.plot(ts(A["ustar"]), A["ustar"], color=INK, lw=1.4, ls=(0, (5, 3)), label="Durağan işsizlik oranı")
ax.set_ylim(0, 30)
ax.set_ylabel("%")
years(ax)
ax.legend(loc="upper left", fontsize=8, ncol=3)
fig.tight_layout()
save(fig, "R08_duragan_issizlik")


# ---------------------------------------------------------------- R09 sources of the fall in steady-state unemployment
def steady(p):
    eu, en, ue, un, ne, nu = (p[k] / 100 for k in keys)
    Mx = np.array([[1 - eu - en, eu, en], [ue, 1 - ue - un, un], [ne, nu, 1 - ne - nu]])
    wv, v = np.linalg.eig(Mx.T)
    st = np.real(v[:, np.argmin(abs(wv - 1))])
    st = st / st.sum()
    return st[1] / (st[0] + st[1]) * 100


R = A[["r_" + k for k in keys]].rename(columns=lambda c: c[2:]).reindex(pd.period_range("2014Q1", "2026Q2", freq="Q"))
R4 = R.rolling(4).mean()


def decomp(src, b, e):
    pb, pe = src.loc[P(b)], src.loc[P(e)]
    ub, ue = steady(pb), steady(pe)
    out = {"u_b": ub, "u_e": ue}
    for k in keys:
        p1 = pb.copy(); p1[k] = pe[k]
        p2 = pe.copy(); p2[k] = pb[k]
        out[k] = 0.5 * ((steady(p1) - ub) + (ue - steady(p2)))
    return out


dA = decomp(R4, "2022Q4", "2026Q2")
dB = decomp(R, "2021Q2", "2026Q2")
pd.DataFrame({"2023-2026 (4q, 2022Q4->2026Q2)": dA, "2021-2026 (tek ceyrek, 2021Q2->2026Q2)": dB}).round(3).to_csv(OUT / "t11_ustar_decomp.csv")
fig, ax = plt.subplots(figsize=(W, W * 0.42))
y = np.arange(len(keys))
ax.barh(y - 0.19, [dA[k] for k in keys], 0.36, color=C_U, label=f"2023–2026: {tr(dA['u_b'])} → {tr(dA['u_e'])}")
ax.barh(y + 0.19, [dB[k] for k in keys], 0.36, color="#e3a183", label=f"2021–2026: {tr(dB['u_b'])} → {tr(dB['u_e'])}")
for i, k in enumerate(keys):
    for off, dd in [(-0.19, dA), (0.19, dB)]:
        v = dd[k]
        ax.text(v + (0.04 if v >= 0 else -0.04), i + off, tr(v, 2), va="center", ha="left" if v >= 0 else "right", fontsize=7.8)
ax.axvline(0, color="#555", lw=0.7)
ax.set_yticks(y, [LAB[k] for k in keys], fontsize=8.5)
ax.invert_yaxis()
ax.set_xlabel("durağan işsizlik oranına katkı, puan")
ax.set_xlim(-1.95, 0.95)
ax.grid(axis="y", visible=False)
ax.legend(loc="lower left", fontsize=8, title="Durağan işsizlik oranı", title_fontsize=8)
fig.tight_layout()
save(fig, "R09_duragan_kaynaklar")

# ---------------------------------------------------------------- R10 hours and part-time
h = qcsv("t5_hours_productivity_4q.csv").loc[P("2015Q1"):]
stt = qcsv("t7_status_parttime_4q.csv").loc[P("2015Q1"):]
fig, axs = plt.subplots(1, 2, figsize=(W, W * 0.4))
ax = axs[0]
shade(ax)
ax.plot(ts(h), h["h_emp"], color=INK, lw=1.6, label="Tüm çalışanlar")
ax.plot(ts(h), h["h_ft"], color=C_E, lw=1.6, ls="--", label="Tam zamanlı")
ax.set_ylabel("saat")
ax.set_ylim(40, 51)
years(ax)
ax.legend(loc="upper right", fontsize=8)
ax.set_title("(a) Haftalık fiili çalışma süresi")
ax = axs[1]
shade(ax)
ax.plot(ts(stt), stt["PT"] / stt["EMP"] * 100, color=C_U, lw=1.6)
ax.set_ylabel("istihdamın %'si")
ax.set_ylim(8, 15.5)
years(ax)
ax.set_title("(b) Kısmi süreli çalışanların payı")
fig.tight_layout()
save(fig, "R10_saatler_kismi")

# ---------------------------------------------------------------- R11 output, employment and hours
fig, ax = plt.subplots(figsize=(W, W * 0.38))
shade(ax)
b = h.loc[P("2021Q2")]
for col, c, ls, lab in [("gdp_cv", INK, "-", "GSYH (zincirlenmiş hacim)"), ("emp", C_E, "--", "İstihdam"),
                        ("total_hours", C_U, "-.", "Toplam çalışılan saat")]:
    ax.plot(ts(h), h[col] / b[col] * 100, color=c, ls=ls, lw=1.6, label=lab)
ax.set_ylabel("2021Q2 = 100")
years(ax)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
save(fig, "R11_uretim_istihdam_saat")

# ---------------------------------------------------------------- R12 employment change by sector
sec = pd.read_csv(OUT / "t6_sector_employment_changes.csv").set_index("phase")
SL = [("Manufacturing", "İmalat"), ("Trade", "Ticaret"), ("Public, education, health", "Kamu, eğitim, sağlık"),
      ("Accommodation and food", "Konaklama-yiyecek"), ("Business services", "İş hizmetleri"), ("Transport", "Ulaştırma"),
      ("Construction", "İnşaat"), ("Other services", "Diğer hizmetler"), ("Agriculture", "Tarım")]
fig, ax = plt.subplots(figsize=(W, W * 0.45))
y = np.arange(len(SL))
for off, (ph, lab, c) in zip([-0.27, 0, 0.27], [("Cut", "İndirim evresi", C_E), ("Tight", "Sıkılaştırma", C_U),
                                              ("Norm", "Normalleşme (2025Q4'e kadar)", C_N)]):
    ax.barh(y + off, [sec.loc[ph, k] for k, _ in SL], 0.26, color=c, label=lab)
ax.axvline(0, color="#555", lw=0.7)
ax.set_yticks(y, [l for _, l in SL], fontsize=8.5)
ax.invert_yaxis()
ax.set_xlabel("istihdamdaki değişim, bin kişi (dört çeyrek ortalaması)")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
save(fig, "R12_sektorler")
print("report figures written:", sorted(p.name for p in FIG.glob("*.svg")))
print("ustar decomposition:", {k: round(v, 2) for k, v in dA.items()}, {k: round(v, 2) for k, v in dB.items()})
print("destination shares 2022Q4/2026Q2:", round(sep_u.loc[P("2022Q4")], 1), round(sep_u.iloc[-1], 1),
      round(ue_n.loc[P("2022Q4")], 1), round(ue_n.iloc[-1], 1), round(ne_u.loc[P("2022Q4")], 1), round(ne_u.iloc[-1], 1))
