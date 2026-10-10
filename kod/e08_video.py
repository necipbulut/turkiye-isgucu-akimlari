# -*- coding: utf-8 -*-
"""e08_video.py - animated three-state labour-flow video, 2010Q2-2026Q2 (Turkish labels).

Data: Eurostat lfsi_long_q (transitions, thousand persons and rates), lfsi_emp_q (labour force, employment,
population) and lfsi_sla_q (underemployed part-time, potential labour force); Türkiye, 15-74, seasonally
adjusted. The data are quarterly; frames between quarters are a visual interpolation, and the quarter label
always names the nearest observed quarter. 2021Q1 has no transition data (questionnaire change): the arrows
fade out and back in around it. Output: ../video/isgucu_akimlari_2010_2026.mp4 and a poster PNG.
"""
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle  # noqa: E402

from jsonstat import read  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "eurostat"
VID = ROOT / "video"
VID.mkdir(exist_ok=True)

C_E, C_U, C_N = "#008c84", "#cf5a2e", "#8256ab"
INK, GREY, BAND, BG = "#1c1e21", "#6b7076", "#e7e4dc", "#fbfaf7"
mpl.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555",
                     "axes.linewidth": 0.8, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5})


def ser(df, **f):
    for k, v in f.items():
        df = df[df[k] == v]
    s = df.set_index("time")["value"].astype(float).sort_index()
    s.index = pd.PeriodIndex([x.replace("-", "") for x in s.index], freq="Q")
    return s


lg, _ = read(DATA / "lfsi_long_q.json")
em, _ = read(DATA / "lfsi_emp_q.json")
sl, _ = read(DATA / "lfsi_sla_q.json")
IDX = pd.period_range("2010Q2", "2026Q2", freq="Q")
K = {"EU": "EMP_UNE", "EN": "EMP_INAC", "UE": "UNE_EMP", "UN": "UNE_INAC", "NE": "INAC_EMP", "NU": "INAC_UNE"}
UNIT = {"EMP": "PC_EMP", "UNE": "PC_UNE", "INAC": "PC_INAC"}
cnt = pd.DataFrame({k: ser(lg, s_adj="SA", sex="T", unit="THS_PER", wstatus=w) for k, w in K.items()}).reindex(IDX)
rate = pd.DataFrame({k: ser(lg, s_adj="SA", sex="T", unit=UNIT[w.split("_")[0]], wstatus=w) for k, w in K.items()}).reindex(IDX)
act = ser(em, s_adj="SA", sex="T", age="Y15-74", unit="THS_PER", indic_em="ACT")
actp = ser(em, s_adj="SA", sex="T", age="Y15-74", unit="PC_POP", indic_em="ACT")
emp = ser(em, s_adj="SA", sex="T", age="Y15-74", unit="THS_PER", indic_em="EMP_LFS")
st = pd.DataFrame({"ACT": act, "E": emp, "POP": act / (actp / 100)}).reindex(IDX)
st["U"], st["N"] = st.ACT - st.E, st.POP - st.ACT
w = {x: ser(sl, s_adj="SA", sex="T", age="Y15-74", unit="THS_PER", wstatus=x).reindex(IDX) for x in ["NSEEK_AVL", "SEEK_NAVL", "UEMP_PT"]}
st["PLF"], st["PT"] = w["NSEEK_AVL"] + w["SEEK_NAVL"], w["UEMP_PT"]
st["u"] = st.U / st.ACT * 100
st["slack"] = (st.U + st.PT + st.PLF) / (st.ACT + st.PLF) * 100


def steady(p):
    eu, en, ue, un, ne, nu = (p[k] / 100 for k in ["EU", "EN", "UE", "UN", "NE", "NU"])
    M = np.array([[1 - eu - en, eu, en], [ue, 1 - ue - un, un], [ne, nu, 1 - ne - nu]])
    wv, v = np.linalg.eig(M.T)
    s = np.real(v[:, np.argmin(abs(wv - 1))])
    s = s / s.sum()
    return s[1] / (s[0] + s[1]) * 100


r4, c4 = rate.rolling(4).mean(), cnt.rolling(4).mean()
st["ustar"] = [steady(r4.loc[i]) if r4.loc[i].notna().all() else np.nan for i in IDX]
st["sep_u"] = c4.EU / (c4.EU + c4.EN) * 100
st["ue_n"] = c4.UN / (c4.UE + c4.UN) * 100
NQ = len(IDX)
LAB = [str(p) for p in IDX]
qi = lambda s: LAB.index(s)

# ------------------------------------------------------------------ frame schedule
FPS, FPQ, HOLD = 30, 14, 22
holds = {qi(q) for q in ["2014Q1", "2018Q3", "2020Q2", "2021Q2", "2022Q4", "2023Q2", "2024Q2", "2026Q2"]}
xs = []
for i in range(NQ):
    if i in holds:
        xs += [float(i)] * HOLD
    if i < NQ - 1:
        xs += [i + k / FPQ for k in range(FPQ)]
xs += [float(NQ - 1)] * (FPS * 2)
print("frames", len(xs), "seconds", round(len(xs) / FPS, 1))


def ease(t):
    return t * t * (3 - 2 * t)


def at(col, x):
    """Value of a stock/rate series at fractional quarter x (NaN-safe linear blend)."""
    i = int(np.floor(x)); f = ease(x - i)
    a = st[col].iloc[i]
    b = st[col].iloc[min(i + 1, NQ - 1)]
    if np.isnan(a):
        return b
    if np.isnan(b) or f == 0:
        return a
    return a + (b - a) * f


def flows_at(x):
    i = int(np.floor(x)); f = ease(x - i)
    A, B = cnt.iloc[i], cnt.iloc[min(i + 1, NQ - 1)]
    oa, ob = (0.0 if A.isna().any() else 1.0), (0.0 if B.isna().any() else 1.0)
    if oa == 0 and ob == 0:
        return None, 0.0
    if oa == 0:
        A = B
    if ob == 0:
        B = A
    return (A + (B - A) * f), oa + (ob - oa) * f


# ------------------------------------------------------------------ figure skeleton
fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
fig.text(0.03, 0.935, "Türkiye'de işgücü akımları, 2010–2026", fontsize=24, fontweight="bold", color=INK)
fig.text(0.03, 0.895, "Her çeyrek istihdam, işsizlik ve işgücü dışı arasında yer değiştiren kişiler (bin kişi, bir önceki çeyrekten)",
         fontsize=12.5, color=GREY)
fig.text(0.03, 0.025, "Kaynak: Eurostat lfsi_long_q, lfsi_emp_q, lfsi_sla_q (TÜİK HİA), 15–74 yaş, mevsimsellikten arındırılmış. "
         "Veri çeyreklik; çeyrekler arası geçiş görsel yumuşatmadır.  Necip Bulut · github.com/necipbulut/turkiye-isgucu-akimlari",
         fontsize=9.5, color=GREY)
qtext = fig.text(0.535, 0.80, "", fontsize=34, fontweight="bold", color=INK, ha="right", family="DejaVu Sans Mono")
ptext = fig.text(0.535, 0.765, "", fontsize=13, color=GREY, ha="right")
etext = fig.text(0.03, 0.085, "", fontsize=13.5, color=INK, va="bottom",
                 bbox=dict(boxstyle="round,pad=0.45", fc="#f3eef8", ec="none"))
ntext = fig.text(0.03, 0.135, "", fontsize=11, color=GREY, va="bottom")

axm = fig.add_axes([0.0, 0.12, 0.58, 0.74])
axm.set_xlim(-1.85, 1.85)
axm.set_ylim(-0.8, 1.72)
axm.set_aspect("equal")
axm.axis("off")
POS = {"E": (-1.0, -0.02), "N": (1.0, -0.02), "U": (0.0, 1.17)}
COL = {"E": C_E, "U": C_U, "N": C_N}
NAME = {"E": "İstihdam", "U": "İşsiz", "N": "İşgücü dışı"}
maxS = np.nanmax(np.r_[st.E.values, st.N.values])
RM = 0.60
rad = lambda v: RM * np.sqrt(max(v, 0) / maxS)
maxF = np.nanmax(cnt.values)
nodes = {}
for k in "EUN":
    o = Circle(POS[k], 0.1, facecolor=mpl.colors.to_rgba(COL[k], 0.13), edgecolor=COL[k], lw=1.6, zorder=1)
    h = Circle(POS[k], 0.05, facecolor="none", edgecolor=COL[k], hatch="////", lw=(0 if k == "U" else 0.8),
               ls=("-" if k == "U" else "--"), zorder=2)
    axm.add_patch(o); axm.add_patch(h)
    nodes[k] = dict(o=o, h=h, name=axm.text(0, 0, NAME[k], ha="center", fontsize=15, fontweight="bold", zorder=5),
                    val=axm.text(0, 0, "", ha="center", fontsize=13.5, zorder=5),
                    sub=axm.text(0, 0, "", ha="center", fontsize=10.5, color=GREY, zorder=5))
FLW = [("EU", "E", "U"), ("EN", "E", "N"), ("UE", "U", "E"), ("UN", "U", "N"), ("NE", "N", "E"), ("NU", "N", "U")]
arrows, labels = {}, {}
for key, a, b in FLW:
    ar = FancyArrowPatch((0, 0), (1, 1), connectionstyle="arc3,rad=0.32", color=COL[b], lw=2, arrowstyle="-|>",
                         mutation_scale=10, zorder=3, shrinkA=0, shrinkB=0)
    axm.add_patch(ar)
    arrows[key] = ar
    labels[key] = axm.text(0, 0, "", ha="center", va="center", fontsize=11, color=INK, zorder=6,
                           bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.85))

# time-series panels
x_all = np.arange(NQ)
ax1 = fig.add_axes([0.635, 0.53, 0.335, 0.31], facecolor=BG)
ax2 = fig.add_axes([0.635, 0.14, 0.335, 0.29], facecolor=BG)
nr0, nr1 = qi("2021Q3"), qi("2024Q2")
for ax in (ax1, ax2):
    ax.axvspan(nr0 - 0.5, nr1 + 0.5, color=BAND, lw=0, zorder=0)
    ax.axvline(qi("2014Q1") - 0.5, color="#aaa", lw=0.7, ls=":", zorder=0)
    ax.set_xlim(-0.5, NQ - 0.5)
    yt = [qi(f"{y}Q1") for y in range(2011, 2027, 3)]
    ax.set_xticks(yt, [str(y) for y in range(2011, 2027, 3)])
    ax.grid(axis="y", color="#e3e1dc", lw=0.6)
ax1.set_ylim(0, 30)
ax1.set_title("Oranlar, %", loc="left", fontsize=13, fontweight="bold")
ax2.set_ylim(15, 60)
ax2.set_title("Akımların yönü, % (dört çeyrek ortalaması)", loc="left", fontsize=13, fontweight="bold")
L = {}
for ax, key, c, ls, lab in [(ax1, "slack", C_N, "-", "Atıl işgücü (AB)"), (ax1, "u", C_U, "-", "İşsizlik"),
                            (ax1, "ustar", INK, (0, (4, 2)), "Durağan işsizlik"),
                            (ax2, "ue_n", C_N, "-", "İşsizlikten çıkanların işgücü dışına gidenleri"),
                            (ax2, "sep_u", C_U, "-", "İşten ayrılanların işsizliğe gidenleri")]:
    L[key], = ax.plot([], [], color=c, ls=ls, lw=2.2, label=lab)
    L[key + "_dot"], = ax.plot([], [], "o", color=c, ms=6)
ax1.legend(loc="upper left", fontsize=10, frameon=False, ncol=3, handlelength=1.6, columnspacing=0.9)
ax2.legend(loc="upper left", fontsize=10, frameon=False, handlelength=1.6)
cur1 = ax1.axvline(0, color=INK, lw=1.0, alpha=0.6)
cur2 = ax2.axvline(0, color=INK, lw=1.0, alpha=0.6)

EVENTS = [("2010Q2", "2013Q4", "2014 öncesi: eski yöntem ve nüfus projeksiyonu; düzeyler tam karşılaştırılamaz"),
          ("2018Q3", "2019Q3", "2018: kur krizi"),
          ("2020Q2", "2020Q4", "2020: pandemi, işten çıkarma yasağı ve kısa çalışma ödeneği"),
          ("2021Q1", "2021Q1", "2021Q1: anket yöntemi değişti, akım verisi yok"),
          ("2021Q3", "2022Q4", "Negatif reel faiz: politika faizi indirimleri (Eylül 2021–Şubat 2023)"),
          ("2023Q1", "2023Q2", "Şubat 2023 depremi ve EYT ile erken emeklilik"),
          ("2023Q3", "2024Q2", "Haziran 2023 politika dönüşü: sıkılaştırma evresi"),
          ("2024Q3", "2026Q2", "Yüksek reel faiz: normalleşme")]
PHASE = [("2010Q2", "2017Q2", "Sermaye girişi yılları"), ("2017Q3", "2019Q4", "Kredi garantisi ve kur krizi"),
         ("2020Q1", "2021Q2", "Pandemi"), ("2021Q3", "2023Q2", "NRIR · indirim evresi"),
         ("2023Q3", "2024Q2", "NRIR · sıkılaştırma evresi"), ("2024Q3", "2026Q2", "Normalleşme")]


def label_for(i, table):
    q = IDX[i]
    for a, b, t in table:
        if pd.Period(a, "Q") <= q <= pd.Period(b, "Q"):
            return t
    return ""


def fmt(v):
    return f"{v:,.0f}".replace(",", ".")


def draw(x):
    i_near = int(round(x))
    R = {k: rad(at(k, x)) for k in "EUN"}
    inner = {"E": rad(at("PT", x)), "N": rad(at("PLF", x))}
    for k in "EUN":
        n, (cx, cy) = nodes[k], POS[k]
        n["o"].set_radius(R[k])
        if k == "U":
            n["h"].center = (cx, cy); n["h"].set_radius(R[k])
            n["name"].set_position((cx, cy + R[k] + 0.2)); n["val"].set_position((cx, cy + R[k] + 0.06))
            n["val"].set_text(f"{at('U', x) / 1000:.1f} m".replace(".", ","))
            n["sub"].set_text("")
        else:
            ri = inner[k]
            n["h"].center = (cx, cy - R[k] + ri + 0.02); n["h"].set_radius(ri)
            sub_y = cy - R[k] + 2 * ri + 0.07           # just above the hatched inner circle
            val_y = max(cy - 0.01, sub_y + 0.14)        # stock value stays clear of the sub-label
            n["name"].set_position((cx, val_y + 0.15)); n["val"].set_position((cx, val_y))
            n["val"].set_text(f"{at(k, x) / 1000:.1f} m".replace(".", ","))
            n["sub"].set_position((cx, sub_y))
            sub = at("PT", x) if k == "E" else at("PLF", x)
            n["sub"].set_text(("eksik çalışan " if k == "E" else "çalışmaya hazır ") + f"{sub / 1000:.1f} m".replace(".", ","))
    F, alpha = flows_at(x)
    for key, a, b in FLW:
        ar, lb = arrows[key], labels[key]
        if F is None or alpha <= 0.01:
            ar.set_visible(False); lb.set_visible(False); continue
        v = F[key]
        lw = 0.8 + v / maxF * 9.5
        (ax_, ay_), (bx_, by_) = POS[a], POS[b]
        th = np.arctan2(by_ - ay_, bx_ - ax_); dl = 0.33
        sx, sy = ax_ + (R[a] + 0.03) * np.cos(th - dl), ay_ + (R[a] + 0.03) * np.sin(th - dl)
        ex, ey = bx_ + (R[b] + 0.045) * np.cos(th + np.pi + dl), by_ + (R[b] + 0.045) * np.sin(th + np.pi + dl)
        ar.set_positions((sx, sy), (ex, ey))
        ar.set_linewidth(lw)
        ar.set_arrowstyle(f"-|>,head_length={0.35 + lw * 0.06},head_width={0.22 + lw * 0.045}")
        ar.set_alpha(0.9 * alpha); ar.set_visible(True)
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        cx_, cy_ = mx + 0.32 * (ey - sy), my - 0.32 * (ex - sx)
        t = 0.36
        px = (1 - t) ** 2 * sx + 2 * t * (1 - t) * cx_ + t ** 2 * ex
        py = (1 - t) ** 2 * sy + 2 * t * (1 - t) * cy_ + t ** 2 * ey
        nx, ny = (ey - sy), -(ex - sx); nn = np.hypot(nx, ny); off = 0.075 + lw * 0.006
        lb.set_position((px + nx / nn * off, py + ny / nn * off)); lb.set_text(fmt(v))
        lb.set_alpha(alpha); lb.get_bbox_patch().set_alpha(0.85 * alpha); lb.set_visible(True)
    qtext.set_text(LAB[i_near])
    ptext.set_text(label_for(i_near, PHASE))
    ev = label_for(i_near, EVENTS)
    etext.set_text(ev); etext.set_visible(bool(ev) and not ev.startswith("2014 öncesi"))
    ntext.set_text(ev if ev.startswith("2014 öncesi") else ""); ntext.set_visible(ev.startswith("2014 öncesi"))
    # progressive lines up to x
    for key in ["slack", "u", "ustar", "ue_n", "sep_u"]:
        y = st[key].values.copy()
        k0 = int(np.floor(x))
        xx = list(x_all[:k0 + 1]); yy = list(y[:k0 + 1])
        if x > k0 and k0 + 1 < NQ:
            xx.append(x); yy.append(at(key, x) if not (np.isnan(y[k0]) or np.isnan(y[k0 + 1])) else np.nan)
        L[key].set_data(xx, yy)
        last = yy[-1] if len(yy) else np.nan
        L[key + "_dot"].set_data([xx[-1]] if not np.isnan(last) else [], [last] if not np.isnan(last) else [])
    cur1.set_xdata([x, x]); cur2.set_xdata([x, x])


writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=-1,
                      extra_args=["-pix_fmt", "yuv420p", "-crf", "21", "-preset", "slow", "-movflags", "+faststart"],
                      metadata={"title": "Türkiye'de işgücü akımları, 2010–2026", "artist": "Necip Bulut"})
out = VID / "isgucu_akimlari_2010_2026.mp4"
with writer.saving(fig, str(out), dpi=100):
    for n, x in enumerate(xs):
        draw(x)
        writer.grab_frame(facecolor=BG)
        if n % 200 == 0:
            print("frame", n)
draw(float(NQ - 1))
fig.savefig(VID / "isgucu_akimlari_poster.png", dpi=100, facecolor=BG)
for q in ["2012Q2", "2020Q2", "2021Q1", "2022Q4"]:
    draw(float(qi(q)))
    fig.savefig(VID / f"_check_{q}.png", dpi=60, facecolor=BG)
print("written", out, round(out.stat().st_size / 1e6, 1), "MB")
