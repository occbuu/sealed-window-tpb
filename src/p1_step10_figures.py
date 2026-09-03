"""Paper 1 - publication figures (300 dpi, colour-vision-safe palette).

Palette validated with the dataviz six-checks validator in light mode:
#0F6FC5 #C25708 #1B9E77 #9E4FA3 (all PASS); #6B7280 is reserved for
reference lines and annotation ink and never encodes a series.
"""
import os, re, pickle, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                             average_precision_score)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK, TAB = os.path.join(ROOT, "work"), os.path.join(ROOT, "outputs", "tables")
FIG = os.path.join(ROOT, "outputs", "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, GREEN, PURPLE = "#0F6FC5", "#C25708", "#1B9E77", "#9E4FA3"
INK, MUTED, GRID = "#1F2328", "#4B5563", "#E3E6EA"
SEQ = ["#DCE9F7", "#B7D3EF", "#7FB0E2", "#3E86D0", "#0F6FC5", "#0A4B87"]

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 9,
    "font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white"})

df = pd.read_parquet(os.path.join(WORK, "analysis.parquet"))
LAB = {"ATT": "Attitude", "SN": "Subjective\nnorm", "PBC": "Perceived\ncontrol",
       "SAT": "Satisfaction", "BI": "Revisit\nintention"}
FLAT = {k: v.replace("\n", " ") for k, v in LAB.items()}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name, flush=True)


def parse(cell):
    if not isinstance(cell, str) or not cell.strip():
        return np.nan, np.nan, ""
    m = re.match(r"(-?[\d.]+)(\*{0,3})\s*\(([\d.]+)\)", cell.strip())
    return (float(m.group(1)), float(m.group(3)), m.group(2)) if m else (np.nan, np.nan, "")


# ---------------------------------------------------------------- Fig 1
def fig1():
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.set_xlim(-0.4, 30.4); ax.set_ylim(0, 13.2); ax.axis("off")
    band_y, band_h = 6.6, 1.9
    ax.add_patch(Rectangle((0, band_y), 4.2, band_h, fc=BLUE, ec="none", alpha=.93))
    ax.text(2.1, band_y + band_h / 2, "$t_0$\nfirst review", ha="center",
            va="center", color="white", fontsize=7.4, fontweight="bold",
            linespacing=1.35)
    ax.add_patch(Rectangle((4.2, band_y), 3.4, band_h, fc="#F3F4F6", ec=MUTED, lw=.7))
    ax.text(5.9, band_y + band_h / 2, "blackout\n90 days", ha="center",
            va="center", fontsize=7.6, color=MUTED)
    ax.add_patch(Rectangle((7.6, band_y), 17.7, band_h, fc=GREEN, ec="none", alpha=.9))
    ax.text(16.45, band_y + band_h / 2, "behaviour observation window: months 3–24",
            ha="center", va="center", color="white", fontsize=8.4, fontweight="bold")
    ax.add_patch(Rectangle((25.3, band_y), 4.5, band_h, fc="#F3F4F6", ec=MUTED,
                           lw=.7, hatch="///"))
    ax.text(27.55, band_y + band_h / 2, "censored\n(excluded)", ha="center",
            va="center", fontsize=6.9, color=MUTED, linespacing=1.35,
            bbox=dict(fc="white", ec="none", pad=1.2, alpha=.85))
    ax.annotate("", xy=(30.2, 5.9), xytext=(0, 5.9),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1))
    for x, t in [(0, "$t_0$"), (4.2, "+90 d"), (25.3, "+24 m"),
                 (29.8, "snapshot\n2026-06")]:
        ax.plot([x, x], [5.7, 6.1], color=MUTED, lw=1)
        ax.text(x, 5.3, t, ha="center", va="top", fontsize=7.3, color=MUTED)
    ax.text(2.1, 11.9, "MEASUREMENT", ha="center", fontsize=8.2,
            fontweight="bold", color=BLUE)
    ax.text(2.1, 10.9, "ATT · SN · PBC\nSAT · BI\nfrom this text only",
            ha="center", va="top", fontsize=7.2, color=INK, linespacing=1.45)
    ax.text(16.45, 11.9, "OUTCOME", ha="center", fontsize=8.2,
            fontweight="bold", color=GREEN)
    ax.text(16.45, 10.9, "Y1 same-listing return    ·    Y2 platform continuance",
            ha="center", va="top", fontsize=7.4, color=INK)
    ax.text(0, 3.5, "Source article: constructs and behaviour are read from the "
            "same pooled set of a user's reviews, so\nintention can be recorded "
            "after the return has already happened.", fontsize=7.4, color=MUTED,
            va="top", linespacing=1.5)
    ax.text(0, 1.4, "This study: the measurement text is sealed at $t_0$ and "
            "cannot contain post-behaviour information.", fontsize=7.4,
            color=INK, va="top", fontweight="bold")
    save(fig, "fig1_design.png")


# ---------------------------------------------------------------- Fig 2
def fig2():
    g = df.groupby("year").agg(n=("y_revisit", "size"), y1=("y_revisit", "mean"),
                               y2=("y_continue", "mean")).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))
    ax = axes[0]
    ax.bar(g.year, g.n / 1000, color=BLUE, width=.72)
    ax.set_ylabel("Focal reviews (thousands)")
    ax.set_title("(a) Analytic sample size", fontsize=8.6, loc="left")
    ax = axes[1]
    ax.plot(g.year, 100 * g.y2, color=GREEN, lw=2, marker="o", ms=4)
    ax.set_ylabel("Continuance rate (%)")
    ax.set_ylim(0, 28)
    ax.set_title("(b) Y2 platform continuance", fontsize=8.6, loc="left")
    ax = axes[2]
    ax.plot(g.year, 100 * g.y1, color=ORANGE, lw=2, marker="s", ms=4)
    ax.set_ylabel("Same-listing return rate (%)")
    ax.set_ylim(0, 2.7)
    ax.set_title("(c) Y1 same-listing return", fontsize=8.6, loc="left")
    for ax in axes:
        ax.set_xlabel("Year of first review ($t_0$)", fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.tick_params(labelsize=7.5)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig2_corpus.png")


# ---------------------------------------------------------------- Fig 3
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0),
                             gridspec_kw={"width_ratios": [1, 1.18]})
    ax = axes[0]
    ks = ["ATT", "SN", "PBC", "SAT", "BI"]
    vals = [100 * df[f"pres_{k}"].mean() for k in ks]
    bars = ax.barh(range(len(ks)), vals, color=[BLUE, ORANGE, GREEN, PURPLE, "#0A4B87"],
                   height=.62)
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([FLAT[k] for k in ks], fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("Reviews containing ≥1 construct term (%)")
    ax.grid(axis="y", visible=False)
    for b, v in zip(bars, vals):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center",
                fontsize=7.6, color=INK)
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_title("(a) Construct prevalence", fontsize=9, loc="left")

    C = pd.read_csv(os.path.join(TAB, "table4_correlations.csv"), index_col=0)
    ax = axes[1]
    im = ax.imshow(C.values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq", SEQ), vmin=-0.1, vmax=1.0)
    ax.set_xticks(range(len(C))); ax.set_xticklabels(C.columns, fontsize=7.5, rotation=45)
    ax.set_yticks(range(len(C))); ax.set_yticklabels(C.index, fontsize=7.5)
    ax.grid(visible=False)
    for i in range(len(C)):
        for j in range(len(C)):
            v = C.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.9,
                    color="white" if v > 0.55 else INK)
    ax.set_title("(b) Spearman correlations", fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, fraction=.045, pad=.03)
    save(fig, "fig3_constructs.png")


# ---------------------------------------------------------------- Fig 4
def fig4():
    """Mediation diagram; direct paths are listed rather than over-plotted."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    keys = ["Attitude", "Subjective norm",
            "Perceived behavioural control", "Satisfaction"]
    short = dict(zip(keys, ["ATT", "SN", "PBC", "SAT"]))
    for ax, tag, title in zip(axes, ["revisit", "continue"],
                              ["(a) Y1 — same-listing return",
                               "(b) Y2 — platform continuance"]):
        t = pd.read_csv(os.path.join(TAB, f"table_gsem_{tag}.csv")).set_index("variable")
        ax.set_xlim(0, 10); ax.set_ylim(0, 10.6); ax.axis("off")
        ax.set_title(title, fontsize=9, loc="left")
        ys = [9.3, 7.7, 6.1, 4.5]
        for y, k in zip(ys, keys):
            ax.add_patch(FancyBboxPatch((0.15, y - .48), 1.7, .96,
                                        boxstyle="round,pad=0.04", fc="white",
                                        ec=MUTED, lw=.8))
            ax.text(1.0, y, short[k], ha="center", va="center", fontsize=8.4)
        ax.add_patch(FancyBboxPatch((4.15, 6.4), 1.8, .96,
                                    boxstyle="round,pad=0.04", fc="#EAF2FB",
                                    ec=BLUE, lw=1.2))
        ax.text(5.05, 6.88, "BI", ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=BLUE)
        ax.add_patch(FancyBboxPatch((7.55, 6.4), 2.3, .96,
                                    boxstyle="round,pad=0.04", fc="#E7F5EF",
                                    ec=GREEN, lw=1.2))
        ax.text(8.7, 6.88, "BEHAVIOUR", ha="center", va="center", fontsize=7.4,
                fontweight="bold", color=GREEN)
        for y, k in zip(ys, keys):
            c, _, st = parse(t.loc[k, "eq1_BI"])
            col = BLUE if c > 0 else ORANGE
            ax.add_patch(FancyArrowPatch((1.9, y), (4.12, 6.9),
                                         arrowstyle="-|>", mutation_scale=8,
                                         color=col, lw=1.1))
            fx, fy = 1.9 + .42 * (4.12 - 1.9), y + .42 * (6.9 - y)
            ax.text(fx, fy + .26, f"{c:+.3f}{st}", fontsize=6.6, color=col,
                    ha="center", va="bottom",
                    bbox=dict(fc="white", ec="none", pad=.9, alpha=.92))
        cb, _, stb = parse(t.loc["Revisit intention (BI)", "eq2_behaviour"])
        ax.add_patch(FancyArrowPatch((5.98, 6.88), (7.52, 6.88),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color=GREEN, lw=2.2))
        ax.text(6.75, 7.42, f"{cb:+.3f}{stb}", fontsize=7.8, color=GREEN,
                ha="center", fontweight="bold")
        ax.text(0.15, 3.55, "Direct paths to behaviour ($c'$), same model:",
                fontsize=7.2, color=INK, fontweight="bold", va="top")
        for i, k in enumerate(keys):
            c2, _, st2 = parse(t.loc[k, "eq2_behaviour"])
            col2 = BLUE if c2 > 0 else ORANGE
            ax.text(0.45, 2.80 - 0.62 * i, f"{short[k]} → behaviour", fontsize=7,
                    color=col2, va="top")
            ax.text(5.15, 2.80 - 0.62 * i, f"{c2:+.3f}{st2}", fontsize=7,
                    color=col2, va="top", ha="right")
        ax.text(0.15, 0.20, "blue = positive, orange = negative;  "
                "*** p<.001, ** p<.01, * p<.05", fontsize=6.4, color=MUTED,
                va="top")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig4_path_model.png")


# ---------------------------------------------------------------- Fig 5
def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    for ax, y, ttl in zip(axes, ["y_revisit", "y_continue"],
                          ["(a) Y1 — same-listing return",
                           "(b) Y2 — platform continuance"]):
        t = pd.read_csv(os.path.join(TAB, f"table_ml_bakeoff_{y}.csv"))
        t = (t.sort_values("test_auc", ascending=False)
               .drop_duplicates("model").sort_values("test_auc"))
        cols = [MUTED if m.startswith("B") else BLUE for m in t.model]
        ax.barh(range(len(t)), t.test_auc, color=cols, height=.62)
        ax.set_yticks(range(len(t)))
        ax.set_yticklabels([m.split(" ", 1)[1] for m in t.model], fontsize=7.4)
        ax.axvline(0.5, color=ORANGE, lw=1.2, ls="--")
        ax.set_xlim(0.45, 0.76)
        ax.set_xlabel("Test ROC-AUC (2022–2024 hold-out)", fontsize=8)
        ax.grid(axis="y", visible=False)
        ax.tick_params(labelsize=7.5)
        for i, v in enumerate(t.test_auc):
            ax.text(v + .006, i, f"{v:.3f}", va="center", fontsize=6.9, color=INK)
        ax.set_title(ttl, fontsize=9, loc="left")
    fig.text(0.5, -0.03, "Dashed line = chance (AUC 0.500). Grey bars are "
             "reference baselines; blue bars are full-feature learners.",
             ha="center", fontsize=6.8, color=MUTED)
    fig.tight_layout(w_pad=3.2)
    save(fig, "fig5_bakeoff.png")


# ---------------------------------------------------------------- Fig 6
def fig6():
    import sys
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from p1_step6_ml import make_features
    YL = {"y_revisit": "Y1 same-listing return", "y_continue": "Y2 platform continuance"}
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.0))
    for r, y in enumerate(["y_revisit", "y_continue"]):
        with open(os.path.join(WORK, f"lgbm_{y}.pkl"), "rb") as fh:
            blob = pickle.load(fh)
        X = make_features(df, df["year"] <= 2019, y)
        te = (df["year"] >= 2022).to_numpy()
        Xte = blob["scaler"].transform(X.loc[te])
        yte = df[y].to_numpy()[te]
        p = blob["model"].predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, p)
        axes[r, 0].plot(fpr, tpr, color=BLUE, lw=2)
        axes[r, 0].plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
        axes[r, 0].set_xlabel("False positive rate")
        axes[r, 0].set_ylabel("True positive rate")
        axes[r, 0].set_title(f"ROC — {YL[y]}\n(AUC {roc_auc_score(yte,p):.3f})",
                             fontsize=8.0, loc="left")
        pr, rc, _ = precision_recall_curve(yte, p)
        axes[r, 1].plot(rc, pr, color=GREEN, lw=2)
        axes[r, 1].axhline(yte.mean(), color=ORANGE, lw=1.2, ls="--")
        axes[r, 1].set_xlabel("Recall"); axes[r, 1].set_ylabel("Precision")
        axes[r, 1].set_title(
            f"Precision–recall\nAP {average_precision_score(yte,p):.4f} vs base {yte.mean():.4f}",
            fontsize=8.0, loc="left")
        q = pd.qcut(p, 10, labels=False, duplicates="drop")
        cal = pd.DataFrame({"p": p, "y": yte, "q": q}).groupby("q").mean()
        axes[r, 2].plot(cal.p, cal.y, color=PURPLE, lw=2, marker="o", ms=4)
        lim = max(cal.p.max(), cal.y.max()) * 1.1
        axes[r, 2].plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--")
        axes[r, 2].set_xlabel("Mean predicted probability")
        axes[r, 2].set_ylabel("Observed rate")
        axes[r, 2].set_title("Calibration (deciles)", fontsize=8.3, loc="left")
    fig.tight_layout()
    save(fig, "fig6_curves.png")


# ---------------------------------------------------------------- Fig 7
def fig7():
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.3))
    for ax, y, ttl in zip(axes, ["y_revisit", "y_continue"],
                          ["(a) Y1 — same-listing return",
                           "(b) Y2 — platform continuance"]):
        t = pd.read_csv(os.path.join(TAB, f"table_shap_importance_{y}.csv"))
        t = t.sort_values("mean_abs_shap").tail(12)
        pretty = {"city_rate": "City base rate", "log_prior": "Listing prior reviews",
                  "log_age": "Listing age", "log_len": "Review length",
                  "n_exclaim": "Exclamation marks", "month_cos": "Season (cos)",
                  "month_sin": "Season (sin)", "covid": "During COVID-19",
                  "SENT": "Sentiment (compound)", "vader_pos": "Sentiment (positive)",
                  "vader_neg": "Sentiment (negative)"}
        cmap = {"TPB construct": BLUE, "sentiment": GREEN, "control": MUTED}
        ax.barh(range(len(t)), t.share_pct, color=[cmap[b] for b in t.block],
                height=.62)
        ax.set_yticks(range(len(t)))
        ax.set_yticklabels([FLAT.get(f, pretty.get(f, f)) for f in t.feature],
                           fontsize=7.2)
        ax.set_xlabel("Share of total mean |SHAP| (%)", fontsize=8)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 52)
        ax.tick_params(labelsize=7.4)
        ax.set_title(ttl, fontsize=9, loc="left")
        share = t.loc[t.block == "TPB construct", "share_pct"].sum()
        ax.text(.96, .30, f"TPB block ≈ {share:.0f}%\nof attribution",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.2,
                color=BLUE, linespacing=1.4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, GREEN, MUTED)]
    axes[1].legend(handles, ["TPB construct", "Sentiment", "Control"],
                   loc="lower right", fontsize=7)
    fig.tight_layout(w_pad=3.0)
    save(fig, "fig7_shap.png")


# ---------------------------------------------------------------- Fig 8
def fig8():
    fig, axes = plt.subplots(2, 5, figsize=(7.8, 3.6), sharex=False)
    for r, y in enumerate(["y_revisit", "y_continue"]):
        a = pd.read_csv(os.path.join(WORK, f"ale_{y}.csv"))
        for c, k in enumerate(["ATT", "SN", "PBC", "SAT", "BI"]):
            ax = axes[r, c]
            s = a[a.feature == k]
            ax.plot(s.x, 100 * s.ale, color=BLUE if r == 0 else GREEN, lw=1.8)
            ax.axhline(0, color=MUTED, lw=.8, ls="--")
            ax.set_title(FLAT[k], fontsize=7.6, loc="left")
            ax.tick_params(labelsize=6.5)
            if c == 0:
                ax.set_ylabel(("Y1 " if r == 0 else "Y2 ") + "ALE (pp)", fontsize=7.4)
            if r == 1:
                ax.set_xlabel("standardised score", fontsize=6.8)
    fig.tight_layout()
    save(fig, "fig8_ale.png")


# ---------------------------------------------------------------- Fig 9
def fig9():
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    y = "y_continue"
    s = pd.read_csv(os.path.join(WORK, f"gam_shapes_{y}.csv"))
    cols = {"ATT": BLUE, "SN": ORANGE, "PBC": GREEN, "SAT": PURPLE, "BI": "#0A4B87"}
    ax = axes[0]
    for k, c in cols.items():
        d = s[s.feature == k]
        ax.plot(d.x, d.f, color=c, lw=1.8, label=FLAT[k])
    ax.axhline(0, color=MUTED, lw=.8, ls="--")
    ax.set_xlabel("Standardised construct score")
    ax.set_ylabel("Additive contribution (log-odds)")
    ax.set_title("(a) Glassbox GAM shape functions — Y2", fontsize=9, loc="left")
    ax.legend(fontsize=6.8, ncol=2)

    cate = pd.read_parquet(os.path.join(WORK, f"cate_{y}.parquet"))
    ax = axes[1]
    ax.hist(cate.cate, bins=60, color=BLUE, alpha=.85)
    ax.axvline(cate.cate.mean(), color=ORANGE, lw=1.6)
    ax.text(cate.cate.mean(), ax.get_ylim()[1] * .93,
            f"  ATE = {cate.cate.mean():.3f}", color=ORANGE, fontsize=7.4)
    ax.set_xlabel("Causal-forest CATE of revisit intention on Y2")
    ax.set_ylabel("Guests")
    ax.set_title("(b) Effect heterogeneity", fontsize=9, loc="left")
    fig.tight_layout()
    save(fig, "fig9_gam_cate.png")


# --------------------------------------------------------------- Fig 10
def fig10():
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    d = pd.read_csv(os.path.join(TAB, "table_dml.csv"))
    ax = axes[0]
    order = ["ATT", "SN", "PBC", "SAT", "BI"]
    for i, (y, col, off) in enumerate([("y_revisit", ORANGE, -.16),
                                       ("y_continue", GREEN, .16)]):
        s = d[d.outcome == y].set_index("construct").loc[order]
        pos = np.arange(len(order)) + off
        sc = 100.0
        ax.errorbar(sc * s.theta, pos, xerr=1.96 * sc * s.se, fmt="o", ms=5,
                    color=col, lw=1.4, capsize=2.5,
                    label="Y1 same-listing return" if i == 0 else "Y2 continuance")
    ax.axvline(0, color=MUTED, lw=1, ls="--")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([FLAT[k] for k in order], fontsize=7.6)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("DML effect on behaviour (percentage points per 1 SD)",
                  fontsize=8)
    ax.set_title("(a) Double machine learning", fontsize=9, loc="left")
    ax.tick_params(labelsize=7.4)
    leg_h, leg_l = ax.get_legend_handles_labels()

    ax = axes[1]
    c1 = pd.read_csv(os.path.join(TAB, "table_counterfactual_y_revisit.csv"))
    c2 = pd.read_csv(os.path.join(TAB, "table_counterfactual_y_continue.csv"))
    h = .36; ys = np.arange(len(order))
    v1 = [float(c1.set_index("construct").loc[k, "share_flippable_pct"]) for k in order]
    v2 = [float(c2.set_index("construct").loc[k, "share_flippable_pct"]) for k in order]
    ax.barh(ys - h / 2, v1, h, color=ORANGE, label="Y1 same-listing return")
    ax.barh(ys + h / 2, v2, h, color=GREEN, label="Y2 continuance")
    ax.set_yticks(ys); ax.set_yticklabels([FLAT[k] for k in order], fontsize=7.6)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Guests moved into the top risk decile (%)", fontsize=8)
    ax.set_xlim(0, 24)
    ax.tick_params(labelsize=7.4)
    for yy, v in list(zip(ys - h / 2, v1)) + list(zip(ys + h / 2, v2)):
        if v > 0.05:
            ax.text(v + .4, yy, f"{v:.1f}", va="center", fontsize=6.5, color=INK)
    ax.set_title("(b) Counterfactual leverage", fontsize=9, loc="left")
    fig.legend(leg_h, leg_l, loc="lower center", ncol=2, fontsize=7.4,
               bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(w_pad=2.4)
    save(fig, "fig10_causal_counterfactual.png")


# --------------------------------------------------------------- Fig 11
def fig11():
    t = pd.read_csv(os.path.join(TAB, "table_gsem_robustness.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
    modes = ["count", "dens", "pres"]
    mlab = {"count": "keyword count", "dens": "keyword density",
            "pres": "binary presence"}
    cols = {"count": BLUE, "dens": ORANGE, "pres": GREEN}
    paths = ["ATT->BI", "SN->BI", "PBC->BI", "SAT->BI"]
    for ax, y, ttl in zip(axes, ["y_revisit", "y_continue"],
                          ["(a) Y1 — same-listing return",
                           "(b) Y2 — platform continuance"]):
        pl = paths + [f"{k}->Y" for k in ["ATT", "SN", "PBC", "SAT"]] + ["BI->Y"]
        for j, m in enumerate(modes):
            row = t[(t.measurement == m) & (t.outcome == y)].iloc[0]
            vals = [parse(row[p])[0] for p in pl]
            ses = [parse(row[p])[1] for p in pl]
            pos = np.arange(len(pl)) + (j - 1) * .26
            ax.errorbar(vals, pos, xerr=[1.96 * s for s in ses], fmt="o", ms=3.6,
                        color=cols[m], lw=1.1, capsize=2, label=mlab[m])
        ax.axvline(0, color=MUTED, lw=1, ls="--")
        ax.set_yticks(range(len(pl)))
        ax.set_yticklabels([p.replace("->", " → ") for p in pl], fontsize=6.8)
        ax.invert_yaxis(); ax.grid(axis="y", visible=False)
        ax.set_xlabel("Coefficient (95% CI)", fontsize=8)
        ax.set_title(ttl, fontsize=9, loc="left")
        ax.tick_params(labelsize=7.2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.4,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, "fig11_robustness.png")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
    fns = {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5, "6": fig6,
           "7": fig7, "8": fig8, "9": fig9, "10": fig10, "11": fig11}
    for k in which:
        fns[k]()
