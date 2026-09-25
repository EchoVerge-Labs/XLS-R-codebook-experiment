#!/usr/bin/env python
"""Regenerate the documentation figures from the versioned result files.

Every chart in README.md and docs/ is produced here, and every number it draws is read
from a result file that is tracked in this repository - nothing is typed in by hand.
Each figure is written twice, for GitHub's light and dark themes, and the pages select
between them with <picture>.

    python docs/scripts/make_figures.py          # writes docs/assets/*.png
    python docs/scripts/make_figures.py --print  # also prints the values drawn

Colour follows the entity, never the rank: English, Sinhala and Tamil keep the same hue
in every chart, and positive controls are neutral grey. Absolute CER depends on the
script, so the budget chart gives each language its own panel and y-axis rather than
inviting a cross-language comparison the data does not support.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "assets"
A = REPO / "Experiment_A_Diagnostic" / "results"
B = REPO / "Experiment_B_Multilingual" / "results"
C = REPO / "Experiment_C_LayerProbe" / "results"

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", band="#ecebe6", band2="#f5f4f0",
                  English="#2a78d6", Sinhala="#eb6834", Tamil="#1baf7a"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", axis="#383835", band="#2f2f2c", band2="#242423",
                  English="#3987e5", Sinhala="#d95926", Tamil="#199e70"),
}
LANGS = ["English", "Sinhala", "Tamil"]


def style(ax, t, grid_axis="y"):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
    ax.tick_params(colors=t["ink2"], labelsize=9, length=0, pad=6)
    if grid_axis != "none":
        ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])


def new_fig(t, w, h, ncols=1, **kw):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), **kw)
    fig.patch.set_facecolor(t["surface"])
    return fig, axes


def title(fig, t, text, sub):
    fig.text(0.012, 0.975, text, ha="left", va="top", fontsize=12.5, weight="bold", color=t["ink"])
    fig.text(0.012, 0.915, sub, ha="left", va="top", fontsize=9.5, color=t["ink2"])


def save(fig, name, theme):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}-{theme}.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------- 1. fit
def fig_codebook_fit(t, theme, show):
    s = json.load(open(A / "masked_summary.json"))
    en = s["English_CV"]["nce_mean"]
    rows = [("English · Common Voice", "English_CV", "English"),
            ("Sinhala · read", "Sinhala_read", "Sinhala"),
            ("Tamil · read", "Tamil_read", "Tamil"),
            ("Sinhala · YouTube", "Sinhala_YT", "Sinhala"),
            ("Tamil · YouTube", "Tamil_YT", "Tamil"),
            (None, None, None),
            ("Speed/pitch 1.45×", "Ctl_SpeedPitch", None),
            ("Six-way babble", "Ctl_Babble", None),
            ("Time reversal", "Ctl_Reversed", None),
            ("Sine chirps", "Ctl_Tones", None),
            ("White noise", "Ctl_WhiteNoise", None)]
    fig, ax = new_fig(t, 8, 4.6)
    fig.subplots_adjust(left=0.25, right=0.97, top=0.80, bottom=0.12)
    style(ax, t, grid_axis="x")
    y = 0
    labels, ys = [], []
    for label, key, lang in rows:
        if key is None:
            y += 0.6
            continue
        r = s[key]["nce_mean"] / en
        c = t[lang] if lang else t["muted"]
        ax.plot([1.0, r], [y, y], color=c, linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter([r], [y], s=64, color=c, edgecolor=t["surface"], linewidth=2, zorder=3)
        labels.append(label); ys.append(y)
        if show:
            print(f"  fit  {label:24s} {r:.3f}x")
        y += 1
    ax.axvline(1.0, color=t["ink2"], linewidth=1, zorder=1)
    chance = np.log(101) / en
    ax.axvline(chance, color=t["muted"], linewidth=1, linestyle=(0, (2, 2)), zorder=1)
    ax.text(chance, -0.95, "chance", color=t["muted"], fontsize=8.5, ha="center")
    ax.text(1.0, -0.95, "English", color=t["ink2"], fontsize=8.5, ha="center")
    ax.set_yticks(ys, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.6, 1.8)
    ax.set_xlabel("Masked InfoNCE relative to English  (lower = better fit)")
    title(fig, t, "Study 1 · The instrument has range; Sinhala and Tamil sit at or below English",
          "XLS-R 0.3B, masked positions. Grey rows are positive controls built from the English clips.")
    save(fig, "fig1-codebook-fit", theme)


# ---------------------------------------------------------------------------- 2. deficit
def fig_codebook_deficit(t, theme, show):
    d = json.load(open(A / "codebook_deficit" / "codebook_deficit_xlsr300m_mask0.065.json"))
    keys = {"English": "English_CV", "Sinhala": "Sinhala_read", "Tamil": "Tamil_read"}
    fig, ax = new_fig(t, 8, 4.3)
    fig.subplots_adjust(left=0.10, right=0.86, top=0.80, bottom=0.14)
    style(ax, t)
    margin = d["config"]["margin"]
    study2 = d["config"]["study2_max_diff"]
    for lang in LANGS:
        c = d["languages"][keys[lang]]["conditions"]
        xs, ys, lo, hi = [0.0], [0.0], [0.0], [0.0]
        for k in ("top_5", "top_10", "top_25", "top_50"):
            xs.append(100 * c[k]["used_entries_blocked_frac"])
            ys.append(c[k]["delta_nce_perfile"])
            lo.append(c[k]["delta_ci95"][0]); hi.append(c[k]["delta_ci95"][1])
        ax.fill_between(xs, lo, hi, color=t[lang], alpha=0.18, linewidth=0)
        ax.plot(xs, ys, color=t[lang], linewidth=2, marker="o", markersize=5.5,
                markeredgecolor=t["surface"], markeredgewidth=1.5)
        ax.text(xs[-1] + 1.2, ys[-1], lang, color=t["ink"], fontsize=9.5, va="center")
        if show:
            print(f"  deficit {lang:8s}", " ".join(f"{x:.0f}%:{v:.3f}" for x, v in zip(xs, ys)))
    for yv, lab, x, ha in ((margin, f"equivalence margin, {margin:.3f}", 0.8, "left"),
                           (study2, f"largest Study 2 language difference, {study2:.3f}", 57.5, "right")):
        ax.axhline(yv, color=t["muted"], linewidth=1, linestyle=(0, (2, 2)))
        ax.text(x, yv + 0.012, lab, color=t["ink2"], fontsize=8.5, ha=ha)
    ax.set_xlim(0, 58)
    ax.set_ylim(0, 0.62)
    ax.set_xlabel("Most-used codebook entries forbidden at inference (% of entries the language uses)")
    ax.set_ylabel("Increase in masked InfoNCE")
    title(fig, t, "Post-hoc · The metric detects a damaged codebook",
          "Clean audio, codebook entries removed. XLS-R 0.3B, mask probability 0.065; bands are 95% CIs.")
    save(fig, "fig2-codebook-deficit", theme)


# ---------------------------------------------------------------------------- 3. equivalence
def fig_equivalence(t, theme, show):
    st = json.load(open(B / "final_statistics.json"))
    arms = [("Full clips", "full"), ("6.0 s crop", "crop6"), ("9.0 s crop", "crop9")]
    m1, m2 = st["margin_primary"], st["margin_secondary"]
    fig, ax = new_fig(t, 8, 3.2)
    fig.subplots_adjust(left=0.14, right=0.82, top=0.73, bottom=0.19)
    style(ax, t, grid_axis="none")
    ax.axvspan(-m2, m2, color=t["band2"], zorder=0)
    ax.axvspan(-m1, m1, color=t["band"], zorder=0)
    ax.axvline(0, color=t["ink2"], linewidth=1, zorder=1)
    for i, (label, key) in enumerate(arms):
        a = st["arms"][key]["tost_0.289"]
        ax.plot([a["ci95_lo"], a["ci95_hi"]], [i, i], color=t["ink"], linewidth=1.2, zorder=2)
        ax.plot([a["ci90_lo"], a["ci90_hi"]], [i, i], color=t["ink"], linewidth=4.5,
                solid_capstyle="butt", zorder=3)
        ax.scatter([a["mean"]], [i], s=70, color=t["surface"], edgecolor=t["ink"], linewidth=2, zorder=4)
        ax.text(m2 + 0.04, i, f"TOST p = {a['p_tost']:.4f}".rstrip("0"), va="center",
                color=t["ink2"], fontsize=9)
        if show:
            print(f"  equiv {label:11s} mean {a['mean']:+.3f} 90% [{a['ci90_lo']:+.3f},{a['ci90_hi']:+.3f}] p {a['p_tost']:.4f}")
    ax.set_yticks(range(len(arms)), [a[0] for a in arms])
    ax.set_ylim(len(arms) - 0.4, -0.6)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-m2 - 0.05, m2 + 0.05)
    ax.set_xticks([-m2, -m1, 0, m1, m2], [f"−{m2:.2f}", f"−{m1:.3f}", "0", f"{m1:.3f}", f"{m2:.2f}"])
    ax.set_xlabel("Paired InfoNCE difference, unseen − seen  (thick: 90% CI, thin: 95% CI)")
    ax.tick_params(axis="y", labelsize=9.5)
    title(fig, t, "Study 2 · Pre-training membership has an equivalence-bounded null effect",
          "Nine family-matched pairs on XLSR-53. Shaded: margins from the speed/pitch (dark) and time-reversal (light) controls.")
    save(fig, "fig3-equivalence", theme)


# ---------------------------------------------------------------------------- 4. layers
def fig_layer_profiles(t, theme, show):
    p = json.load(open(C / "probe_results_xlsr300m.json"))
    fl = json.load(open(C / "probe_results_floor.json"))
    fig, ax = new_fig(t, 8, 4.2)
    fig.subplots_adjust(left=0.10, right=0.80, top=0.80, bottom=0.14)
    style(ax, t)
    for lang in LANGS:
        L = p["layers"][lang.lower()]
        layers = sorted(int(k) for k in L)
        cer = np.array([L[str(k)]["ctc"]["cer_mean"] for k in layers])
        norm = cer / cer.min()
        ax.plot(layers, norm, color=t[lang], linewidth=2)
        best = layers[int(cer.argmin())]
        ax.scatter([best], [1.0], s=64, color=t[lang], edgecolor=t["surface"], linewidth=2, zorder=4)
        ax.text(layers[-1] + 0.5, norm[-1], f"{lang} (best L{best})", color=t["ink"], fontsize=9, va="center")
        if show:
            print(f"  layers {lang:8s} best L{best} CER {cer.min():.3f}")
        F = fl["layers"][lang.lower()]
        fcer = np.array([F[str(k)]["ctc"]["cer_mean"] for k in sorted(int(k) for k in F)])
        ax.plot(sorted(int(k) for k in F), fcer / fcer.min(), color=t["muted"], linewidth=1,
                linestyle=(0, (1.5, 2)), zorder=1)
    ax.text(0.3, 0.84, "dotted: random-init floor, no depth structure", color=t["muted"], fontsize=8.5,
            va="center")
    ax.set_ylim(0.72, 4.25)
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("CER ÷ own best-layer CER")
    title(fig, t, "Study 3 · The same depth profile in all three languages",
          "Linear CTC probes on frozen XLS-R 0.3B, 3 h per language, mean of 3 seeds; normalised within each language.")
    save(fig, "fig4-layer-profiles", theme)


# ---------------------------------------------------------------------------- 5. budget
def fig_budget(t, theme, show):
    p = json.load(open(C / "probe_results_xlsr300m.json"))
    ft3 = json.load(open(C / "finetune.json"))
    ft6 = json.load(open(C / "finetune_6h.json"))
    fig, axes = new_fig(t, 8, 3.9, ncols=3)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.74, bottom=0.17, wspace=0.28)
    for ax, lang in zip(axes, LANGS):
        key = lang.lower()
        style(ax, t)
        sc = p["scaling"][key]
        hs = [0.5, 1.0, 2.0, 3.0]
        frozen = [sc[str(h)]["cer_mean"] for h in hs]
        ax.plot(hs, frozen, color=t[lang], linewidth=2, marker="o", markersize=5,
                markeredgecolor=t["surface"], markeredgewidth=1.5)
        fx, fy = [3.0], [float(np.mean([r["cer_final"] for r in ft3[key].values()]))]
        if key in ft6:
            fx.append(6.0); fy.append(float(np.mean([r["cer_final"] for r in ft6[key].values()])))
        ax.plot(fx, fy, color=t[lang], linewidth=2, linestyle=(0, (4, 2)), marker="o", markersize=6.5,
                markerfacecolor=t["surface"], markeredgecolor=t[lang], markeredgewidth=2)
        ax.set_ylim(0, max(frozen) * 1.25)
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.42, 8.5)
        ax.set_xticks([0.5, 1, 2, 3, 6], ["0.5", "1", "2", "3", "6"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(lang, loc="left", fontsize=10.5, color=t["ink"], weight="bold")
        ax.set_xlabel("Labelled hours (log scale)")
        ax.text(3.3, frozen[-1], "frozen\nprobe", color=t["ink2"], fontsize=8, va="center")
        ax.text(fx[-1], fy[-1] - max(frozen) * 0.09, "fine-tuned", color=t["ink2"], fontsize=8,
                ha="right" if len(fx) > 1 else "left")
        if key not in ft6:
            ax.text(6.0, fy[0] + max(frozen) * 0.12, "no 6 h:\ncorpus\nexhausted", color=t["muted"],
                    fontsize=7.5, ha="center", va="bottom")
        if show:
            print(f"  budget {lang:8s} frozen {[round(v,3) for v in frozen]} finetuned {dict(zip(fx,[round(v,3) for v in fy]))}")
    axes[0].set_ylabel("Character error rate")
    title(fig, t, "Fine-tuning · The limit is labelled data, not the representation",
          "Frozen best-layer probe vs full fine-tuning on the same splits. CER is script-dependent: compare within a panel only.")
    save(fig, "fig5-budget", theme)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--print", action="store_true", help="print the values drawn")
    a = ap.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    for theme, t in THEMES.items():
        show = a.print and theme == "light"
        for f in (fig_codebook_fit, fig_codebook_deficit, fig_equivalence, fig_layer_profiles, fig_budget):
            f(t, theme, show)
    print(f"wrote {len(list(OUT.glob('*.png')))} files to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
