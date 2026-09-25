# Experiment B: pre-training membership

**Question.** Across many languages, does being in the pre-training data predict how well
the codebook fits?

**Answer.** No. Over nine family-matched pairs the effect of membership is statistically
equivalent to zero, in all three context regimes.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/fig3-equivalence-dark.png">
  <img alt="Forest plot of the paired InfoNCE difference, unseen minus seen, for full clips, 6.0 s crops and 9.0 s crops. All three 90% confidence intervals lie inside the ±0.289 equivalence margin." src="assets/fig3-equivalence-light.png">
</picture>

## Design

Pre-registered in [`selection.py`](../Experiment_B_Multilingual/selection.py) before any
metric was computed.

Experiment A's XLSR-53 result was one accidental comparison: Tamil seen, Sinhala unseen.
This experiment scales it to 18 languages, paired so that language family cannot explain a
group difference. Each pair holds one language XLSR-53 saw in pre-training and one it did
not, with membership taken from the constituent corpus lists (MLS, Common Voice, BABEL)
in the model's own repository.

| Seen by XLSR-53 | Unseen | Relationship |
|---|---|---|
| Bengali | Marathi | both Indo-Aryan |
| Polish | Czech | both West Slavic |
| Persian | Greek | Indo-European, different branches |
| Estonian | Finnish | both Finnic |
| Kazakh | Uzbek | Turkic: Kipchak vs Karluk |
| Tamil | Malayalam | both South Dravidian |
| Indonesian | Javanese | Malayo-Polynesian, different branches |
| Mandarin | Burmese | Sinitic vs Burmish |
| Zulu | Xhosa | both Nguni |

| | |
|---|---|
| Data | FLEURS test split, 500 clips per language. FLEURS reads the same Flores-101 sentences in every language, removing the domain confound of Experiment A |
| Measurement | Experiment A's masked instrument, imported unchanged |
| Primary model | XLSR-53; XLS-R 0.3B, which saw 17 of the 18 languages (all but Xhosa), as a negative control |
| Unit of analysis | Language mean; the test statistic is the nine within-pair differences (unseen − seen) |
| Tests | Wilcoxon signed-rank; TOST against ±0.289 (primary) and ±0.590 (secondary) |
| Power | Minimum detectable effect at 80% power, d_z = 1.07 |

### Context regimes

Clip duration lowers InfoNCE (median per-language Spearman ρ = −0.36), and mean FLEURS
clip length differs by up to 68% across languages. So both a full-clip analysis and
fixed-window crops were pre-registered. The registered 9.0 s window discarded 4–48% of
clips per language, so a 6.0 s window, which keeps 90–100%, was added as an amendment
before any cropped result existed.

## Results

Source: [`results/final_statistics.json`](../Experiment_B_Multilingual/results/final_statistics.json);
Wilcoxon and d_z recomputed from
[`results/per_file_metrics.parquet`](../Experiment_B_Multilingual/results/per_file_metrics.parquet).

| Regime | Mean difference | d_z | 90% CI | 95% CI | Wilcoxon p | TOST p (±0.289) | Pairs favouring seen |
|---|---:|---:|---|---|---:|---:|---:|
| Full clips | +0.033 | 0.11 | [−0.160, +0.225] | [−0.206, +0.271] | 0.820 | 0.019 | 5 of 9 |
| 6.0 s crop | +0.040 | 0.21 | [−0.080, +0.160] | [−0.109, +0.189] | 0.652 | 0.0025 | 5 of 9 |
| 9.0 s crop | +0.041 | 0.20 | [−0.084, +0.167] | [−0.115, +0.197] | 0.570 | 0.0032 | 5 of 9 |

Every 90% CI lies inside the margin, and against the secondary margin every TOST p is
below 0.001. The tightest margins at which equivalence still holds at α = 0.05 are the
CI upper bounds: ±0.225, ±0.160 and ±0.167, equivalent to removing roughly 13–36% of a
language's codebook entries ([Experiment A](experiment-a.md#synthetic-codebook-deficit)).

**Negative control.** The same split on XLS-R 0.3B, which saw 17 of the 18 languages,
gives d_z = −0.01 (Wilcoxon p = 1.00), as it should.

**Dose–response.** XLS-R pre-training hours show no association with fit among the twelve
languages with 33–321 h (ρ = −0.01, p = 0.97). The weak association over the full range
is carried by six languages above 5,000 h, all European or English, and cannot be
separated from region. Source:
[`results/dose_response_check.json`](../Experiment_B_Multilingual/results/dose_response_check.json).

**Variance.** Group membership explains no more variance than chance: ω² = −0.05 for
group and −0.15 for family (a negative ω² means less than expected under the null).

## Files

| File | Role |
|---|---|
| [`build_language_lists.py`](../Experiment_B_Multilingual/build_language_lists.py) | Authoritative pre-training language lists for both checkpoints |
| [`selection.py`](../Experiment_B_Multilingual/selection.py) | Pre-registered pairs, analysis unit and crop rule |
| [`stage_fleurs.py`](../Experiment_B_Multilingual/stage_fleurs.py), [`make_crops.py`](../Experiment_B_Multilingual/make_crops.py) | Stage the FLEURS sample and build the cropped arms |
| [`presweep_check.py`](../Experiment_B_Multilingual/presweep_check.py), [`pre_run_report.py`](../Experiment_B_Multilingual/pre_run_report.py) | Balance checks before the GPU sweep |
| [`run_sweep.py`](../Experiment_B_Multilingual/run_sweep.py), [`run_crops.sh`](../Experiment_B_Multilingual/run_crops.sh) | The sweep, full and cropped |
| [`analyse_b.py`](../Experiment_B_Multilingual/analyse_b.py), [`final_stats.py`](../Experiment_B_Multilingual/final_stats.py), [`dose_check.py`](../Experiment_B_Multilingual/dose_check.py) | Paired analysis, equivalence tests, dose–response |
| [`figures_b.py`](../Experiment_B_Multilingual/figures_b.py), [`report_b.py`](../Experiment_B_Multilingual/report_b.py), [`consolidate.py`](../Experiment_B_Multilingual/consolidate.py) | Figures, report, consolidated per-file table |
