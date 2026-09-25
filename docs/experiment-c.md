# Experiment C: layer-wise probing

**Question.** If the codebook is not the bottleneck, does a low-resource penalty appear at
some depth in the transformer?

**Answer.** No depth-localised penalty. All three languages share the same depth profile,
none of the probes has saturated at 3 h, and fine-tuning more than halves their error,
still improving at 6 h. The one language-specific signal, probe instability at the best
layers, could not be separated from corpus.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/fig4-layer-profiles-dark.png">
  <img alt="CTC character error rate by transformer layer, normalised to each language's best layer. All three languages bottom out at layers 15 to 17; a randomly initialised model shows no depth structure." src="assets/fig4-layer-profiles-light.png">
</picture>

## Design

Pre-registered in [`config.py`](../Experiment_C_LayerProbe/config.py), including the
verdict criteria, before any probe was trained.

| | |
|---|---|
| Model | Frozen XLS-R 0.3B, all 24 transformer layers |
| Data | Sinhala (OpenSLR-52), Tamil (OpenSLR-65), English (Common Voice 17) |
| Budget | Exactly 3.00 h train and 0.50 h test per language, 40 speakers (30 train, 10 held out) |
| ASR probe | One linear layer with CTC over characters; lr 1e-2, batch 32, 25 epochs, AdamW; speaker-disjoint split |
| Speaker probe | One linear classifier on mean-pooled features, 3 training utterances per speaker |
| Seeds | 3 per configuration |
| Floor control | The same probes on a randomly initialised copy of the model |

Both probes are deliberately a single linear layer: a deeper probe would measure its own
capacity rather than the representation's.

Character error rate (CER) depends on the script, and the three inventories are 106, 49
and 34 characters. So only **curve shape** is compared across languages. Each language's
CER is normalised by its own best layer, and the pre-registered criterion for a
depth-localised penalty was a shift from English of at least 3 layers in the best layer,
or 2 in the inverse-CER-weighted centroid.

A minority of CTC probe initialisations converge to degenerate solutions. They are
detected from training loss alone (final-to-initial loss ratio above three times the best
ratio in the same configuration) and replaced with fresh seeds.

## Results

Source: [`results/probe_results_xlsr300m.json`](../Experiment_C_LayerProbe/results/probe_results_xlsr300m.json)
and [`results/analysis_xlsr300m.json`](../Experiment_C_LayerProbe/results/analysis_xlsr300m.json).

| | Sinhala | Tamil | English |
|---|---:|---:|---:|
| Best CTC layer | 15 | 17 | 15 |
| Shift from English | 0 | 2 | — |
| Depth centroid | 11.69 | 11.61 | 12.38 |
| Centroid shift from English | 0.68 | 0.77 | — |
| Best-layer CER | 0.273 | 0.170 | 0.219 |
| Weighted sum of all layers, CER | 0.260 | 0.164 | 0.211 |
| Best speaker-ID layer (accuracy) | 4 (0.878) | 4 (0.914) | 4 (0.960) |
| Pre-registered verdict | uniform | uniform | — |

**Validation.** The expected profile appears in every language: speaker identity peaks
early (layer 4), character information in the middle (layers 15–17), matching prior
layer-wise analyses. Probes on the randomly initialised model stay at a best-layer CER of
0.76–0.85, with no depth structure.

**Speaker accuracy**, the one metric comparable across languages, is lower for Sinhala and
Tamil than English. The gap is present at layer 0 and does not grow with depth
(ρ ≈ 0), pointing to the corpora rather than the representation: Common Voice
contributors record on their own devices, which can make speakers separable by channel.

### Probe stability

Probe stability is the one place the languages differ. Across nine initialisations at
eight layers, CTC probes collapsed in 12 of 144 runs for Sinhala and Tamil and in none of
72 for English (Fisher's exact test, p = 0.010). Source:
[`results/collapse_rate.json`](../Experiment_C_LayerProbe/results/collapse_rate.json).

Collapses concentrated at layers 15 and 17: 22% of 36 runs there, against 4% of the other
108 (p = 0.002). That comparison was chosen after the best layers were known and is
uncorrected. Discarding collapsed runs moved CER at those layers by at most 0.003, so the
shape comparison is unaffected.

Because English is Common Voice while Sinhala and Tamil are OpenSLR, language and corpus
are confounded. A Common Voice Tamil arm was added to separate them. It collapsed in 4 of
72 runs, statistically indistinguishable from both OpenSLR Tamil (p = 0.53) and Common
Voice English (p = 0.12), so the question remains open. Source:
[`results/collapse_rate_cv_tamil.json`](../Experiment_C_LayerProbe/results/collapse_rate_cv_tamil.json).

## Post-hoc: fine-tuning at 3 h and 6 h

Added after review and not pre-registered.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/fig5-budget-dark.png">
  <img alt="Character error rate against labelled hours for each language, frozen probe versus fine-tuned model. Fine-tuning roughly halves the frozen-probe error at 3 h, and Sinhala and English keep improving at 6 h." src="assets/fig5-budget-light.png">
</picture>

Frozen probes bound what a linear head can read from a representation, not what the model
achieves once adapted. The fine-tuned arm trains the same checkpoint with a CTC head on the
same splits, vocabularies and seeds, freezing only the CNN feature encoder (lr 1e-4,
batch 8, 30 epochs). Fine-tuning discards the codevectors, so it does not re-test the
codebook; it asks whether the probes' within-language conclusions survive adaptation.

Sources: [`results/finetune.json`](../Experiment_C_LayerProbe/results/finetune.json)
and [`results/finetune_6h.json`](../Experiment_C_LayerProbe/results/finetune_6h.json).

| | Sinhala | Tamil | English |
|---|---:|---:|---:|
| Frozen best layer, 3 h | 0.275 | 0.170 | 0.221 |
| Fine-tuned, 3 h (mean ± SD, 3 seeds) | 0.129 ± 0.006 | 0.062 ± 0.003 | 0.093 ± 0.003 |
| Relative reduction | 53% | 64% | 58% |
| Fine-tuned, 6 h | 0.108 ± 0.002 | — | 0.081 ± 0.002 |
| Gain over fine-tuned 3 h | 16% | — | 14% |

The frozen values here come from the data-scaling run, whose 3 h figures differ from the
layer sweep by at most 0.002.

At 3 h the frozen representation is not the binding constraint for any language. At 6 h,
with the same speakers and test sets and only more audio per speaker, both curves keep
falling, and the seed ranges at 6 h are disjoint from those at 3 h. The low-resource
language gains slightly more. Tamil has no 6 h point because OpenSLR-65's 30 training
speakers hold only 3.48 h; beyond 6 h is untested.

## Files

| File | Role |
|---|---|
| [`config.py`](../Experiment_C_LayerProbe/config.py) | Pre-registered configuration and verdict criteria |
| [`stage_si_ta.py`](../Experiment_C_LayerProbe/stage_si_ta.py), [`fetch_english.py`](../Experiment_C_LayerProbe/fetch_english.py), [`fetch_cv_tamil.py`](../Experiment_C_LayerProbe/fetch_cv_tamil.py) | Stage the three corpora and the Common Voice Tamil arm |
| [`build_splits.py`](../Experiment_C_LayerProbe/build_splits.py), [`build_cv_tamil_split.py`](../Experiment_C_LayerProbe/build_cv_tamil_split.py) | Budget-exact, speaker-disjoint splits |
| [`extract_features.py`](../Experiment_C_LayerProbe/extract_features.py) | One-pass extraction of all 24 layers to an fp16 cache |
| [`probes.py`](../Experiment_C_LayerProbe/probes.py), [`run_probes.py`](../Experiment_C_LayerProbe/run_probes.py) | Linear probes, weighted sum, data scaling |
| [`analyse_c.py`](../Experiment_C_LayerProbe/analyse_c.py) | Curve shape, pre-registered verdict, figures |
| [`collapse_rate.py`](../Experiment_C_LayerProbe/collapse_rate.py), [`fig_collapse.py`](../Experiment_C_LayerProbe/fig_collapse.py) | Collapse rate as its own data series |
| [`finetune_ctc.py`](../Experiment_C_LayerProbe/finetune_ctc.py), [`build_6h_split.py`](../Experiment_C_LayerProbe/build_6h_split.py), [`run_6h_arm.sh`](../Experiment_C_LayerProbe/run_6h_arm.sh) | Post-hoc fine-tuning arms |
