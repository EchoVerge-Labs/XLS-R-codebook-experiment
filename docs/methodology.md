# Methodology

This page explains the measurement used throughout, why the obvious alternative does not
work, and the conventions that let a null result be read as evidence rather than as a
failure to detect anything.

## The question

Self-supervised speech models such as XLS-R underperform on low-resource languages. For
wav2vec 2.0-style models, an obvious suspect is the **product quantizer**. Its codebook is
learned jointly with the encoder on pre-training data dominated by a few languages, so it
may hold no codevectors near the sounds of a scarce or absent language. If that were so,
extending or adapting the codebook would be a direct remedy.

These experiments test that premise for Sinhala and Tamil. Every claim is scoped to
product-quantized models; HuBERT-family objectives have no comparable codebook and are
not tested.

## Why the usual residual cannot be used

In XLS-R 0.3B the convolutional encoder emits `z_t ∈ R^512`. A Gumbel-softmax product
quantizer selects one of V = 320 codevectors in each of G = 2 groups, concatenated into
`q_t ∈ R^768`.

The codevectors are free parameters that are **never trained to reconstruct `z_t`**, and
the two vectors do not even share a dimension. The vector-quantization residual
`‖z_t − q_t‖²` that this question usually invokes is therefore undefined.

The natural substitute, the cosine distance between the projected context and the
projected target at *unmasked* frames, is nearly blind. It differs by only **1.11×**
between clean English and white noise, because pre-training aligns the two projections
only where the input is masked. No audio could clear a detection threshold on it.

## The instrument: masked-position contrastive fit

The measurement evaluates the pre-training objective at masked positions, the only place
pre-training aligns context and target, and therefore the only place a codebook deficit
would have to appear.

| Setting | Value |
|---|---|
| Mask span | 10 frames |
| Mask probability | 0.065 (about 6.5% of frames in the transformers implementation; about 49% under fairseq pre-training) |
| Distractors | K = 100, drawn from the masked frames of the same utterance |
| Temperature | 0.1 |
| Mask draws per file | 3 |

Two quantities are computed and aggregated per file:

- **InfoNCE**: the negative log-probability of the true target against the K distractors.
  Lower is better; chance is ln 101 ≈ 4.62.
- **Positive–negative gap**: cosine similarity to the true target minus mean similarity to
  the distractors. Larger is better.

Code-usage statistics exclude frames whose RMS is below 10% of the file's 90th-percentile
RMS, so silence cannot drive a comparison. The implementation is
[`masked_probe.py`](../Experiment_A_Diagnostic/masked_probe.py); Experiment B imports it
unchanged rather than reimplementing it.

## Positive controls

A null result only means something if the instrument can move. Five controls are derived
from the **same** English clips, so content and channel are held constant and anything
they shift is a property of the corruption:

| Control | What it does |
|---|---|
| Speed/pitch 1.45× | Resamples to 1.45× the samples, slowing the clip and lowering its pitch together (hence its longer mean duration) |
| Six-way babble | The clip plus five other English clips, renormalised to its RMS level |
| Time reversal | Plays the clip backwards |
| Sine chirps | Four random linear chirps at the clip's RMS level: structured, but not speech |
| White noise | Gaussian noise at the clip's RMS level |

These corrupt the *audio*. A post-hoc test corrupts the *codebook* instead, by forbidding
entries at inference; see [Experiment A](experiment-a.md#synthetic-codebook-deficit).

## Equivalence testing

A non-significant test is not evidence of no effect. Absence claims here use **two
one-sided tests (TOST)**, which ask whether the whole 90% confidence interval lies inside a
margin chosen in advance.

The primary margin is **±0.289** InfoNCE, the increase caused by the mildest positive
control (speed/pitch 1.45×). The secondary margin is ±0.590, from time reversal. The
synthetic codebook-deficit test translates the primary margin into codebook terms: on
XLSR-53 it corresponds to removing 27–43% of the entries a language uses.

The margin is a calibrated reference point rather than a formally matched control. It is
measured on unpaired, cross-corpus controls on XLS-R 0.3B and applied to paired
differences on XLSR-53, where the matching control would give a more lenient 0.327.

## Pre-registration and amendments

Each experiment's design, comparisons and decision criteria were committed to the
repository before any result was computed:

| Experiment | Where the pre-registration lives |
|---|---|
| A | Arms in [`stage_a_diagnostic.py`](../Experiment_A_Diagnostic/stage_a_diagnostic.py); the original decision thresholds (residual ratio above 1.1× moderate, above 1.5× strong evidence of a gap) in [`analyse.py`](../Experiment_A_Diagnostic/analyse.py) |
| B | [`selection.py`](../Experiment_B_Multilingual/selection.py) (pairs, analysis unit, crop rule) |
| C | [`config.py`](../Experiment_C_LayerProbe/config.py) (budgets, probe hyperparameters, verdict criteria) |

Experiment A's thresholds were set on the residual. When the residual proved insensitive,
the masked-position instrument replaced it, validated against the same positive controls
rather than against a threshold chosen after the fact. The post-hoc codebook-deficit test
states its own decision rules, R1–R3, in the docstring of
[`codebook_deficit.py`](../Experiment_A_Diagnostic/codebook_deficit.py), fixed before
that script produced any result.

When a pre-registered choice had to change, the change is recorded in the same file as an
**AMENDMENT** block, with the evidence that forced it, rather than edited silently. Git
history timestamps both the original and the amendment.

Analyses added after the fact are labelled **post-hoc** in their script docstrings, commit
messages and these docs.
