# Data

Every corpus and checkpoint used, its licence, and how it enters each experiment. Audio is
never committed to this repository; the scripts fetch or stage it and regenerate every
derived file.

## Corpora

| Corpus | Used in | Content | Licence |
|---|---|---|---|
| [OpenSLR-30](https://openslr.org/30/), Sinhala TTS (Google) | A: read Sinhala | Studio-quality multi-speaker read Sinhala | CC BY-SA 4.0 |
| [OpenSLR-52](https://openslr.org/52/), Large Sinhala ASR training data set (Google) | C: Sinhala | Crowd-sourced read Sinhala, 185,293 utterances from 478 speakers | CC BY-SA 4.0 |
| [OpenSLR-65](https://openslr.org/65/), Crowdsourced high-quality Tamil multi-speaker speech (Google) | A: read Tamil; C: Tamil | Read Tamil, 4,291 utterances from 50 speakers | CC BY-SA 4.0 |
| [Common Voice 17](https://commonvoice.mozilla.org/) (Mozilla) | A: English; C: English and the Common Voice Tamil arm | Read prompted sentences, self-recorded by contributors | CC0 1.0 |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) (Google) | B: all 18 languages, plus English | The same Flores-101 sentences read in every language; test split | CC BY 4.0 |
| In-the-wild Sinhala and Tamil (YouTube) | A: YouTube arms | Publicly available videos, collected by the team | Not redistributed |

The YouTube audio is not redistributed in any form. Only derived statistics (InfoNCE, gap
and code-usage counts) are released.

## Checkpoints

| Checkpoint | Role | Licence |
|---|---|---|
| [`facebook/wav2vec2-xls-r-300m`](https://huggingface.co/facebook/wav2vec2-xls-r-300m), XLS-R 0.3B | Primary model in A and C; negative control in B | Apache 2.0 |
| [`facebook/wav2vec2-large-xlsr-53`](https://huggingface.co/facebook/wav2vec2-large-xlsr-53), XLSR-53 | Secondary model in A; primary model in B | Apache 2.0 |

Pre-training language membership for both checkpoints is taken from primary sources, the
XLS-R paper's language table and the fairseq repository's corpus lists, and recorded in
[`xlsr_300m_languages.json`](../Experiment_B_Multilingual/results/xlsr_300m_languages.json) and
[`xlsr53_languages.json`](../Experiment_B_Multilingual/results/xlsr53_languages.json).
XLSR-53 includes Tamil but not Sinhala; XLS-R includes both.

## Sampling

All sampling is deterministic, so a rerun draws the same clips:

- **Experiment A** draws 1,000 clips per arm with `random.Random(1234)` over the sorted file
  list ([`sample_and_stage.py`](../Experiment_A_Diagnostic/sample_and_stage.py)) and truncates
  each to 30 s. [`build_read_manifest.py`](../Experiment_A_Diagnostic/build_read_manifest.py)
  rebuilds the read arms from the public sources for machines without the team's storage
  mount.
- **Experiment B** stages 500 test clips per language from FLEURS
  ([`stage_fleurs.py`](../Experiment_B_Multilingual/stage_fleurs.py)).
- **Experiment C** builds exact 3.00 h / 0.50 h splits from 40 speakers per language,
  speaker-disjoint for the ASR probe, with the smallest speakers held out for testing
  ([`build_splits.py`](../Experiment_C_LayerProbe/build_splits.py)). Transcripts are NFC
  normalised and lowercased, and clips over 20 s are excluded.

## What is and is not versioned

| Versioned | Not versioned |
|---|---|
| Every script, and the pre-registered configurations | Audio corpora and staged copies (`**/data/`) |
| Summary results: JSON, CSV and the consolidated Experiment B parquet | Experiment C's layer-feature cache, about 150 GB with the random-init floor (`**/cache/`) |
| Figures | Per-frame arrays (`*.npz`, `*.npy`) and model weights |
| | Manifests that record absolute paths on the machine that ran them |

The rules are in [`.gitignore`](../.gitignore), which whitelists the experiment folders
rather than ignoring files one by one.
