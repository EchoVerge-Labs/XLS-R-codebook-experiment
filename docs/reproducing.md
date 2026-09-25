# Reproducing the results

## Hardware

All three experiments ran on an **NVIDIA DGX Spark** (GB10, ARM64, CUDA 13.0). A GPU is
required for inference in A and B and for feature extraction, probing and fine-tuning in C.
Disk is the main constraint: Experiment C's feature cache is about 150 GB.

## Environment

The versions that produced the committed results:

| Package | Version |
|---|---|
| Python | 3.12.3 |
| torch / torchaudio | 2.14.0 / 2.11.0 (CUDA 13.0 builds) |
| transformers | 5.16.1 |
| numpy, scipy | 2.5.2, 1.18.1 |
| pandas, pyarrow | 3.0.5, 25.0.1 |
| matplotlib, soundfile | 3.11.1, 0.14.0 |

```bash
python -m venv ~/venv
~/venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
~/venv/bin/pip install transformers soundfile numpy scipy pandas pyarrow matplotlib
```

> **ARM64:** the `cu121` wheels commonly cited do not exist for aarch64, and would not
> support GB10 (sm_121) if they did. Use the `cu130` index.

## Paths

Every script resolves its own folder from its location, so the repository can be cloned
anywhere and each script run from any working directory. Experiment B imports
Experiment A's measurement code from the sibling folder, so keep the three experiment
folders side by side.

Scripts that read the team's storage mount (`inventory.py`, `sample_and_stage.py`,
`stage_si_ta.py`, `build_6h_split.py`) expect it at `~/google-drive/Community Datasets`
(and the YouTube corpora at `~/google-drive/Pre Processed Data`). Without the mount,
`build_read_manifest.py` rebuilds Experiment A's read arms from public sources.

## Experiment A

Run from `Experiment_A_Diagnostic/`.

```bash
python inventory.py                  # index the YouTube corpora on the storage mount
python fetch_english.py              # stream Common Voice 17 English (dev) clips
python sample_and_stage.py           # deterministic 1,000-clip sample per arm
python make_controls.py              # five positive controls, appended to the manifest
python stage_a_diagnostic.py         # frame-level diagnostic (unmasked residual, codes)
python masked_probe.py               # the masked-position instrument
python analyse.py && python ladder.py && python fig_ladder.py
```

For XLSR-53, rerun the diagnostic, probe and analysis with the checkpoint and output
directory switched:

```bash
export XLSR_MODEL=facebook/wav2vec2-large-xlsr-53 XLSR_RESULTS=$PWD/results_xlsr53
python stage_a_diagnostic.py && python masked_probe.py && python analyse.py
```

Post-hoc analyses:

```bash
python code_overlap.py               # reads the stored code assignments; CPU only
python build_read_manifest.py        # read-arm manifest the deficit test uses
for m in xlsr300m xlsr53; do for p in 0.065 0.65; do
  python codebook_deficit.py --model $m --mask-prob $p
done; done
```

## Experiment B

Run from `Experiment_B_Multilingual/`.

```bash
python build_language_lists.py       # pre-training membership for both checkpoints
python stage_fleurs.py               # 500 FLEURS test clips per language
python presweep_check.py && python pre_run_report.py
python run_sweep.py                  # full clips, both checkpoints
python make_crops.py 6.0 && python make_crops.py 9.0
bash run_crops.sh                    # 6.0 s (primary) then 9.0 s (sensitivity)
python analyse_b.py && python final_stats.py && python dose_check.py
python consolidate.py && python figures_b.py && python report_b.py
```

## Experiment C

Run from `Experiment_C_LayerProbe/`.

```bash
python stage_si_ta.py && python fetch_english.py && python build_splits.py
python extract_features.py                                   # all 24 layers, fp16 cache
python extract_features.py --tag floor --random-init --max-hours 1
python run_probes.py                                         # layers, weighted sum, scaling
python run_probes.py --tag floor --stages layers
python analyse_c.py
python collapse_rate.py

# Common Voice Tamil arm (language vs corpus)
python fetch_cv_tamil.py && python build_cv_tamil_split.py
python extract_features.py --langs tamil_cv
python collapse_rate.py --langs tamil_cv --out results/collapse_rate_cv_tamil.json
```

Post-hoc fine-tuning. Each 3 h run takes about 50 minutes on GB10 and each 6 h run about
100–220 minutes, so run these in `tmux` or `screen`:

```bash
python finetune_ctc.py --langs sinhala tamil english --seeds 0 1 2 --epochs 30
python build_6h_split.py
python finetune_ctc.py --langs sinhala english --seeds 0 1 2 --epochs 30 \
    --split-suffix 6h --tag finetune_6h
```

`finetune_ctc.py` skips any (language, seed) cell already present in its output file, so
an interrupted sweep resumes where it stopped.

## Documentation figures

```bash
python docs/scripts/make_figures.py --print
```

This reads only versioned result files and so runs on a fresh clone without any data or
GPU. It writes light and dark versions of every figure to `docs/assets/`.
