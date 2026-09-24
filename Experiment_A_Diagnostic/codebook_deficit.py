#!/usr/bin/env python
"""
Synthetic codebook deficit: can masked InfoNCE see a codebook that lacks units?

POST-HOC. Added after the pre-registered diagnostic. The positive controls of
masked_probe.py corrupt the AUDIO; they show the masked metric detects degraded speech, not a degraded CODEBOOK. Here
the audio is untouched and the codebook is damaged instead: at inference, chosen entries
are forbidden (their quantizer logits set to -inf), so every frame whose best code was
removed falls back to its next-best available code. That is the literal form of the
hypothesis under test - "no codevector near this language's sounds" - produced on demand.

Only the quantizer changes. The context network reads unquantized features, so the
prediction project_hid(c_t) is identical across conditions; what moves is the target
project_q(q_t) and, through it, the distractors. Masks, distractor indices and clips are
the same in every condition (seed = 1000*i + k, as in masked_probe.py), so every
comparison is paired per file.

Conditions, per language (read arms only: English_CV, Sinhala_read, Tamil_read):
  base           no entries removed; must reproduce masked_probe.py (checks the sample
                 and code)
  rand_F_sS      F in {10,25,50}% of the 320 entries per group, chosen at random,
                 3 draws; identical sets for every language and model
  top_F          F in {5,10,25,50}% of the entries THIS language uses (gated frames),
                 most-used first: the worst case for the language itself
  xta_top_F      English only: F in {10,25,50}% of the entries TAMIL uses, most-used
                 first

Decision rules, fixed before any result of this script existed:
  R1  Sensitivity. The metric is taken to see a codebook deficit if, on English with
      XLS-R 0.3B, top_10 raises per-file InfoNCE by more than 0.041 (the largest paired
      language difference in Experiment B) with a 95% bootstrap CI excluding zero.
  R2  Margin in codebook units. For each language and model we report the fraction of
      its used entries (top_F) and of all entries (rand_F), linearly interpolated, at
      which the InfoNCE increase reaches the Experiment B margin of 0.289. ">50%" if never.
  R3  Null. If top_50 on English raises InfoNCE by less than 0.041, the metric cannot
      distinguish even a severe codebook deficit, and the conclusions of Experiments A
      and B about codebook adequacy must be withdrawn, not reworded.
  Random removal is reported but is not decisive: about 70% of entries are unused by any
  language, so a random block mostly removes codes nobody needs.

--mask-prob 0.065 is the protocol of masked_probe.py. Note that transformers
reads mask_time_prob as the fraction of frames masked, whereas fairseq's 0.065 is the
probability of a span START (~49% of frames masked). --mask-prob 0.65 reproduces the
fairseq pre-training density and is run as a sensitivity check.
"""
import os, json, argparse, warnings, time
import numpy as np, torch, soundfile as sf, torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    _compute_mask_indices, _sample_negative_indices)

warnings.filterwarnings("ignore")
ROOT = os.path.expanduser("~/EchoVerge-LABS/Experiment_A_Diagnostic")
REPO = os.path.dirname(os.path.abspath(__file__))
MODELS = {"xlsr300m": "facebook/wav2vec2-xls-r-300m",      # XLS-R 0.3B (Experiment A)
          "xlsr53": "facebook/wav2vec2-large-xlsr-53"}      # XLSR-53 (Experiment B)
SR, MASK_SPAN, N_NEG, TEMP, MAX_S = 16000, 10, 100, 0.1, 30.0
SPEECH_Q, SPEECH_FRAC = 0.90, 0.10          # identical gate to analyse.py / code_overlap.py
LANGS = ["English_CV", "Sinhala_read", "Tamil_read"]
STUDY2_MAX_DIFF, MARGIN = 0.041, 0.289
RAND_F, TOP_F, XTA_F, RAND_SEEDS = (0.10, 0.25, 0.50), (0.05, 0.10, 0.25, 0.50), \
    (0.10, 0.25, 0.50), 3
CHUNK = 4096


def load16k(p):
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    t = torch.from_numpy(x.mean(1))
    if sr != SR:
        t = torchaudio.functional.resample(t, sr, SR)
    return t[: int(MAX_S * SR)]


def frame_rms(wave, n_frames, stride=320):
    n = n_frames * stride
    w = torch.nn.functional.pad(wave, (0, max(0, n - wave.numel())))[:n]
    return w.view(n_frames, stride).pow(2).mean(dim=1).sqrt()


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16)


@torch.no_grad()
def quantize(model, ext, block=None):
    """Hard (eval-mode) Gumbel quantization with forbidden entries -> project_q(q), codes."""
    q = model.quantizer
    G, V = q.num_groups, q.num_vars
    with autocast():
        lg = q.weight_proj(ext)
    lg = lg.float().view(-1, G, V)
    if block is not None:
        lg = lg.masked_fill(block, float("-inf"))
    idx = lg.argmax(-1)                                            # (n,G)
    cv = q.codevectors[0].view(G, V, -1)
    qv = torch.cat([cv[g, idx[:, g]] for g in range(G)], -1)
    with autocast():
        qp = model.project_q(qv.to(model.project_q.weight.dtype))
    return qp.float(), idx


@torch.no_grad()
def forward_file(model, fe, path, dev, seed, mask_prob):
    """One mask draw, exactly as masked_probe.probe_file. Returns masked rows only."""
    w = load16k(path)
    if w.numel() < SR // 2:
        raise ValueError("too short")
    iv = fe(w.numpy(), sampling_rate=SR, return_tensors="pt").input_values.to(dev)
    T = int(model._get_feat_extract_output_lengths(iv.shape[-1]))
    if T < MASK_SPAN * 3:
        raise ValueError("too few frames")
    np.random.seed(seed)
    mask = _compute_mask_indices((1, T), mask_prob, MASK_SPAN, min_masks=2)
    if mask.sum() < 2:
        raise ValueError("empty mask")
    neg = _sample_negative_indices((1, T), N_NEG, mask)
    mask_t = torch.tensor(mask, dtype=torch.bool, device=dev)
    with autocast():
        base = model.wav2vec2(iv, mask_time_indices=mask_t)
        hid = model.project_hid(base.last_hidden_state)
    hid, ext = hid.float()[0], base.extract_features.float()[0]
    m = mask[0]
    local = np.cumsum(m) - 1                                       # absolute -> masked row
    negm = neg[0][m]                                               # (n_masked, N_NEG)
    assert m[negm].all(), "distractor outside the masked set"
    return dict(hid=hid[mask_t[0]], ext_m=ext[mask_t[0]], neg=torch.from_numpy(local[negm]),
                ext_all=ext, iv=iv[0].float().cpu(), T=T)


@torch.no_grad()
def check_reproduces_model(model, fe, path, dev):
    """quantize() must equal the model's own projected_quantized_states."""
    w = load16k(path)
    iv = fe(w.numpy(), sampling_rate=SR, return_tensors="pt").input_values.to(dev)
    with autocast():
        out = model(iv)
        ext = model.wav2vec2(iv).extract_features
    mine, _ = quantize(model, ext.float()[0])
    err = (mine - out.projected_quantized_states.float()[0]).abs().max().item()
    assert err < 1e-2, f"quantize() differs from the model by {err}"
    return err


@torch.no_grad()
def collect(model, fe, recs, dev, mask_prob, repeats):
    """Forward every file once per mask draw; keep masked rows and gated codes."""
    HID, EXT, NEG, FID, codes, fails, off = [], [], [], [], [], 0, 0
    for i, r in enumerate(recs):
        for k in range(repeats):
            try:
                o = forward_file(model, fe, r["path"], dev, 1000 * i + k, mask_prob)
            except Exception:
                fails += 1; continue
            n = o["hid"].shape[0]
            HID.append(o["hid"]); EXT.append(o["ext_m"]); NEG.append(o["neg"] + off)
            FID.append(torch.full((n,), i, dtype=torch.long)); off += n
            if k == 0:                                             # usage: all gated frames
                _, idx = quantize(model, o["ext_all"])
                rms = frame_rms(o["iv"], o["T"]).numpy()
                keep = rms > SPEECH_FRAC * np.quantile(rms, SPEECH_Q)
                codes.append(idx.cpu().numpy()[keep])
    return dict(hid=torch.cat(HID), ext=torch.cat(EXT), neg=torch.cat(NEG).to(dev),
                fid=torch.cat(FID).numpy(), codes=np.concatenate(codes), fails=fails)


@torch.no_grad()
def evaluate(model, D, block):
    """Frame-level InfoNCE, gap, strict accuracy and tie rate under one blocked set."""
    qp = torch.cat([quantize(model, D["ext"][s:s + CHUNK], block)[0]
                    for s in range(0, len(D["ext"]), CHUNK)])
    nce, gap, acc, tie = [], [], [], []
    for s in range(0, len(qp), CHUNK):
        h = D["hid"][s:s + CHUNK]
        tgt = torch.cat([qp[s:s + CHUNK].unsqueeze(1), qp[D["neg"][s:s + CHUNK]]], 1)
        sims = torch.cosine_similarity(h.unsqueeze(1), tgt, dim=-1)       # (n,1+N)
        logits = sims / TEMP
        nce.append(torch.nn.functional.cross_entropy(
            logits, torch.zeros(len(h), dtype=torch.long, device=h.device), reduction="none"))
        gap.append(sims[:, 0] - sims[:, 1:].mean(1))
        mx = logits[:, 1:].max(1).values
        acc.append((logits[:, 0] > mx).float()); tie.append((logits[:, 0] == mx).float())
    return {k: torch.cat(v).cpu().numpy() for k, v in
            dict(nce=nce, gap=gap, acc=acc, tie=tie).items()}


def per_file(x, fid):
    u, inv = np.unique(fid, return_inverse=True)
    return np.bincount(inv, weights=x) / np.bincount(inv)


def boot_ci(d, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (reps, len(d)))].mean(1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


def usage(codes, G, V):
    return np.stack([np.bincount(codes[:, g], minlength=V) for g in range(G)])


def top_block(cnt, f):
    """Most-used f of the entries this distribution uses, per group."""
    B = np.zeros(cnt.shape, bool)
    for g in range(cnt.shape[0]):
        used = np.where(cnt[g] > 0)[0]
        k = int(round(f * len(used)))
        B[g, used[np.argsort(-cnt[g, used], kind="stable")[:k]]] = True
    return B


def rand_block(f, s, G, V):
    rng = np.random.default_rng(10_000 + 100 * s + int(round(100 * f)))
    B = np.zeros((G, V), bool)
    for g in range(G):
        B[g, rng.choice(V, int(round(f * V)), replace=False)] = True
    return B


def crossing(levels):
    """Linear interpolation of the fraction at which the increase reaches MARGIN."""
    pts = [(0.0, 0.0)] + sorted(levels)
    for (f0, d0), (f1, d1) in zip(pts, pts[1:]):
        if d1 >= MARGIN > d0:
            return f0 + (MARGIN - d0) * (f1 - f0) / (d1 - d0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="xlsr300m")
    ap.add_argument("--mask-prob", type=float, default=0.065)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--manifest", default=f"{ROOT}/results/read_manifest.json")
    ap.add_argument("--out", default=f"{REPO}/results/codebook_deficit")
    a = ap.parse_args()

    dev, t0 = "cuda", time.time()
    tag = f"{a.model}_mask{a.mask_prob:g}" + (f"_limit{a.limit}" if a.limit else "")
    os.makedirs(a.out, exist_ok=True)
    man = json.load(open(a.manifest))
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODELS[a.model])
    model = Wav2Vec2ForPreTraining.from_pretrained(MODELS[a.model]).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    G, V = model.quantizer.num_groups, model.quantizer.num_vars
    err = check_reproduces_model(model, fe, man["English_CV"][0]["path"], dev)
    print(f"[{tag}] G={G} V={V}; quantize() matches model to {err:.2e}", flush=True)

    data = {}
    for L in LANGS:
        recs = man[L][: a.limit] if a.limit else man[L]
        data[L] = collect(model, fe, recs, dev, a.mask_prob, a.repeats)
        print(f"  collected {L:<13} files={len(np.unique(data[L]['fid']))} "
              f"masked={len(data[L]['fid']):,} gated={len(data[L]['codes']):,} "
              f"fails={data[L]['fails']}  ({time.time()-t0:.0f}s)", flush=True)
    cnt = {L: usage(data[L]["codes"], G, V) for L in LANGS}

    out = dict(config=dict(model=MODELS[a.model], mask_prob=a.mask_prob, repeats=a.repeats,
                           limit=a.limit, n_neg=N_NEG, temp=TEMP, span=MASK_SPAN,
                           margin=MARGIN, study2_max_diff=STUDY2_MAX_DIFF,
                           quantize_check_err=err),
               languages={})
    perfile = {}
    for L in LANGS:
        D = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in data[L].items()}
        conds = {"base": None}
        conds.update({f"rand_{int(f*100)}_s{s}": rand_block(f, s, G, V)
                      for f in RAND_F for s in range(RAND_SEEDS)})
        conds.update({f"top_{int(f*100)}": top_block(cnt[L], f) for f in TOP_F})
        if L == "English_CV":
            conds.update({f"xta_top_{int(f*100)}": top_block(cnt["Tamil_read"], f) for f in XTA_F})
        res, base_pf = {}, None
        used = cnt[L] > 0
        for name, B in conds.items():
            bt = None if B is None else torch.tensor(B, device=dev)
            r = evaluate(model, D, bt)
            pf = per_file(r["nce"], D["fid"])
            if base_pf is None:
                base_pf = pf
            d = pf - base_pf
            lo, hi = boot_ci(d)
            mass = 0.0 if B is None else float(np.mean(
                [cnt[L][g][B[g]].sum() / cnt[L][g].sum() for g in range(G)]))
            res[name] = dict(
                nce_mean=float(r["nce"].mean()), gap_mean=float(r["gap"].mean()),
                acc_mean=float(r["acc"].mean()), tie_rate=float(r["tie"].mean()),
                delta_nce_perfile=float(d.mean()), delta_ci95=[lo, hi],
                entries_blocked_per_group=0 if B is None else int(B.sum() / G),
                used_entries_blocked_frac=0.0 if B is None else float((B & used).sum() / used.sum()),
                frame_mass_blocked=mass)
            perfile[f"{L}/{name}"] = pf
            print(f"    {L:<13} {name:<12} NCE={res[name]['nce_mean']:.3f} "
                  f"dNCE={d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] mass={mass:.2f}", flush=True)
        rnd = {}
        for f in RAND_F:
            ds = [res[f"rand_{int(f*100)}_s{s}"]["delta_nce_perfile"] for s in range(RAND_SEEDS)]
            rnd[f] = float(np.mean(ds))
        out["languages"][L] = dict(
            n_files=int(len(base_pf)), masked_frames=int(len(D["fid"])), fails=data[L]["fails"],
            used_entries_per_group=[int(x) for x in used.sum(1)], conditions=res,
            margin_crossing=dict(
                top_frac_of_used=crossing([(f, res[f"top_{int(f*100)}"]["delta_nce_perfile"])
                                           for f in TOP_F]),
                rand_frac_of_all=crossing(list(rnd.items()))))
        del D; torch.cuda.empty_cache()

    en = out["languages"]["English_CV"]["conditions"]
    t10, t50 = en["top_10"], en["top_50"]
    out["decision"] = dict(
        R1_sensitive=bool(t10["delta_nce_perfile"] > STUDY2_MAX_DIFF and t10["delta_ci95"][0] > 0),
        R3_null=bool(t50["delta_nce_perfile"] < STUDY2_MAX_DIFF),
        note="R1/R3 are defined on XLS-R 0.3B at mask_prob 0.065; other runs are supporting.")
    path = f"{a.out}/codebook_deficit_{tag}.json"
    json.dump(out, open(path, "w"), indent=1)
    np.savez_compressed(f"{a.out}/codebook_deficit_{tag}_perfile.npz", **perfile)
    print(f"\n[{tag}] decision {out['decision']}\n-> {path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
