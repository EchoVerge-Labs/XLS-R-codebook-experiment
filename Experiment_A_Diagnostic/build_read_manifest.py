#!/usr/bin/env python
"""Rebuild the read-arm sample of the diagnostic (English_CV, Sinhala_read, Tamil_read)
from the public sources, for machines without the Drive mount.

sample_and_stage.py drew each arm with random.Random(1234).sample over the SORTED list of
file paths, so the same file set under directories that sort the same way reproduces the
same clips in the same order. The layout below mirrors the Drive one:
    OpenSLR30_Sinhala/audio/*.wav
    OpenSLR65_Tamil/audio/{female,male}/*.wav
English is streamed from the same Common Voice 17 tar as fetch_english.py (first 1,200).

Records keep the manifest shape masked_probe.py expects, and the list order it uses to
seed each file's mask draws (seed = 1000*i + k). Whether the sample is truly identical is
checked downstream: the unblocked InfoNCE must reproduce masked_probe.py.
"""
import os, json, random, tarfile, zipfile, urllib.request

ROOT = os.path.expanduser("~/EchoVerge-LABS/Experiment_A_Diagnostic")
DATA = f"{ROOT}/data"
SLR = f"{DATA}/openslr"
SEED, N_PER_LANG = 1234, 1000
CV_URL = ("https://huggingface.co/datasets/fsicoli/common_voice_17_0/resolve/main/"
          "audio/en/dev/en_dev_0.tar")


def fetch_english_stream(out=f"{DATA}/english_cv", target=1200):
    os.makedirs(out, exist_ok=True)
    have = [f for f in os.listdir(out) if f.endswith(".mp3")]
    if len(have) >= target:
        print("english: present", len(have)); return
    req = urllib.request.Request(CV_URL, headers={"User-Agent": "curl/8"})
    n = 0
    with urllib.request.urlopen(req, timeout=120) as r:
        tf = tarfile.open(fileobj=r, mode="r|")
        for m in tf:
            if not m.isfile():
                continue
            name = os.path.basename(m.name)
            if not name.lower().endswith((".mp3", ".wav")):
                continue
            open(os.path.join(out, name), "wb").write(tf.extractfile(m).read())
            n += 1
            if n >= target:
                break
    print("english: extracted", n)


def unpack_sinhala():
    dst = f"{SLR}/OpenSLR30_Sinhala/audio"
    if os.path.isdir(dst) and os.listdir(dst):
        return dst
    os.makedirs(dst, exist_ok=True)
    with tarfile.open(f"{SLR}/si_lk.tar.gz") as tf:
        for m in tf:
            if m.isfile() and m.name.lower().endswith(".wav"):
                open(os.path.join(dst, os.path.basename(m.name)), "wb").write(tf.extractfile(m).read())
    return dst


def unpack_tamil():
    base = f"{SLR}/OpenSLR65_Tamil/audio"
    for sex in ("female", "male"):
        dst = f"{base}/{sex}"
        if os.path.isdir(dst) and os.listdir(dst):
            continue
        os.makedirs(dst, exist_ok=True)
        with zipfile.ZipFile(f"{SLR}/ta_in_{sex}.zip") as z:
            for n in z.namelist():
                if n.lower().endswith(".wav"):
                    open(os.path.join(dst, os.path.basename(n)), "wb").write(z.read(n))
    return f"{base}/female", f"{base}/male"


def glob_wavs(*dirs):
    out = []
    for d in dirs:
        out += [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".wav")]
    return out


def sample_flat(paths, n, genre):
    rng = random.Random(SEED)
    return [(p, genre) for p in rng.sample(sorted(paths), min(n, len(paths)))]


def main():
    fetch_english_stream()
    eng = sorted(os.path.join(f"{DATA}/english_cv", f)
                 for f in os.listdir(f"{DATA}/english_cv") if f.endswith(".mp3"))
    C = {"English_CV": sample_flat(eng, N_PER_LANG, "read"),
         "Sinhala_read": sample_flat(glob_wavs(unpack_sinhala()), N_PER_LANG, "read"),
         "Tamil_read": sample_flat(glob_wavs(*unpack_tamil()), N_PER_LANG, "read")}
    man = {c: [dict(idx=i, src=p, path=p, genre=g) for i, (p, g) in enumerate(items)]
           for c, items in C.items()}
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    path = f"{ROOT}/results/read_manifest.json"
    json.dump(man, open(path, "w"), indent=1)
    print({c: len(v) for c, v in man.items()}, "->", path)


if __name__ == "__main__":
    main()
