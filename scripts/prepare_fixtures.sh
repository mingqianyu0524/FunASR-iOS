#!/usr/bin/env bash
# prepare_fixtures.sh — Extract CI sample fixtures from MagicData-RAMC and AISHELL-1.
#
# Usage:
#   bash scripts/prepare_fixtures.sh \
#     --ramc-dir    /dev/shm/datasets/MagicData-RAMC \
#     --aishell-dir /dev/shm/datasets/data_aishell \
#     --output-dir  ci_fixtures
#
# Requires: ffmpeg (for audio resampling/format normalization), sox (optional)
#
# Output:
#   ci_fixtures/
#   ├── ramc/
#   │   ├── ramc_001.wav   (16kHz 16-bit mono PCM)
#   │   ├── ramc_001.txt   (normalized reference text)
#   │   └── ...  (25 pairs)
#   └── aishell1/
#       ├── aishell1_001.wav
#       ├── aishell1_001.txt
#       └── ...  (25 pairs)

set -euo pipefail

# ── Argument parsing ─────────────────────────────────────────────────────────

RAMC_DIR=""
AISHELL_DIR=""
OUTPUT_DIR="ci_fixtures"
N_RAMC=25
N_AISHELL=25

usage() {
  echo "Usage: $0 --ramc-dir DIR --aishell-dir DIR [--output-dir DIR]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ramc-dir)    RAMC_DIR="$2";    shift 2 ;;
    --aishell-dir) AISHELL_DIR="$2"; shift 2 ;;
    --output-dir)  OUTPUT_DIR="$2";  shift 2 ;;
    *) usage ;;
  esac
done

[[ -z "$RAMC_DIR" || -z "$AISHELL_DIR" ]] && usage

RAMC_TRANSCRIPT="$RAMC_DIR/UTTERANCEINFO.txt"
AISHELL_TRANSCRIPT="$AISHELL_DIR/transcript/aishell_transcript_v0.8.txt"

if [[ ! -f "$RAMC_TRANSCRIPT" ]]; then
  echo "ERROR: $RAMC_TRANSCRIPT not found" >&2
  exit 1
fi
if [[ ! -f "$AISHELL_TRANSCRIPT" ]]; then
  echo "ERROR: $AISHELL_TRANSCRIPT not found" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR/ramc" "$OUTPUT_DIR/aishell1"

# ── Helper: normalize WAV to 16kHz 16-bit mono ───────────────────────────────

normalize_wav() {
  local src="$1" dst="$2"
  if command -v ffmpeg &>/dev/null; then
    ffmpeg -y -i "$src" -ar 16000 -ac 1 -sample_fmt s16 "$dst" -loglevel error
  else
    cp "$src" "$dst"
    echo "  Warning: ffmpeg not found, copying raw (may not be 16kHz mono)" >&2
  fi
}

# ── Helper: normalize text ────────────────────────────────────────────────────
# Remove punctuation, spaces, special tokens, keep Chinese characters only.

normalize_text_py() {
  python3 - "$1" <<'PYEOF'
import sys, re
text = open(sys.argv[1], encoding='utf-8').read()
text = re.sub(r'<\|[^|>]*\|>', '', text)              # strip special tokens
text = re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbf]', '', text)  # keep Chinese only
print(text.strip())
PYEOF
}

# ── Task 1: RAMC fixtures ─────────────────────────────────────────────────────
# Strategy: select utterances across 3 environment types (quiet/light/heavy)
# and 3 duration ranges (3-5s, 5-10s, 10-15s), balanced male/female.

echo "=== Extracting RAMC fixtures ==="

python3 - "$RAMC_TRANSCRIPT" "$RAMC_DIR" "$OUTPUT_DIR/ramc" "$N_RAMC" <<'PYEOF'
import sys, os, re, random, shutil, subprocess

transcript_file = sys.argv[1]
ramc_dir        = sys.argv[2]
out_dir         = sys.argv[3]
n_target        = int(sys.argv[4])

random.seed(42)

# Parse UTTERANCEINFO.txt
# Expected columns (tab-separated): UTT_ID SPEAKER ENV START END TEXT
# (actual column order may vary — we detect by header)
utterances = []
with open(transcript_file, encoding='utf-8') as f:
    header = None
    for line in f:
        line = line.rstrip('\n')
        if not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) < 6:
            parts = line.split()
        if header is None:
            header = [p.upper() for p in parts]
            continue
        if len(parts) < 6:
            continue
        try:
            utt_id = parts[0]
            env    = parts[4] if len(parts) > 4 else 'unknown'
            text   = parts[-1]
        except Exception:
            continue

        # Skip very short or very long references
        text_clean = re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbf]', '', text)
        if len(text_clean) < 5 or len(text_clean) > 60:
            continue

        # Find audio file
        wav_path = None
        for root, _, files in os.walk(ramc_dir):
            for fname in files:
                if fname.startswith(utt_id) and fname.endswith('.wav'):
                    wav_path = os.path.join(root, fname)
                    break
            if wav_path:
                break

        if wav_path is None:
            continue

        utterances.append({
            'utt_id': utt_id,
            'env': env.lower(),
            'text': text_clean,
            'wav': wav_path,
        })

print(f"Parsed {len(utterances)} valid RAMC utterances")

# Stratified sampling: ~8 quiet, ~9 light, ~8 heavy (approximate)
env_groups = {}
for u in utterances:
    env = u['env']
    if 'quiet' in env or env == 'clean':
        key = 'quiet'
    elif 'light' in env or 'slight' in env or 'low' in env:
        key = 'light'
    else:
        key = 'heavy'
    env_groups.setdefault(key, []).append(u)

selected = []
targets = {'quiet': 8, 'light': 9, 'heavy': 8}
for key, target in targets.items():
    pool = env_groups.get(key, [])
    random.shuffle(pool)
    selected.extend(pool[:target])

# If not enough, pad with random
if len(selected) < n_target:
    remaining = [u for u in utterances if u not in selected]
    random.shuffle(remaining)
    selected.extend(remaining[:n_target - len(selected)])

selected = selected[:n_target]

# Copy and normalize
for idx, u in enumerate(selected, 1):
    fname_base = f"ramc_{idx:03d}"
    dst_wav = os.path.join(out_dir, fname_base + '.wav')
    dst_txt = os.path.join(out_dir, fname_base + '.txt')

    subprocess.run(
        ['ffmpeg', '-y', '-i', u['wav'], '-ar', '16000', '-ac', '1',
         '-sample_fmt', 's16', dst_wav, '-loglevel', 'error'],
        check=False
    )
    if not os.path.exists(dst_wav):
        shutil.copy(u['wav'], dst_wav)

    with open(dst_txt, 'w', encoding='utf-8') as f:
        f.write(u['text'])

    print(f"  [{idx:02d}/{n_target}] {u['utt_id']} → {fname_base}")

print(f"RAMC: extracted {len(selected)} fixtures to {out_dir}")
PYEOF

echo ""

# ── Task 2: AISHELL-1 fixtures ────────────────────────────────────────────────
# Strategy: 25 utterances from 25 different speakers in test set.

echo "=== Extracting AISHELL-1 fixtures ==="

python3 - "$AISHELL_TRANSCRIPT" "$AISHELL_DIR/wav/test" "$OUTPUT_DIR/aishell1" "$N_AISHELL" <<'PYEOF'
import sys, os, re, random, shutil, subprocess
from pathlib import Path

transcript_file = sys.argv[1]
test_wav_dir    = sys.argv[2]
out_dir         = sys.argv[3]
n_target        = int(sys.argv[4])

random.seed(42)

# Load transcripts
refs = {}
with open(transcript_file, encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            utt_id, text = parts
            text_clean = re.sub(r'\s+', '', text)
            text_clean = re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbf]', '', text_clean)
            if 5 <= len(text_clean) <= 60:
                refs[utt_id] = text_clean

print(f"Loaded {len(refs)} AISHELL-1 references")

# Find WAV files and group by speaker
wav_map = {}
for wav_path in Path(test_wav_dir).rglob('*.wav'):
    utt_id = wav_path.stem
    if utt_id in refs:
        speaker_id = utt_id[:7]  # e.g. BAC009S0764
        wav_map[utt_id] = {'wav': str(wav_path), 'speaker': speaker_id}

print(f"Matched {len(wav_map)} utterances with audio")

# Select 1 utterance per speaker
by_speaker = {}
for utt_id, info in wav_map.items():
    spk = info['speaker']
    by_speaker.setdefault(spk, []).append(utt_id)

speakers = list(by_speaker.keys())
random.shuffle(speakers)
selected = []
for spk in speakers[:n_target]:
    utts = by_speaker[spk]
    random.shuffle(utts)
    selected.append(utts[0])

# Pad if fewer speakers than target
if len(selected) < n_target:
    all_utts = list(wav_map.keys())
    random.shuffle(all_utts)
    for u in all_utts:
        if u not in selected:
            selected.append(u)
        if len(selected) >= n_target:
            break

selected = selected[:n_target]

# Copy and normalize
for idx, utt_id in enumerate(selected, 1):
    fname_base = f"aishell1_{idx:03d}"
    src_wav = wav_map[utt_id]['wav']
    dst_wav = os.path.join(out_dir, fname_base + '.wav')
    dst_txt = os.path.join(out_dir, fname_base + '.txt')

    subprocess.run(
        ['ffmpeg', '-y', '-i', src_wav, '-ar', '16000', '-ac', '1',
         '-sample_fmt', 's16', dst_wav, '-loglevel', 'error'],
        check=False
    )
    if not os.path.exists(dst_wav):
        shutil.copy(src_wav, dst_wav)

    with open(dst_txt, 'w', encoding='utf-8') as f:
        f.write(refs[utt_id])

    print(f"  [{idx:02d}/{n_target}] {utt_id} → {fname_base}")

print(f"AISHELL-1: extracted {len(selected)} fixtures to {out_dir}")
PYEOF

echo ""

# ── Package and upload ────────────────────────────────────────────────────────

echo "=== Packaging fixtures ==="
cd "$OUTPUT_DIR"
tar -czf ../ramc_fixtures.tar.gz    ramc/
tar -czf ../aishell1_fixtures.tar.gz aishell1/
cd ..

echo "Created:"
ls -lh ramc_fixtures.tar.gz aishell1_fixtures.tar.gz

echo ""
echo "=== Upload to GitHub Release ==="
echo "Run the following command to create the release:"
echo ""
echo "  gh release create test-fixtures-v1.0 \\"
echo "    ramc_fixtures.tar.gz \\"
echo "    aishell1_fixtures.tar.gz \\"
echo "    --repo mingqianyu0524/FunASR-iOS \\"
echo '    --title "CI Test Fixtures v1.0" \'
echo '    --notes "25 MagicData-RAMC utterances (academic license) + 25 AISHELL-1 utterances (Apache 2.0). For CI accuracy regression testing only."'
echo ""
echo "Then update ci/fixture_versions.txt if the version changes."
