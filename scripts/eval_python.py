#!/usr/bin/env python3
"""
eval_python.py — ASR accuracy evaluation script for SenseVoice and Paraformer ONNX int8 models.

Usage:
    python3 scripts/eval_python.py \
        --model sensevoice \
        --model-dir ~/asr_models \
        --audio-dir /dev/shm/datasets/MagicData-RAMC/test \
        --transcript /dev/shm/datasets/MagicData-RAMC/UTTERANCEINFO.txt \
        --output results/sensevoice_ramc.json

Supports: sensevoice, paraformer
Datasets:  MagicData-RAMC (UTTERANCEINFO.txt format), AISHELL-1 (aishell_transcript_v0.8.txt format)
"""

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_wav_as_float(path: str) -> np.ndarray:
    """Load a 16kHz 16-bit mono WAV file as float32 array in [-1, 1]."""
    import struct
    import wave

    with wave.open(path, "rb") as wf:
        assert wf.getsampwidth() == 2, "Only 16-bit PCM WAV supported"
        assert wf.getnchannels() == 1, "Only mono WAV supported"
        raw = wf.readframes(wf.getnframes())
        n = len(raw) // 2
        samples = struct.unpack(f"<{n}h", raw)
        return np.array(samples, dtype=np.float32) / 32768.0


def load_ramc_transcript(transcript_path: str) -> dict:
    """
    Parse MagicData-RAMC UTTERANCEINFO.txt.
    Expected format (tab-separated):
        UTTERANCE_ID  SPEAKER_ID  START  END  ENVIRONMENT  TEXT
    Returns: {utterance_id: text}
    """
    refs = {}
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("UTTERANCE"):
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                # Try space-split fallback
                parts = line.split()
            if len(parts) >= 6:
                utt_id = parts[0]
                text = parts[-1]
                refs[utt_id] = normalize_text(text)
    return refs


def load_aishell_transcript(transcript_path: str) -> dict:
    """
    Parse AISHELL-1 aishell_transcript_v0.8.txt.
    Each line: BAC009S0002W0122 他 仅 凭 着 一 腔 热 情 做 事
    Returns: {utt_id: text_no_spaces}
    """
    refs = {}
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                utt_id, text = parts
                refs[utt_id] = normalize_text(text)
    return refs


def normalize_text(text: str) -> str:
    """Remove spaces, punctuation, special tokens, and convert to uppercase for comparison."""
    # Strip SenseVoice/FunASR special tokens like <|zh|>
    text = re.sub(r"<\|[^|>]*\|>", "", text)
    # Remove punctuation (Chinese and ASCII)
    text = re.sub(r"[^\u4e00-\u9fff\u3400-\u4dbf\uff00-\uffef\u3000-\u303f\w]", "", text)
    # Remove ASCII digits and latin letters (keep Chinese)
    text = re.sub(r"[a-zA-Z0-9_]", "", text)
    # Remove whitespace
    text = re.sub(r"\s+", "", text)
    return text


def levenshtein_with_details(ref: list, hyp: list) -> tuple:
    """
    Compute Levenshtein distance between two token lists.
    Returns (distance, deletions, insertions, substitutions).
    """
    n, m = len(ref), len(hyp)
    # dp[i][j] = (distance, d, i_count, s)
    INF = 10**9
    dp = [[(INF, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = (0, 0, 0, 0)
    for i in range(1, n + 1):
        dp[i][0] = (i, i, 0, 0)  # i deletions
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, j, 0)  # j insertions
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # substitution
                sub_d, sub_del, sub_ins, sub_sub = dp[i - 1][j - 1]
                sub = (sub_d + 1, sub_del, sub_ins, sub_sub + 1)
                # deletion (ref token not in hyp)
                del_d, del_del, del_ins, del_sub = dp[i - 1][j]
                deletion = (del_d + 1, del_del + 1, del_ins, del_sub)
                # insertion (hyp token not in ref)
                ins_d, ins_del, ins_ins, ins_sub = dp[i][j - 1]
                insertion = (ins_d + 1, ins_del, ins_ins + 1, ins_sub)
                dp[i][j] = min(sub, deletion, insertion, key=lambda x: x[0])
    return dp[n][m]


def run_sensevoice(model_dir: str, audio_files: list) -> dict:
    """Run SenseVoice ONNX inference on a list of audio files. Returns {filename: text}."""
    import onnxruntime as ort

    model_path = os.path.join(model_dir, "sensevoice.int8.onnx")
    tokens_path = os.path.join(model_dir, "sensevoice_tokens.txt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(tokens_path):
        raise FileNotFoundError(f"Tokens not found: {tokens_path}")

    # Load tokenizer
    tokens = {}
    with open(tokens_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                tokens[int(parts[1])] = parts[0]

    # Create ONNX session
    sess_options = ort.SessionOptions()
    sess_options.inter_op_num_threads = 2
    sess_options.intra_op_num_threads = 2
    session = ort.InferenceSession(model_path, sess_options=sess_options)

    # Feature extraction: use funasr WavFrontend (forward_fbank + forward_lfr_cmvn)
    # Fall back to built-in implementation if unavailable.
    frontend = None
    try:
        import torch
        from funasr.frontends.wav_frontend import WavFrontend
        frontend = WavFrontend(cmvn_file=None, fs=16000, window="hamming",
                               n_mels=80, frame_length=25, frame_shift=10,
                               lfr_m=7, lfr_n=6)
    except Exception:
        pass

    results = {}
    for audio_path in tqdm(audio_files, desc="SenseVoice inference", unit="utt"):
        try:
            pcm = load_wav_as_float(audio_path)

            if frontend is not None:
                import torch
                pcm_tensor = torch.from_numpy(pcm).unsqueeze(0)  # [1, T]
                feat_len_tensor = torch.tensor([len(pcm)], dtype=torch.int32)
                fbank, fbank_len = frontend.forward_fbank(pcm_tensor, feat_len_tensor)
                feats, feat_len_t = frontend.forward_lfr_cmvn(fbank, fbank_len)
                feats = feats.detach().numpy()  # [1, T_lfr, 560]
            else:
                # Built-in fallback: compute log-mel + LFR without funasr
                feats = _compute_fbank_lfr(pcm)

            feat_len_arr = np.array([feats.shape[1]], dtype=np.int32)

            # SenseVoice model inputs: x [N,T,560], x_length [N], language [N], text_norm [N]
            language = np.array([0], dtype=np.int32)   # 0 = auto
            text_norm = np.array([14], dtype=np.int32)  # 14 = withinutterance

            outputs = session.run(None, {
                "x": feats.astype(np.float32),
                "x_length": feat_len_arr,
                "language": language,
                "text_norm": text_norm,
            })

            logits = outputs[0]  # [1, T, vocab]
            token_ids = np.argmax(logits[0], axis=-1)

            # CTC greedy decode: collapse repeats, remove blank (0)
            prev = -1
            decoded = []
            for tid in token_ids:
                if tid != prev and tid != 0:
                    decoded.append(tid)
                prev = tid

            text = "".join(tokens.get(t, "") for t in decoded)
            text = normalize_text(text)
            results[os.path.basename(audio_path)] = text
        except Exception as e:
            print(f"  Warning: failed on {audio_path}: {e}", file=sys.stderr)
            results[os.path.basename(audio_path)] = ""

    return results


def run_paraformer(model_dir: str, audio_files: list) -> dict:
    """Run Paraformer ONNX inference on a list of audio files. Returns {filename: text}."""
    import onnxruntime as ort

    enc_path = os.path.join(model_dir, "paraformer_enc.int8.onnx")
    dec_path = os.path.join(model_dir, "paraformer_dec.int8.onnx")
    tokens_path = os.path.join(model_dir, "paraformer_tokens.txt")
    mvn_path = os.path.join(model_dir, "paraformer_am.mvn")

    for p in [enc_path, dec_path, tokens_path, mvn_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Model file not found: {p}")

    # Load tokenizer
    tokens = {}
    with open(tokens_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                tokens[int(parts[1])] = parts[0]

    # Load CMVN stats
    cmvn_mean, cmvn_istd = _load_mvn(mvn_path)

    sess_options = ort.SessionOptions()
    sess_options.inter_op_num_threads = 2
    sess_options.intra_op_num_threads = 2
    enc_session = ort.InferenceSession(enc_path, sess_options=sess_options)
    dec_session = ort.InferenceSession(dec_path, sess_options=sess_options)

    results = {}
    for audio_path in tqdm(audio_files, desc="Paraformer inference", unit="utt"):
        try:
            pcm = load_wav_as_float(audio_path)
            feats = _compute_fbank(pcm)           # [T, 80]
            feats = _apply_lfr(feats)              # [T_lfr, 560]  LFR: m=7, n=6

            # Apply CMVN (560-dim, applied after LFR stacking)
            feats = (feats - cmvn_mean) * cmvn_istd
            feats = feats[None].astype(np.float32)  # [1, T_lfr, 560]
            feat_len = np.array([feats.shape[1]], dtype=np.int32)

            # Encoder
            enc_outputs = enc_session.run(None, {
                "speech": feats,
                "speech_lengths": feat_len,
            })
            enc, enc_len, alphas = enc_outputs[0], enc_outputs[1], enc_outputs[2]

            # CIF: predict token count from alphas
            token_count = max(1, int(np.round(float(alphas.sum()))))
            acoustic_embeds = _cif_forward(enc[0], alphas[0], token_count)
            acoustic_embeds = acoustic_embeds[None].astype(np.float32)  # [1, N, D]
            acoustic_embeds_len = np.array([token_count], dtype=np.int32)

            # Initialize caches as zeros (offline mode: no streaming cache needed)
            # Shape from model: [batch_size, 512, 10]
            caches_in = {f"in_cache_{i}": np.zeros((1, 512, 10), dtype=np.float32)
                         for i in range(16)}

            dec_inputs = {
                "enc": enc,
                "enc_len": enc_len,
                "acoustic_embeds": acoustic_embeds,
                "acoustic_embeds_len": acoustic_embeds_len,
                **caches_in,
            }
            dec_outputs = dec_session.run(None, dec_inputs)
            logits = dec_outputs[0]  # [1, N, vocab]

            token_ids = np.argmax(logits[0], axis=-1)
            text = "".join(tokens.get(int(t), "") for t in token_ids)
            text = normalize_text(text)
            results[os.path.basename(audio_path)] = text
        except Exception as e:
            print(f"  Warning: failed on {audio_path}: {e}", file=sys.stderr)
            results[os.path.basename(audio_path)] = ""

    return results


def _compute_fbank(pcm: np.ndarray, sr: int = 16000, n_mels: int = 80,
                   frame_ms: int = 25, hop_ms: int = 10) -> np.ndarray:
    """Compute 80-dim log-mel filterbank features. Returns [T, 80]."""
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    n_fft = 512

    # Pre-emphasis
    pcm = np.append(pcm[0], pcm[1:] - 0.97 * pcm[:-1])

    # Pad to full frames
    num_frames = 1 + (len(pcm) - frame_len) // hop_len
    padded_len = (num_frames - 1) * hop_len + frame_len
    pcm = np.pad(pcm, (0, max(0, padded_len - len(pcm))))

    # Hamming window
    window = np.hamming(frame_len)

    # Framing
    frames = np.stack([pcm[i * hop_len:i * hop_len + frame_len] * window
                       for i in range(num_frames)])

    # FFT -> power spectrum
    spec = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2

    # Mel filterbank
    mel_filters = _mel_filterbank(sr, n_fft, n_mels)
    mel_spec = spec @ mel_filters.T
    log_mel = np.log(np.maximum(mel_spec, 1e-10))

    return log_mel.astype(np.float32)


def _apply_lfr(feats: np.ndarray, lfr_m: int = 7, lfr_n: int = 6) -> np.ndarray:
    """Apply Low Frame Rate (LFR) stacking to fbank features.

    Stacks lfr_m consecutive frames (centered) with stride lfr_n.
    Input:  [T, D]
    Output: [T_lfr, D*lfr_m]
    """
    T, D = feats.shape
    out = []
    # Pad left with lfr_m//2 copies of first frame
    left_pad = lfr_m // 2
    padded = np.concatenate([np.tile(feats[0], (left_pad, 1)), feats], axis=0)  # [T+left_pad, D]
    for i in range(0, T, lfr_n):
        start = i  # index into padded (already offset by left_pad)
        chunk = padded[start:start + lfr_m]
        if len(chunk) < lfr_m:
            pad = np.tile(feats[-1], (lfr_m - len(chunk), 1))
            chunk = np.concatenate([chunk, pad], axis=0)
        out.append(chunk.flatten())
    return np.stack(out).astype(np.float32)  # [T_lfr, D*lfr_m]


def _compute_fbank_lfr(pcm: np.ndarray, sr: int = 16000, lfr_m: int = 7,
                       lfr_n: int = 6) -> np.ndarray:
    """Compute fbank + LFR for SenseVoice. Returns [1, T_lfr, 560]."""
    feats = _compute_fbank(pcm, sr)  # [T, 80]
    T, D = feats.shape
    # LFR stacking
    out = []
    for i in range(0, T, lfr_n):
        start = max(0, i - lfr_m // 2)
        end = min(T, i + lfr_m - lfr_m // 2)
        chunk = feats[start:end]
        # Pad if needed
        if len(chunk) < lfr_m:
            pad = np.zeros((lfr_m - len(chunk), D), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)
        out.append(chunk.flatten())
        if i + lfr_n >= T:
            break
    lfr_feats = np.stack(out)[None].astype(np.float32)  # [1, T_lfr, 560]
    return lfr_feats


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Compute mel filterbank matrix of shape [n_mels, n_fft//2+1]."""
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    low_hz, high_hz = 0.0, sr / 2
    low_mel = hz_to_mel(low_hz)
    high_mel = hz_to_mel(high_hz)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_m_minus, f_m, f_m_plus = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            filters[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            filters[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    return filters.astype(np.float32)


def _load_mvn(mvn_path: str):
    """Load CMVN stats from FunASR am.mvn file. Returns (mean, istd) as numpy arrays.

    The .mvn file format used by online Paraformer:
        <AddShift> 560 560
        <LearnRateCoef> 0 [ v1 v2 ... v560 ]
        <Rescale> 560 560
        <LearnRateCoef> 0 [ v1 v2 ... v560 ]
    where AddShift stores (-mean) and Rescale stores (1/std), both in 560-dim
    space (i.e., applied AFTER LFR stacking of 7×80=560 fbank frames).
    """
    with open(mvn_path, encoding="utf-8") as f:
        content = f.read()

    # Extract values in [ ... ] blocks after <AddShift> and <Rescale>
    add_shift_match = re.search(r"<AddShift>.*?\[\s*(.*?)\s*\]", content, re.DOTALL)
    rescale_match = re.search(r"<Rescale>.*?\[\s*(.*?)\s*\]", content, re.DOTALL)

    if add_shift_match and rescale_match:
        neg_means = np.array([float(x) for x in add_shift_match.group(1).split()])
        istds = np.array([float(x) for x in rescale_match.group(1).split()])
        # AddShift = -mean → mean = -AddShift
        cmvn_mean = -neg_means
        cmvn_istd = istds
    else:
        # Fallback: assume no CMVN
        print("Warning: Could not parse CMVN stats, using identity", file=sys.stderr)
        cmvn_mean = np.zeros(560, dtype=np.float32)
        cmvn_istd = np.ones(560, dtype=np.float32)

    return cmvn_mean.astype(np.float32), cmvn_istd.astype(np.float32)


def _cif_forward(enc_out: np.ndarray, alphas: np.ndarray, token_count: int) -> np.ndarray:
    """
    Simplified CIF (Continuous Integrate-and-Fire) forward pass.
    enc_out: [T, D], alphas: [T], token_count: int
    Returns: acoustic_embeds [N, D]
    """
    T, D = enc_out.shape
    alphas = alphas[:T]

    # Normalize alphas to sum to token_count
    alpha_sum = alphas.sum()
    if alpha_sum > 0:
        alphas = alphas * (token_count / alpha_sum)

    embeds = []
    accumulated = 0.0
    current = np.zeros(D, dtype=np.float32)

    for t in range(T):
        alpha_t = float(alphas[t])
        while accumulated + alpha_t >= 1.0:
            remaining = 1.0 - accumulated
            current = current + remaining * enc_out[t]
            embeds.append(current.copy())
            alpha_t -= remaining
            accumulated = 0.0
            current = np.zeros(D, dtype=np.float32)
        accumulated += alpha_t
        current = current + alpha_t * enc_out[t]

    if len(embeds) < token_count and accumulated > 0:
        embeds.append(current)

    if not embeds:
        embeds = [np.zeros(D, dtype=np.float32)]

    result = np.stack(embeds[:token_count])
    # Pad if short
    if len(result) < token_count:
        pad = np.zeros((token_count - len(result), D), dtype=np.float32)
        result = np.concatenate([result, pad], axis=0)
    return result


def detect_dataset_format(transcript_path: str) -> str:
    """Detect whether transcript is RAMC or AISHELL-1 format."""
    with open(transcript_path, encoding="utf-8") as f:
        first_line = f.readline().strip()
    if "UTTERANCE" in first_line.upper() or "\t" in first_line:
        return "ramc"
    return "aishell1"


def scan_audio_files(audio_dir: str) -> list:
    """Recursively scan directory for .wav files."""
    wav_files = sorted(Path(audio_dir).rglob("*.wav"))
    return [str(p) for p in wav_files]


def compute_metrics(hypotheses: dict, references: dict) -> dict:
    """Compute CER, SER, D/I/S across all utterances."""
    total_chars = 0
    total_dist = 0
    total_d = total_i = total_s = 0
    total_utts = 0
    error_utts = 0

    matched = 0
    for fname, hyp in hypotheses.items():
        # Try to match by utterance ID (stem without extension)
        utt_id = Path(fname).stem
        ref = references.get(utt_id) or references.get(fname) or references.get(utt_id.upper())
        if ref is None:
            # Try partial match
            for k in references:
                if utt_id in k or k in utt_id:
                    ref = references[k]
                    break
        if ref is None:
            continue

        matched += 1
        ref_chars = list(ref)
        hyp_chars = list(hyp)
        dist, d, ins, sub = levenshtein_with_details(ref_chars, hyp_chars)
        total_chars += len(ref_chars)
        total_dist += dist
        total_d += d
        total_i += ins
        total_s += sub
        total_utts += 1
        if dist > 0:
            error_utts += 1

    if total_utts == 0 or total_chars == 0:
        return {
            "total_utterances": 0,
            "matched_utterances": matched,
            "cer": None,
            "ser": None,
            "deletions": 0,
            "insertions": 0,
            "substitutions": 0,
            "total_ref_chars": 0,
        }

    return {
        "total_utterances": total_utts,
        "matched_utterances": matched,
        "cer": round(total_dist / total_chars, 4),
        "ser": round(error_utts / total_utts, 4),
        "deletions": total_d,
        "insertions": total_i,
        "substitutions": total_s,
        "total_ref_chars": total_chars,
    }


def main():
    parser = argparse.ArgumentParser(description="ASR accuracy evaluation")
    parser.add_argument("--model", required=True, choices=["sensevoice", "paraformer"],
                        help="Model to evaluate")
    parser.add_argument("--model-dir", required=True, help="Directory containing model files")
    parser.add_argument("--audio-dir", required=True,
                        help="Directory with .wav files (recursive scan)")
    parser.add_argument("--transcript", required=True,
                        help="Reference transcript file (RAMC or AISHELL-1 format)")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--dataset", default=None,
                        help="Dataset name for JSON output (auto-detected if not set)")
    parser.add_argument("--max-utterances", type=int, default=None,
                        help="Limit number of utterances (for quick testing)")
    args = parser.parse_args()

    # Load reference transcripts
    fmt = detect_dataset_format(args.transcript)
    dataset_name = args.dataset or (
        "RAMC-test" if fmt == "ramc" else "AISHELL1-test"
    )
    print(f"Loading {dataset_name} transcripts from {args.transcript} (format: {fmt})")
    if fmt == "ramc":
        references = load_ramc_transcript(args.transcript)
    else:
        references = load_aishell_transcript(args.transcript)
    print(f"  Loaded {len(references)} reference utterances")

    # Scan audio files
    audio_files = scan_audio_files(args.audio_dir)
    if args.max_utterances:
        audio_files = audio_files[:args.max_utterances]
    print(f"Found {len(audio_files)} WAV files in {args.audio_dir}")

    if not audio_files:
        print("ERROR: No WAV files found.", file=sys.stderr)
        sys.exit(1)

    # Run inference
    if args.model == "sensevoice":
        hypotheses = run_sensevoice(args.model_dir, audio_files)
    else:
        hypotheses = run_paraformer(args.model_dir, audio_files)

    # Compute metrics
    metrics = compute_metrics(hypotheses, references)

    # Build output
    result = {
        "model": args.model,
        "dataset": dataset_name,
        **metrics,
        "eval_date": str(date.today()),
    }

    # Save JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print human-readable summary
    cer_pct = f"{result['cer']*100:.1f}%" if result["cer"] is not None else "N/A"
    ser_pct = f"{result['ser']*100:.1f}%" if result["ser"] is not None else "N/A"
    print(f"""
┌─────────────────────────────────────────────┐
│  Model:    {result['model']:<33} │
│  Dataset:  {result['dataset']:<33} │
│  Utts:     {result['total_utterances']:<33} │
│  CER:      {cer_pct:<33} │
│  SER:      {ser_pct:<33} │
│  D / I / S: {result['deletions']} / {result['insertions']} / {result['substitutions']}
│  Ref chars: {result['total_ref_chars']:<32} │
└─────────────────────────────────────────────┘""")

    if result["cer"] is not None and result["cer"] > 0.30:
        print(f"\nWARNING: CER {cer_pct} exceeds 30% — check model files and audio format")
        sys.exit(1)


if __name__ == "__main__":
    main()
