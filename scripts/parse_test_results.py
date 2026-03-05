#!/usr/bin/env python3
"""
parse_test_results.py — Parse xcodebuild test logs and generate CI accuracy report.

Extracts CER/SER numbers from testBatchEval() output, compares to baseline JSONs,
and writes a Markdown summary table to stdout / $GITHUB_STEP_SUMMARY.

Usage:
    # Primary: read JSON written by testBatchEval()
    python3 scripts/parse_test_results.py \
        --results-json ci_fixtures/batch_eval_results.json \
        --baseline-dir eval_results/ \
        --output accuracy_report.md

    # Legacy: parse xcodebuild build.log (only works if print() goes to stdout)
    python3 scripts/parse_test_results.py \
        --log build.log \
        --baseline-dir eval_results/ \
        --output accuracy_report.md
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


# ── Log parsing ───────────────────────────────────────────────────────────────

BATCH_EVAL_PATTERN = re.compile(
    r"── Batch Eval: (?P<dataset>[^\(]+?)\s+\((?P<n>\d+) utterances?\) ──\n"
    r"CER:\s+(?P<cer>[\d.]+)%\s+SER:\s+(?P<ser>[\d.]+)%\n"
    r"D:\s*(?P<d>\d+)\s+I:\s*(?P<i>\d+)\s+S:\s*(?P<s>\d+)\s+Ref chars:\s*(?P<ref>\d+)",
    re.MULTILINE,
)

SNR_PATTERN = re.compile(
    r"SNR\s+(?P<snr>[\d.]+)dB\s+CER:\s+(?P<cer>[\d.]+)%",
    re.MULTILINE,
)

CER_SINGLE_PATTERN = re.compile(
    r"CER:\s+([\d.]+)%.*RTF:\s+([\d.]+)",
    re.MULTILINE,
)


def parse_results_json(json_path: str) -> dict:
    """Read batch eval results written by testBatchEval() to EVAL_AUDIO_DIR."""
    with open(json_path, encoding="utf-8") as f:
        entries = json.load(f)
    results = {"batch_evals": [], "single_tests": [], "snr_curve": []}
    for entry in entries:
        results["batch_evals"].append({
            "dataset":       entry["dataset"],
            "utterances":    int(entry["utterances"]),
            "cer":           float(entry["cer"]),
            "ser":           float(entry["ser"]),
            "deletions":     int(entry["deletions"]),
            "insertions":    int(entry["insertions"]),
            "substitutions": int(entry["substitutions"]),
            "ref_chars":     int(entry["ref_chars"]),
        })
    return results


def parse_log(log_path: str) -> dict:
    """Extract accuracy metrics from xcodebuild test log."""
    with open(log_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    results = {
        "batch_evals": [],
        "single_tests": [],
        "snr_curve": [],
    }

    # Batch eval blocks
    for m in BATCH_EVAL_PATTERN.finditer(content):
        results["batch_evals"].append({
            "dataset": m.group("dataset").strip(),
            "utterances": int(m.group("n")),
            "cer": float(m.group("cer")) / 100,
            "ser": float(m.group("ser")) / 100,
            "deletions": int(m.group("d")),
            "insertions": int(m.group("i")),
            "substitutions": int(m.group("s")),
            "ref_chars": int(m.group("ref")),
        })

    # Single CER tests (testSenseVoiceSample)
    for m in CER_SINGLE_PATTERN.finditer(content):
        results["single_tests"].append({
            "cer": float(m.group(1)) / 100,
            "rtf": float(m.group(2)),
        })

    # SNR curve
    for m in SNR_PATTERN.finditer(content):
        results["snr_curve"].append({
            "snr_db": float(m.group("snr")),
            "cer": float(m.group("cer")) / 100,
        })

    return results


# ── Baseline comparison ───────────────────────────────────────────────────────

DATASET_TO_BASELINE = {
    "RAMC":     "sensevoice_ramc.json",
    "AISHELL-1": "sensevoice_aishell1.json",
    "AISHELL1": "sensevoice_aishell1.json",
}


def load_baseline(baseline_dir: str, dataset: str) -> dict | None:
    for key, fname in DATASET_TO_BASELINE.items():
        if key.upper() in dataset.upper():
            path = os.path.join(baseline_dir, fname)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
    return None


def delta_str(current: float | None, baseline: float | None) -> str:
    """Format CER delta with trend emoji."""
    if current is None or baseline is None:
        return "N/A"
    delta = current - baseline
    sign = "+" if delta >= 0 else ""
    arrow = "⬆" if delta > 0.01 else ("⬇" if delta < -0.01 else "→")
    return f"{sign}{delta*100:.1f}pp {arrow}"


# ── Markdown generation ───────────────────────────────────────────────────────

def generate_report(parsed: dict, baseline_dir: str | None) -> str:
    lines = ["## ASR Accuracy Report\n"]

    # Single sample test
    if parsed["single_tests"]:
        t = parsed["single_tests"][0]
        lines.append(f"**Smoke test (single RAMC sample):** "
                     f"CER `{t['cer']*100:.1f}%` · RTF `{t['rtf']:.3f}`\n")

    # Batch eval table
    if parsed["batch_evals"]:
        lines.append("\n### Batch Evaluation\n")
        header = "| Dataset | Utts | CER | SER | D | I | S | vs Baseline |"
        sep    = "|---------|------|-----|-----|---|---|---|-------------|"
        lines += [header, sep]

        for ev in parsed["batch_evals"]:
            baseline = load_baseline(baseline_dir, ev["dataset"]) if baseline_dir else None
            bl_cer = baseline.get("cer") if baseline else None
            delta = delta_str(ev["cer"], bl_cer)
            bl_str = f"`{bl_cer*100:.1f}%`" if bl_cer is not None else "N/A"

            row = (f"| {ev['dataset']} | {ev['utterances']} "
                   f"| `{ev['cer']*100:.1f}%` | `{ev['ser']*100:.1f}%` "
                   f"| {ev['deletions']} | {ev['insertions']} | {ev['substitutions']} "
                   f"| {delta} (base {bl_str}) |")
            lines.append(row)

        lines.append("")

        # Regression warning
        for ev in parsed["batch_evals"]:
            baseline = load_baseline(baseline_dir, ev["dataset"]) if baseline_dir else None
            if baseline and baseline.get("cer") is not None:
                delta = ev["cer"] - baseline["cer"]
                if delta > 0.05:
                    lines.append(f"> ⚠️ **Regression detected**: {ev['dataset']} CER worsened "
                                 f"by {delta*100:.1f}pp vs baseline.\n")

    # SNR curve table
    if parsed["snr_curve"]:
        lines.append("\n### SNR Robustness Curve\n")
        lines.append("| SNR (dB) | CER |")
        lines.append("|----------|-----|")
        for entry in sorted(parsed["snr_curve"], key=lambda x: -x["snr_db"]):
            lines.append(f"| {entry['snr_db']:.0f} | `{entry['cer']*100:.1f}%` |")
        lines.append("")

    if not parsed["batch_evals"] and not parsed["single_tests"]:
        lines.append("_No accuracy metrics found in build log. "
                     "Check if testBatchEval/testSenseVoiceSample ran successfully._\n")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate CI accuracy report from build log")
    parser.add_argument("--log", default=None, help="xcodebuild build.log path (legacy)")
    parser.add_argument("--results-json", default=None,
                        help="JSON file written by testBatchEval() (preferred over --log)")
    parser.add_argument("--baseline-dir", default=None,
                        help="Directory containing baseline JSON files")
    parser.add_argument("--output", required=True, help="Output Markdown file path")
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="Exit non-zero if CER regresses > 5pp vs baseline")
    args = parser.parse_args()

    if args.results_json and os.path.exists(args.results_json):
        parsed = parse_results_json(args.results_json)
    elif args.log:
        if not os.path.exists(args.log):
            print(f"ERROR: Log file not found: {args.log}", file=sys.stderr)
            sys.exit(1)
        parsed = parse_log(args.log)
    else:
        print("ERROR: provide --results-json or --log", file=sys.stderr)
        sys.exit(1)
    report = generate_report(parsed, args.baseline_dir)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)

    # Also append to GitHub Step Summary if running in CI
    github_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if github_summary:
        with open(github_summary, "a", encoding="utf-8") as f:
            f.write("\n" + report)

    # Check for regression
    if args.fail_on_regression and args.baseline_dir:
        for ev in parsed["batch_evals"]:
            baseline = load_baseline(args.baseline_dir, ev["dataset"])
            if baseline and baseline.get("cer") is not None:
                delta = ev["cer"] - baseline["cer"]
                if delta > 0.05:
                    print(f"\nFAIL: {ev['dataset']} CER regressed by {delta*100:.1f}pp "
                          f"(threshold: 5pp)", file=sys.stderr)
                    sys.exit(1)

    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
