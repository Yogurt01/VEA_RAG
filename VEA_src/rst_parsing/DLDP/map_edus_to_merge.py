#!/usr/bin/env python3
"""Map EDU lines to a CoNLL file and write a DPLP-compatible .merge file.

This script is for the workflow where EDU segmentation comes from an external
source (e.g., NeuralEDUseg or video scene captions), so we skip segmenter.py.
"""

import argparse
import sys
from typing import List, Tuple


def normalize_token(token: str) -> str:
    """Normalize tokens for robust matching across tokenizers."""
    t = token.strip().lower()
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    t = t.replace("`", "'")
    return t


def collapse_tokens(tokens: List[str]) -> str:
    return "".join(normalize_token(tok) for tok in tokens if normalize_token(tok))


def read_conll(path: str) -> List[List[str]]:
    rows: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            cols = line.split("\t")
            if len(cols) < 8:
                raise ValueError("Invalid CoNLL line (<8 cols): {}".format(line))
            rows.append(cols)
    if not rows:
        raise ValueError("No token rows found in CoNLL file: {}".format(path))
    return rows


def read_edu_lines(path: str) -> List[List[str]]:
    out: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            out.append(line.split())
    if not out:
        raise ValueError("No non-empty EDU lines found: {}".format(path))
    return out


def align_edus_to_conll(
    conll_rows: List[List[str]], edu_lines: List[List[str]]
) -> List[int]:
    """Return eduidx per conll token row (1-based EDU index)."""
    tokens = [row[2] for row in conll_rows]
    edu_idx_per_token = [0] * len(tokens)
    ptr = 0

    for edu_idx, edu_tokens in enumerate(edu_lines, start=1):
        target = collapse_tokens(edu_tokens)
        if not target:
            continue

        start_ptr = ptr
        acc = ""
        matched = False

        while ptr < len(tokens):
            tok_norm = normalize_token(tokens[ptr])
            candidate = acc + tok_norm

            if not target.startswith(candidate):
                break

            acc = candidate
            ptr += 1

            if acc == target:
                for i in range(start_ptr, ptr):
                    edu_idx_per_token[i] = edu_idx
                matched = True
                break

        if not matched:
            context_start = max(0, ptr - 3)
            context_end = min(len(tokens), ptr + 4)
            context = " ".join(tokens[context_start:context_end])
            raise ValueError(
                "Cannot align EDU #{}.\n"
                "EDU text: {}\n"
                "Current CoNLL token pointer: {}\n"
                "Nearby CoNLL tokens: {}".format(
                    edu_idx, " ".join(edu_tokens), ptr, context
                )
            )

    if ptr != len(tokens):
        remain = " ".join(tokens[ptr : min(len(tokens), ptr + 20)])
        raise ValueError(
            "Alignment finished but {} CoNLL tokens remain unmatched.\n"
            "Remaining tokens start with: {}".format(len(tokens) - ptr, remain)
        )

    return edu_idx_per_token


def write_merge(
    conll_rows: List[List[str]], edu_idx_per_token: List[int], output_path: str
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        prev_sidx = None
        for row, edu_idx in zip(conll_rows, edu_idx_per_token):
            sidx = row[0]
            if prev_sidx is not None and sidx != prev_sidx:
                f.write("\n")

            cols = list(row)
            if len(cols) == 8:
                cols.append("")
            elif len(cols) > 9:
                cols = cols[:9]
            cols = cols[:9]
            cols.append(str(edu_idx))
            f.write("\t".join(cols) + "\n")
            prev_sidx = sidx


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create .merge from .conll + EDU-lines text file."
    )
    parser.add_argument("--conll", required=True, help="Input .conll file")
    parser.add_argument(
        "--edu-lines",
        required=True,
        help="Text file where each non-empty line is one EDU",
    )
    parser.add_argument("--output", required=True, help="Output .merge file")
    args = parser.parse_args()

    conll_rows = read_conll(args.conll)
    edu_lines = read_edu_lines(args.edu_lines)
    edu_idx_per_token = align_edus_to_conll(conll_rows, edu_lines)
    write_merge(conll_rows, edu_idx_per_token, args.output)

    print("Wrote merge file: {}".format(args.output))
    print("Tokens: {}, EDUs: {}".format(len(conll_rows), len(edu_lines)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise SystemExit(1)
