#!/usr/bin/env python3
"""
Preprocess raw Midjourney Discord exports into a clean prompt dataset.

This script is a cleaned-up release version of the notebook pipeline used in the paper.
It extracts prompts from the raw Discord export, filters noisy/non-English rows, assigns
Midjourney versions, applies the <=77 token constraint used by CLIP-style text encoders,
and writes the result to a Parquet file.

Expected input columns (raw CSV):
    - Content
    - Attachments
    - Date

Default output columns:
    - Prompt
    - Version

Optional:
    - Attachments (if --keep-attachments is passed)

Example:
    python preprocess_midjourney_dataset.py \
        --input general-9.csv \
        --output cleaned_midjourney.parquet \
        --fasttext-model lid.176.bin
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

import fasttext
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

tqdm.pandas()


URL_PATTERN = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # geometric shapes extended
    "\U0001F800-\U0001F8FF"  # supplemental arrows
    "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
    "\u2702-\u27B0"          # dingbats
    "]+",
    flags=re.UNICODE,
)

# Order matters: longer version strings must come first.
VERSION_REGEX = re.compile(
    r"--v(?:ersion)?\s*(6\.1|6|5\.2|5\.1|5|4|3|2|1)\b",
    flags=re.IGNORECASE,
)

# Release windows used when the prompt does not explicitly specify a version.
DATE_RANGES = {
    "1": ("2022-03-15", "2022-04-11"),
    "2": ("2022-04-13", "2022-07-24"),
    "3": ("2022-07-26", "2022-11-04"),
    "4": ("2022-11-06", "2023-03-14"),
    "5": ("2023-03-16", "2023-05-02"),
    "5.1": ("2023-05-04", "2023-06-21"),
    "5.2": ("2023-06-23", "2023-12-20"),
    "6": ("2023-12-22", "2024-07-30"),
    "6.1": ("2024-08-01", "2099-07-30"),
}
DATE_RANGES = {
    version: (pd.Timestamp(start), pd.Timestamp(end))
    for version, (start, end) in DATE_RANGES.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Midjourney prompt data.")
    parser.add_argument("--input", type=Path, required=True, help="Path to raw CSV export.")
    parser.add_argument("--output", type=Path, required=True, help="Path to output Parquet file.")
    parser.add_argument(
        "--fasttext-model",
        type=Path,
        required=True,
        help="Path to the FastText language ID model (e.g., lid.176.bin).",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
        help="Tokenizer used for the <=77 token filter.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=77,
        help="Maximum allowed tokenized prompt length.",
    )
    parser.add_argument(
        "--keep-attachments",
        action="store_true",
        help="Keep the Attachments column in the output for internal use.",
    )
    return parser.parse_args()


def extract_prompt(content: str) -> Optional[str]:
    """Extract the prompt enclosed in **...** from the raw Discord Content field."""
    if not isinstance(content, str):
        return None
    match = re.search(r"\*\*(.*?)\*\*", content, flags=re.DOTALL)
    if not match:
        return None
    prompt = match.group(1).strip()
    return prompt if prompt else None


def is_nonempty_string(value: object) -> bool:
    if pd.isna(value):
        return False
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return stripped != "" and stripped.lower() != "nan"


def contains_url(text: str) -> bool:
    return bool(URL_PATTERN.search(text))


def has_emoji(text: str) -> bool:
    return bool(EMOJI_PATTERN.search(text))


def is_not_float(value: str) -> bool:
    try:
        float(value)
        return False
    except (ValueError, TypeError):
        return True


def is_mostly_digits(text: str) -> bool:
    if not text:
        return False
    digit_count = sum(char.isdigit() for char in text)
    return digit_count > len(text) / 2


def infer_version(content: str, timestamp: pd.Timestamp) -> Optional[str]:
    """
    Infer Midjourney version.

    Priority:
      1) explicit --v / --version parameter in the raw Content
      2) fallback to timestamp-based release windows
    """
    if isinstance(content, str):
        match = VERSION_REGEX.search(content)
        if match:
            return match.group(1)

    if pd.isna(timestamp):
        return None

    for version, (start, end) in DATE_RANGES.items():
        if start <= timestamp <= end:
            return version

    return None


class EnglishFilter:
    """
    Keep prompts that are English with reasonable confidence.

    This keeps the spirit of the original notebook:
      - FastText-based language detection
      - a confidence / margin check
      - a small word-level fallback for ambiguous cases
    """

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.4,
        margin_threshold: float = 0.15,
        min_word_len: int = 3,
        max_words_to_check: int = 8,
        min_english_word_fraction: float = 0.5,
    ) -> None:
        self.model = fasttext.load_model(str(model_path))
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.min_word_len = min_word_len
        self.max_words_to_check = max_words_to_check
        self.min_english_word_fraction = min_english_word_fraction

    def predict_topk(self, text: str, k: int = 2) -> list[tuple[str, float]]:
        labels, probs = self.model.predict(text, k=k)
        return [(label.replace("__label__", ""), float(prob)) for label, prob in zip(labels, probs)]

    def is_english(self, text: str) -> bool:
        if not isinstance(text, str):
            return False

        cleaned = text.replace("\n", " ").strip()
        if not cleaned:
            return False

        preds = self.predict_topk(cleaned, k=2)
        top_lang, top_prob = preds[0]
        second_prob = preds[1][1] if len(preds) > 1 else 0.0
        margin = top_prob - second_prob

        # Confident English
        if top_lang == "en" and top_prob >= self.confidence_threshold and margin >= self.margin_threshold:
            return True

        # Confident non-English
        if top_lang != "en" and top_prob >= self.confidence_threshold and margin >= self.margin_threshold:
            return False

        # Ambiguous case: inspect a few words
        words = re.findall(r"[A-Za-z]+", cleaned)
        words = [w for w in words if len(w) >= self.min_word_len][: self.max_words_to_check]

        if not words:
            # If there are no alphabetic words to inspect, fall back to the whole-text prediction.
            return top_lang == "en"

        english_votes = 0
        for word in words:
            word_lang = self.predict_topk(word, k=1)[0][0]
            if word_lang == "en":
                english_votes += 1

        return (english_votes / len(words)) >= self.min_english_word_fraction


def build_token_length_filter(tokenizer_name: str, max_tokens: int):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def keep(text: str) -> bool:
        if not isinstance(text, str):
            return False
        token_ids = tokenizer.encode(text)
        return len(token_ids) <= max_tokens

    return keep


def summarize_step(name: str, before: int, after: int) -> None:
    removed = before - after
    print(f"{name:<35} {before:>10,} -> {after:>10,}  (removed {removed:>10,})")


def preprocess_dataframe(
    df: pd.DataFrame,
    english_filter: EnglishFilter,
    keep_token_length,
) -> pd.DataFrame:
    required_columns = {"Content", "Attachments", "Date"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required input columns: {sorted(missing)}")

    # 1) Extract prompt from Content
    df = df.copy()
    df["Prompt"] = df["Content"].apply(extract_prompt)

    before = len(df)
    mask = df["Prompt"].apply(is_nonempty_string) & df["Attachments"].apply(is_nonempty_string)
    df = df[mask].copy()
    summarize_step("Keep rows with prompt+attachment", before, len(df))

    # 2) Remove prompts containing external URLs
    before = len(df)
    df = df[~df["Prompt"].apply(contains_url)].copy()
    summarize_step("Remove URL-based prompts", before, len(df))

    # 3) Convert Date and infer version
    df["RealTime"] = pd.to_datetime(df["Date"], unit="s", errors="coerce")
    df["Version"] = df.apply(lambda row: infer_version(row["Content"], row["RealTime"]), axis=1)

    before = len(df)
    df = df[df["Version"].notna()].copy()
    summarize_step("Keep rows with inferred version", before, len(df))

    # 4) Deduplicate on raw content + attachment
    before = len(df)
    df = df.drop_duplicates(subset=["Content", "Attachments"]).copy()
    summarize_step("Drop duplicate rows", before, len(df))

    # 5) Remove prompts that are just floats
    before = len(df)
    df = df[df["Prompt"].apply(is_not_float)].copy()
    summarize_step("Remove float-only prompts", before, len(df))

    # 6) Remove emoji-heavy prompts
    before = len(df)
    df = df[~df["Prompt"].apply(has_emoji)].copy()
    summarize_step("Remove prompts with emojis", before, len(df))

    # 7) Remove mostly-digit prompts
    before = len(df)
    df = df[~df["Prompt"].apply(is_mostly_digits)].copy()
    summarize_step("Remove mostly-digit prompts", before, len(df))

    # 8) Keep English prompts
    before = len(df)
    df = df[df["Prompt"].progress_apply(english_filter.is_english)].copy()
    summarize_step("Keep English prompts", before, len(df))

    # 9) Apply <=77 token filter
    before = len(df)
    df = df[df["Prompt"].progress_apply(keep_token_length)].copy()
    summarize_step("Apply token-length filter", before, len(df))

    # Final cleanup
    df = df.reset_index(drop=True)

    return df


def main() -> None:
    args = parse_args()

    print("Loading raw data...")
    df = pd.read_csv(args.input)

    print("Loading FastText language ID model...")
    english_filter = EnglishFilter(args.fasttext_model)

    print(f"Loading tokenizer: {args.tokenizer_name}")
    keep_token_length = build_token_length_filter(args.tokenizer_name, args.max_tokens)

    print("Running preprocessing...")
    df = preprocess_dataframe(df, english_filter, keep_token_length)

    # Keep only columns needed for the released dataset by default.
    output_columns = ["Prompt", "Version"]
    if args.keep_attachments:
        output_columns.append("Attachments")

    df_out = df[output_columns].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(args.output, index=False)

    print(f"\nSaved {len(df_out):,} rows to: {args.output}")
    print("Output columns:", ", ".join(df_out.columns))


if __name__ == "__main__":
    main()