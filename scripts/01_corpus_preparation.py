"""
Step 1: Corpus Preparation (Local Files Version)
==================================================
Parses your locally downloaded plain-text Bible files.

Expected file layout — one OT file + one NT file per language:

    Matthew
    1:1  A record of the genealogy...
    1:2  Abraham was the father of...

    Mark
    1:1  The beginning of the good news...

Rules:
  - A line with no chapter:verse pattern = book name
  - A line matching  digits:digits  text = verse
  - Blank lines are ignored

NOTE: Both English (NIV) and Chinese (CUV Traditional) files use
English book titles (Genesis, Matthew, etc.), so a single shared
book name map handles both languages.

Outputs:
  - data/english_verses.csv   : verse_id, text
  - data/chinese_verses.csv   : verse_id, text
  - data/aligned_verses.csv   : verse_id, english_text, chinese_text

Usage:
  python 01_corpus_preparation.py
"""

import re
import pandas as pd
from pathlib import Path

# ─── CONFIGURE THESE PATHS ────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "bible_data"
DATA_DIR.mkdir(exist_ok=True)

FILES = {
    "english": [
        # DATA_DIR / "Bible-OT-norm.en-niv.txt",
        # DATA_DIR / "Bible-NT-norm.en-niv.txt",
        DATA_DIR / "Bible-OT-norm.en-kjv.txt",
        DATA_DIR / "Bible-NT-norm.en-kjv.txt",
    ],
    "chinese": [
        DATA_DIR / "Bible-OT-norm.cht-cuv.txt",
        DATA_DIR / "Bible-NT-norm.cht-cuv.txt",
    ],
}

# ─── Shared Book Name Map (used for BOTH languages) ───────────────────────────
# Both NIV and CUV Traditional files use English book titles.
# Covers common NIV header variants (e.g. "Song of Songs" vs "Song of Solomon").

BOOK_ABBREV = {
    # Pentateuch
    "genesis": "Gen", "exodus": "Exod", "leviticus": "Lev",
    "numbers": "Num", "deuteronomy": "Deut",
    # History
    "joshua": "Josh", "judges": "Judg", "ruth": "Ruth",
    "1 samuel": "1Sam", "2 samuel": "2Sam",
    "1 kings": "1Kgs", "2 kings": "2Kgs",
    "1 chronicles": "1Chr", "2 chronicles": "2Chr",
    "ezra": "Ezra", "nehemiah": "Neh", "esther": "Esth",
    # Poetry & Wisdom
    "job": "Job", "psalms": "Ps", "psalm": "Ps",
    "proverbs": "Prov", "ecclesiastes": "Eccl",
    "song of songs": "Song", "song of solomon": "Song",
    # Major Prophets
    "isaiah": "Isa", "jeremiah": "Jer", "lamentations": "Lam",
    "ezekiel": "Ezek", "daniel": "Dan",
    # Minor Prophets
    "hosea": "Hos", "joel": "Joel", "amos": "Amos",
    "obadiah": "Obad", "jonah": "Jonah", "micah": "Mic",
    "nahum": "Nah", "habakkuk": "Hab", "zephaniah": "Zeph",
    "haggai": "Hag", "zechariah": "Zech", "malachi": "Mal",
    # Gospels & Acts
    "matthew": "Matt", "mark": "Mark", "luke": "Luke",
    "john": "John", "acts": "Acts",
    # Pauline Epistles
    "romans": "Rom",
    "1 corinthians": "1Cor", "2 corinthians": "2Cor",
    "galatians": "Gal", "ephesians": "Eph", "philippians": "Phil",
    "colossians": "Col",
    "1 thessalonians": "1Thess", "2 thessalonians": "2Thess",
    "1 timothy": "1Tim", "2 timothy": "2Tim",
    "titus": "Titus", "philemon": "Phlm",
    # General Epistles
    "hebrews": "Heb", "james": "Jas",
    "1 peter": "1Pet", "2 peter": "2Pet",
    "1 john": "1John", "2 john": "2John", "3 john": "3John",
    "jude": "Jude", "revelation": "Rev",
}

# Verse line pattern: "1:1 text" or "1:1  text"
VERSE_RE = re.compile(r"^(\d+):(\d+)\s+(.+)$")


# ─── Parser ───────────────────────────────────────────────────────────────────

def parse_plain_text(file_paths: list, lang_label: str) -> pd.DataFrame:
    """
    Parse one or more plain-text Bible files into a DataFrame.

    Returns DataFrame: [verse_id, text]
    verse_id format: Abbrev.Chapter.Verse  e.g. Matt.1.1
    """
    records = []
    current_abbrev = None
    unmatched = set()

    for fpath in file_paths:
        fpath = Path(fpath)
        if not fpath.exists():
            raise FileNotFoundError(
                f"\n  File not found: {fpath}\n"
                f"  Place your files in data/ and update the FILES dict above."
            )

        print(f"  [{lang_label}] Reading {fpath.name} ...")
        raw = fpath.read_text(encoding="utf-8", errors="replace")

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue

            m = VERSE_RE.match(line)
            if m:
                if current_abbrev is None:
                    continue   # verse before any book header — skip
                chap, verse, text = m.groups()
                # Strip CUV translator notes: 〔...〕 and (...)
                # e.g. 〔後裔子孫原文都作兒子下同〕 — editorial glosses, not verse text
                # text = re.sub(r"〔[^〕]*〕", "", text)  # CUV-style brackets
                # text = re.sub(r"（[^）]*）", "", text)  # fullwidth parentheses
                # For Chinese preprocessing
                # Remove translator notes (square brackets)
                text = re.sub(r"〔[^〕]*〕", "", text)
                # Keep disputed verses (full-width parens)
                # But strip the parens themselves if entire verse
                if text.startswith("（") and text.endswith("）"):
                    text = text[1:-1]  # Keep content, remove wrapping parens

                text = text.strip()

                # In Step 1, filter out 見上節 verses
                if text.strip() in ["見上節", "〔見上節〕", "（見上節）"]:
                    continue  # Skip this verse

                if not text:
                    continue   # skip if entire verse was a note
                records.append({
                    "verse_id": f"{current_abbrev}.{chap}.{verse}",
                    "text":     text,
                })
            else:
                # Book name line — lowercase and strip punctuation, then look up
                key = re.sub(r"[^\w\s]", "", line).strip().lower()
                if key in BOOK_ABBREV:
                    current_abbrev = BOOK_ABBREV[key]
                else:
                    unmatched.add(line)

    df = pd.DataFrame(records)

    if unmatched:
        print(f"\n  [{lang_label}] WARNING — unrecognized header lines "
              f"(add to BOOK_ABBREV if these are book names):")
        for b in sorted(unmatched)[:20]:
            print(f"      '{b}'")

    n_books = df["verse_id"].str.split(".").str[0].nunique() if len(df) else 0
    print(f"  [{lang_label}] {len(df):,} verses parsed across {n_books} books.")
    return df


# ─── Alignment ────────────────────────────────────────────────────────────────

def align_corpora(en_df: pd.DataFrame, zh_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        en_df.rename(columns={"text": "english_text"}),
        zh_df.rename(columns={"text": "chinese_text"}),
        on="verse_id",
        how="inner",
    )
    print(f"\n  Alignment: {len(merged):,} matched verse pairs")
    print(f"  EN-only (no ZH match): {len(en_df) - len(merged):,}")
    print(f"  ZH-only (no EN match): {len(zh_df) - len(merged):,}")
    if (len(en_df) - len(merged)) > 500:
        print("  WARNING: high mismatch — check that all book names are mapped.")
    return merged


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Step 1: Corpus Preparation (Local Plain-Text Files)")
    print("  English : NIV")
    print("  Chinese : CUV Traditional (共用英文書名)")
    print("=" * 60)

    en_df = parse_plain_text(FILES["english"], "EN")
    zh_df = parse_plain_text(FILES["chinese"], "ZH")

    en_df.to_csv(DATA_DIR / "english_verses.csv", index=False, encoding="utf-8")
    zh_df.to_csv(DATA_DIR / "chinese_verses.csv", index=False, encoding="utf-8")
    print(f"\n  Saved: english_verses.csv  ({len(en_df):,} verses)")
    print(f"  Saved: chinese_verses.csv  ({len(zh_df):,} verses)")

    aligned = align_corpora(en_df, zh_df)
    aligned.to_csv(DATA_DIR / "aligned_verses.csv", index=False, encoding="utf-8")
    print(f"  Saved: aligned_verses.csv  ({len(aligned):,} pairs)")

    print("\n── Sample rows ──")
    print(aligned.head(3).to_string(index=False))
    print("\n✓ Step 1 complete. All downstream scripts unchanged.\n")


if __name__ == "__main__":
    main()
