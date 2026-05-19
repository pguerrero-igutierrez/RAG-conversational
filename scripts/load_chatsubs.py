"""
load_chatsubs.py
----------------
Loads the ChatSubs dataset (Kharitonova et al., 2023) and produces
conversational Spanish turns (label=0) mirroring the SQAC splits:

  Router train  → data/processed/router_train_chitchat.jsonl  (15,036 turns)
  Router val    → data/processed/router_val_chitchat.jsonl    ( 1,664 turns)

The held-out chitchat test set is intentionally NOT sampled from ChatSubs.
Use load_microsoft_chitchat.py to create:

  Unified test  → data/processed/test_chitchat.jsonl          (100 turns)

Dataset
-------
  ChatSubs: Kharitonova et al., 2023
  https://zenodo.org/records/8220853  —  License: CC BY-NC 4.0
  206,706 JSONL files, 96M+ turns. This script uses Spanish
  (open_subtitles_es/) only.

Usage
-----
  mkdir -p data/raw
  wget -O data/raw/ChatSubs.tar.gz \
    https://zenodo.org/records/8220853/files/ChatSubs.tar.gz

  python scripts/load_chatsubs.py
  python scripts/load_chatsubs.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import random
import re
import tarfile
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parents[1]
RAW_DIR       = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
ARCHIVE_PATH  = RAW_DIR / "ChatSubs.tar.gz"

ROUTER_TRAIN_PATH = PROCESSED_DIR / "router_train_chitchat.jsonl"
ROUTER_VAL_PATH   = PROCESSED_DIR / "router_val_chitchat.jsonl"
TEST_PATH         = PROCESSED_DIR / "test_chitchat.jsonl"

Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)

# ── Split sizes — mirror SQAC exactly ─────────────────────────────────────
ROUTER_TRAIN_SIZE = 15_036
ROUTER_VAL_SIZE   = 1_664
TOTAL_NEEDED      = ROUTER_TRAIN_SIZE + ROUTER_VAL_SIZE  # 16,700

RANDOM_SEED = 42
LANG_PREFIX = "open_subtitles_es/"
MAX_FILES   = 50_000   # scan enough files to collect 15k+ clean turns
PROGRESS_EVERY_FILES = 25

# ── Filtering ──────────────────────────────────────────────────────────────
MIN_WORDS = 4
MAX_WORDS = 30

_YEAR_RE        = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_FACTUAL_RE     = re.compile(
    r"\b(cuándo|quién|quiénes|cuál|cuáles|dónde|en qué año|"
    r"qué es|qué son|cuántos|cuántas|cómo se llama|"
    r"qué países|qué idiomas)\b",
    re.IGNORECASE,
)
_PROPER_NOUN_RE = re.compile(r"([A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+\s){2,}")


def _iter_record_texts(record: dict) -> list[str]:
    if "turns" in record:
        return [
            turn.get("text", "").strip()
            for turn in record.get("turns", [])
            if isinstance(turn, dict)
        ]

    texts: list[str] = []
    for dialogue in record.get("dialogues", []):
        if not isinstance(dialogue, str):
            continue
        texts.extend(line.strip() for line in dialogue.splitlines())
    return texts


def _is_valid_turn(text: str) -> bool:
    words = text.split()
    if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
        return False
    if _YEAR_RE.search(text):
        return False
    if _FACTUAL_RE.search(text):
        return False
    if _PROPER_NOUN_RE.search(text):
        return False
    return True


# ── Loader ─────────────────────────────────────────────────────────────────
def log(message: str) -> None:
    print(message, flush=True)


def load_chatsubs(
    archive_path: Path = ARCHIVE_PATH,
    total_needed: int = TOTAL_NEEDED,
    seed:         int = RANDOM_SEED,
    max_files:    int = MAX_FILES,
) -> list[str]:
    """
    Stream Spanish turns from ChatSubs.tar.gz and return a random
    sample of *total_needed* clean conversational turns.
    """
    if not Path(archive_path).exists():
        raise FileNotFoundError(
            f"ChatSubs archive not found at {archive_path}.\n"
            "Download with:\n"
            "  wget -O data/raw/ChatSubs.tar.gz "
            "https://zenodo.org/records/8220853/files/ChatSubs.tar.gz"
        )

    log(f"[ChatSubs] Opening archive: {archive_path}")
    log(f"[ChatSubs] Need {total_needed:,} clean turns — scanning up to "
        f"{max_files:,} Spanish files …")

    candidates: list[str] = []
    seen: set[str]        = set()
    files_scanned         = 0
    archive_members_seen  = 0

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            archive_members_seen += 1
            if not member.name.endswith(".jsonl"):
                continue
            if not member.name.lstrip("./").startswith(LANG_PREFIX):
                continue

            files_scanned += 1
            if files_scanned > max_files:
                log(f"[ChatSubs] Reached max_files={max_files} — stopping.")
                break

            try:
                f = tar.extractfile(member)
                if f is None:
                    continue
                records = [
                    json.loads(line)
                    for line in f.read().decode("utf-8").splitlines()
                    if line.strip()
                ]
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            for record in records:
                for text in _iter_record_texts(record):
                    if not text or text in seen:
                        continue
                    if _is_valid_turn(text):
                        candidates.append(text)
                        seen.add(text)

            if files_scanned % PROGRESS_EVERY_FILES == 0:
                log(f"[ChatSubs] Scanned {files_scanned:,} Spanish files "
                    f"({archive_members_seen:,} archive members seen) — "
                    f"{len(candidates):,} valid turns …")

            # Stop once we have a comfortable surplus
            if len(candidates) >= total_needed * 2:
                log("[ChatSubs] Sufficient candidates — stopping early.")
                break

    log(f"[ChatSubs] Scan complete: {files_scanned:,} Spanish files, "
        f"{len(candidates):,} valid candidates.")

    if len(candidates) < total_needed:
        raise RuntimeError(
            f"Only found {len(candidates):,} valid ChatSubs candidates, "
            f"but need {total_needed:,}. Check the archive layout/language "
            "prefix or increase --max_files."
        )

    random.seed(seed)
    return random.sample(candidates, total_needed)


# ── Helpers ────────────────────────────────────────────────────────────────
def make_records(texts: list[str], id_prefix: str) -> list[dict]:
    return [
        {
            "id":     f"{id_prefix}_{i:05d}",
            "text":   text,
            "label":  0,
            "source": "chatsubs",
        }
        for i, text in enumerate(texts)
    ]


def write_jsonl(records: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ChatSubs] Saved {len(records):,} turns → {path}")


# ── Main ───────────────────────────────────────────────────────────────────
def main(dry_run: bool = False, max_files: int = MAX_FILES) -> None:
    sampled = load_chatsubs(max_files=max_files)

    i = 0
    router_train = sampled[i: i + ROUTER_TRAIN_SIZE]; i += ROUTER_TRAIN_SIZE
    router_val   = sampled[i: i + ROUTER_VAL_SIZE];   i += ROUTER_VAL_SIZE
    if dry_run:
        print("\n── Sample turns (dry run) ────────────────────────────────")
        for text in router_train[:5]:
            print(f"  [train] {text}")
        print(f"\nTotal sampled: {len(sampled):,}. Output not written (--dry_run).")
        return

    write_jsonl(make_records(router_train, "cc_train"), ROUTER_TRAIN_PATH)
    write_jsonl(make_records(router_val,   "cc_val"),   ROUTER_VAL_PATH)

    print("\n" + "─" * 55)
    print("  ChatSubs data preparation complete")
    print("─" * 55)
    print(f"  Router train  : {len(router_train):,} turns  (label=0, full set)")
    print(f"  Router val    : {len(router_val):,} turns  (label=0)")
    print("  Unified test  : handled by load_microsoft_chitchat.py")
    print("─" * 55 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Spanish chitchat turns from ChatSubs."
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print samples without writing output files.",
    )
    parser.add_argument(
        "--max_files", type=int, default=MAX_FILES,
        help=f"Max JSON files to scan (default: {MAX_FILES:,}).",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run, max_files=args.max_files)
