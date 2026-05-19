"""
Train a SetFit router from scratch on oracle-labelled examples.

This deliberately starts from the base sentence-transformer encoder, not from
any previously trained SetFit router checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT_DIR / "corpus"
MODEL_DIR = ROOT_DIR / "models" / "router_oracle" / "setfit"
DEFAULT_DATA_PATH = CORPUS_DIR / "oracle_train_sqac_64_per_label.jsonl"
DEFAULT_SAVE_DIR = MODEL_DIR / "64_per_label"

sys.path.insert(0, str(ROOT_DIR / "scripts"))

from router import (  # noqa: E402
    ENCODER_MODEL,
    TOKENIZER_KWARGS,
    compute_metrics,
)


def _load_oracle_records(path: Path) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["query"])
            labels.append(int(rec["oracle_label"]))
    return texts, labels


def train_oracle_setfit(
    data_path: Path,
    save_dir: Path,
    num_epochs: int,
    num_iterations: int,
    batch_size: int,
    dry_run: bool,
) -> None:
    from collections import Counter

    texts, labels = _load_oracle_records(data_path)
    counts = Counter(labels)
    print(
        f"[OracleSetFit] Loaded {len(texts)} examples from {data_path} | "
        f"labels={dict(sorted(counts.items()))}",
        flush=True,
    )

    if dry_run:
        print("[OracleSetFit] dry_run=true; exiting before model load.")
        return

    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    save_dir.mkdir(parents=True, exist_ok=True)
    train_ds = Dataset.from_dict({"text": texts, "label": labels})

    print(
        f"[OracleSetFit] Starting from base encoder: {ENCODER_MODEL}",
        flush=True,
    )
    model = SetFitModel.from_pretrained(
        ENCODER_MODEL,
        processor_kwargs=TOKENIZER_KWARGS,
    )

    args = TrainingArguments(
        output_dir=str(save_dir / "checkpoints"),
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_iterations=num_iterations,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        metric="f1",
        column_mapping={"text": "text", "label": "label"},
    )

    started_at = time.time()
    trainer.train()
    elapsed = time.time() - started_at

    train_preds = model.predict(texts).tolist()
    metrics = compute_metrics(train_preds, labels)

    model.save_pretrained(str(save_dir))
    summary = {
        "model": "setfit_oracle",
        "base_encoder": ENCODER_MODEL,
        "data_path": str(data_path),
        "save_dir": str(save_dir),
        "n_examples": len(texts),
        "label_counts": dict(sorted(counts.items())),
        "num_epochs": num_epochs,
        "num_iterations": num_iterations,
        "batch_size": batch_size,
        "train_time_s": round(elapsed, 1),
        "train_metrics": {
            k: v for k, v in metrics.items() if k != "report"
        },
    }
    with open(save_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OracleSetFit] Saved model → {save_dir}", flush=True)
    print(f"[OracleSetFit] Train macro F1: {metrics['macro_f1']:.4f}")
    print(metrics["report"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SetFit from scratch on oracle-labelled examples.",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--save_dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--num_iterations", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_oracle_setfit(
        data_path=args.data,
        save_dir=args.save_dir,
        num_epochs=args.num_epochs,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
