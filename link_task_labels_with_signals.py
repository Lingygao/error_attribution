from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from text_normalization import text_norm_all


LABEL_WORD_COLUMN = "asr_word"
SIGNAL_WORD_COLUMN = "normalized_asr_word"
SIGNAL_SOURCE_COLUMNS = {"segment_id", "word_idx", "asr_word", "normalized_asr_word"}


@dataclass(frozen=True)
class LinkResult:
    output_path: Path
    log_path: Path
    label_rows: int
    signal_rows: int
    written_rows: int
    skipped_label_positions: int


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty or has no header")
        return list(reader.fieldnames), list(reader)


def normalize_words(rows: list[dict[str, str]], column: str) -> list[str]:
    words: list[str] = []
    for row in rows:
        normalized = text_norm_all(row.get(column, ""))
        parts = normalized.split()
        if len(parts) > 1:
            raise ValueError(
                f"Column {column} produced multiple normalized tokens from "
                f"{row.get(column, '')!r}: {parts}"
            )
        words.append(parts[0] if parts else "")
    return words


def insertion_index(label_columns: list[str]) -> int:
    for column in ("task3", "pr_v_label"):
        if column in label_columns:
            return label_columns.index(column) + 1
    return len(label_columns)


def output_columns(label_columns: list[str], signal_columns: list[str]) -> list[str]:
    insert_at = insertion_index(label_columns)
    return label_columns[:insert_at] + signal_columns + label_columns[insert_at:]


def signal_value_columns(signal_columns: list[str]) -> list[str]:
    return [column for column in signal_columns if column not in SIGNAL_SOURCE_COLUMNS]


def source_span(rows: list[dict[str, str]], start: int, end: int, word_column: str) -> str:
    values: list[str] = []
    for index in range(start, end):
        row = rows[index]
        position = row.get("asr_index") or row.get("word_idx") or str(index)
        values.append(f"{position}:{row.get(word_column, '')}")
    return " ".join(values)


def link_label_table_with_signals(
    label_csv: Path,
    signal_csv: Path,
    output_csv: Path | None = None,
    log_csv: Path | None = None,
) -> LinkResult:
    label_columns, label_rows = read_csv(label_csv)
    merged_signal_columns, signal_rows = read_csv(signal_csv)

    if LABEL_WORD_COLUMN not in label_columns:
        raise ValueError(f"{label_csv} is missing column {LABEL_WORD_COLUMN!r}")
    if SIGNAL_WORD_COLUMN not in merged_signal_columns:
        raise ValueError(f"{signal_csv} is missing column {SIGNAL_WORD_COLUMN!r}")

    signal_columns = signal_value_columns(merged_signal_columns)
    if not signal_columns:
        raise ValueError(f"{signal_csv} has no signal columns to append")

    label_words = normalize_words(label_rows, LABEL_WORD_COLUMN)
    signal_words = normalize_words(signal_rows, SIGNAL_WORD_COLUMN)
    matcher = SequenceMatcher(None, label_words, signal_words, autojunk=False)

    label_to_signal: dict[int, int] = {}
    skipped_blocks: list[dict[str, str]] = []
    for tag, label_start, label_end, signal_start, signal_end in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(label_end - label_start):
                label_to_signal[label_start + offset] = signal_start + offset
            continue

        skipped_blocks.append(
            {
                "tag": tag,
                "label_start": str(label_start),
                "label_end": str(label_end),
                "signal_start": str(signal_start),
                "signal_end": str(signal_end),
                "label_words": " ".join(label_words[label_start:label_end]),
                "signal_words": " ".join(signal_words[signal_start:signal_end]),
                "label_sources": source_span(
                    label_rows, label_start, label_end, LABEL_WORD_COLUMN
                ),
                "signal_sources": source_span(
                    signal_rows, signal_start, signal_end, SIGNAL_WORD_COLUMN
                ),
            }
        )

    output_csv = output_csv or label_csv.with_name(f"{label_csv.stem}_with_signals.csv")
    log_csv = log_csv or label_csv.with_name(f"{label_csv.stem}_signal_link_skipped.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    log_csv.parent.mkdir(parents=True, exist_ok=True)

    final_columns = output_columns(label_columns, signal_columns)
    with output_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=final_columns)
        writer.writeheader()
        for label_index, label_row in enumerate(label_rows):
            signal_index = label_to_signal.get(label_index)
            if signal_index is None:
                continue

            signal_row = signal_rows[signal_index]
            output_row = dict(label_row)
            for column in signal_columns:
                output_row[column] = signal_row[column]
            writer.writerow(output_row)

    log_columns = [
        "label_csv",
        "signal_csv",
        "tag",
        "label_start",
        "label_end",
        "signal_start",
        "signal_end",
        "label_words",
        "signal_words",
        "label_sources",
        "signal_sources",
    ]
    with log_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=log_columns)
        writer.writeheader()
        for block in skipped_blocks:
            writer.writerow(
                {
                    "label_csv": str(label_csv),
                    "signal_csv": str(signal_csv),
                    **block,
                }
            )

    written_rows = len(label_to_signal)
    return LinkResult(
        output_path=output_csv,
        log_path=log_csv,
        label_rows=len(label_rows),
        signal_rows=len(signal_rows),
        written_rows=written_rows,
        skipped_label_positions=len(label_rows) - written_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Link a task/label table with a merged signal table by normalized asr_word. "
            "Unmatched blocks are dropped; label-table indices are preserved."
        )
    )
    parser.add_argument("label_csv", type=Path, help="Input label/task-label CSV.")
    parser.add_argument("signal_csv", type=Path, help="Input *_merged_signals.csv.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV. Default: <label_stem>_with_signals.csv next to label CSV.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Skipped-block log CSV. Default: <label_stem>_signal_link_skipped.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = link_label_table_with_signals(
        args.label_csv,
        args.signal_csv,
        output_csv=args.output,
        log_csv=args.log,
    )
    print(
        f"{result.output_path} "
        f"(label_rows={result.label_rows}, "
        f"signal_rows={result.signal_rows}, "
        f"written_rows={result.written_rows}, "
        f"skipped_label_positions={result.skipped_label_positions}, "
        f"log={result.log_path})"
    )


if __name__ == "__main__":
    main()
