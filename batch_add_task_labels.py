from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from add_task_labels import label_table


SUMMARY_FIELDS = [
    "speaker",
    "status",
    "rows",
    "task1_ones",
    "task2_ones",
    "task3_ones",
    "input_path",
    "output_path",
    "reason",
]


def infer_output_root(input_root: Path) -> Path:
    name = input_root.name
    if name.endswith("_task"):
        return input_root.with_name(f"{name}_labeled")
    return input_root.with_name(f"{name}_task")


def speaker_name(path: Path) -> str:
    return path.name.removesuffix("_txt")


def label_table_name(input_path: Path) -> str:
    stem = input_path.stem
    if stem.endswith("_task_label_table"):
        return f"{stem}.csv"
    if stem.endswith("_label_table"):
        return f"{stem.removesuffix('_label_table')}_task_label_table.csv"
    return f"{stem}_task.csv"


def find_input_csv(speaker_dir: Path, speaker: str) -> Path | None:
    preferred = speaker_dir / f"{speaker}_{speaker_dir.parent.name}_label_table.csv"
    if preferred.exists():
        return preferred

    matches = sorted(speaker_dir.glob("*_label_table.csv"))
    if len(matches) == 1:
        return matches[0]

    named = [path for path in matches if path.name.startswith(f"{speaker}_")]
    if len(named) == 1:
        return named[0]

    return None


def discover_speaker_dirs(input_root: Path, only_speaker: str | None) -> list[Path]:
    if only_speaker:
        speaker = only_speaker.removesuffix("_txt")
        candidates = [input_root / speaker, input_root / f"{speaker}_txt"]
        return [path for path in candidates if path.is_dir()]

    return sorted(path for path in input_root.iterdir() if path.is_dir())


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch append task1/task2/task3 columns to every speaker label table "
            "under a pr_mt_v2/pr_mt_v3-style folder."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Folder containing one subfolder per speaker, e.g. .../alignment/pr_mt_v2.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Output folder. Default: sibling folder named <input-root>_task.",
    )
    parser.add_argument(
        "--speaker",
        help="Run only one speaker, e.g. fn000049 or fn000049_txt.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output CSV files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = (args.output_root or infer_output_root(input_root)).resolve()

    if not input_root.exists():
        print(f"Input root does not exist: {input_root}", file=sys.stderr)
        return 1
    if not input_root.is_dir():
        print(f"Input root is not a directory: {input_root}", file=sys.stderr)
        return 1

    speaker_dirs = discover_speaker_dirs(input_root, args.speaker)
    summary_rows: list[dict[str, object]] = []
    ok = 0
    skipped = 0
    failed = 0
    total_rows = 0

    for speaker_dir in speaker_dirs:
        speaker = speaker_name(speaker_dir)
        output_speaker_dir = output_root / speaker_dir.name
        input_csv = find_input_csv(speaker_dir, speaker)
        if input_csv is None:
            failed += 1
            summary_rows.append(
                {
                    "speaker": speaker,
                    "status": "failed",
                    "rows": 0,
                    "task1_ones": 0,
                    "task2_ones": 0,
                    "task3_ones": 0,
                    "input_path": "",
                    "output_path": output_speaker_dir,
                    "reason": f"missing or ambiguous *_label_table.csv in {speaker_dir}",
                }
            )
            continue

        output_csv = output_speaker_dir / label_table_name(input_csv)
        if output_csv.exists() and not args.overwrite:
            skipped += 1
            summary_rows.append(
                {
                    "speaker": speaker,
                    "status": "skipped",
                    "rows": 0,
                    "task1_ones": 0,
                    "task2_ones": 0,
                    "task3_ones": 0,
                    "input_path": input_csv,
                    "output_path": output_csv,
                    "reason": "output exists; pass --overwrite to regenerate",
                }
            )
            continue

        try:
            row_count, counts = label_table(input_csv, output_csv)
        except Exception as error:
            failed += 1
            summary_rows.append(
                {
                    "speaker": speaker,
                    "status": "failed",
                    "rows": 0,
                    "task1_ones": 0,
                    "task2_ones": 0,
                    "task3_ones": 0,
                    "input_path": input_csv,
                    "output_path": output_csv,
                    "reason": repr(error),
                }
            )
            continue

        ok += 1
        total_rows += row_count
        summary_rows.append(
            {
                "speaker": speaker,
                "status": "ok",
                "rows": row_count,
                "task1_ones": counts["task1"],
                "task2_ones": counts["task2"],
                "task3_ones": counts["task3"],
                "input_path": input_csv,
                "output_path": output_csv,
                "reason": "",
            }
        )

    summary_path = output_root / f"{input_root.name}_task_summary.csv"
    write_summary(summary_path, summary_rows)

    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Speakers found: {len(speaker_dirs)}")
    print(f"Completed: {ok}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Rows: {total_rows}")
    print(f"Summary: {summary_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
