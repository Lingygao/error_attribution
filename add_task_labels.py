from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED_COLUMNS = ("mt_v_label", "pr_mt_labels", "pr_v_label")
TASK_COLUMNS = ("task1", "task2", "task3")


def split_label_cell(value: str | None) -> list[str]:
    if value is None:
        return []
    value = value.strip()
    if not value:
        return []
    return [part.strip().upper() for part in value.split("|") if part.strip()]


def is_all_c(value: str | None) -> bool:
    labels = split_label_cell(value)
    return bool(labels) and all(label == "C" for label in labels)


def is_empty_or_all_c(value: str | None) -> bool:
    labels = split_label_cell(value)
    return not labels or all(label == "C" for label in labels)


def add_task_labels(row: dict[str, str]) -> dict[str, str]:
    mt_v_label = row.get("mt_v_label", "")
    pr_mt_labels = row.get("pr_mt_labels", "")
    pr_v_label = row.get("pr_v_label", "")
    pr_v = pr_v_label.strip().upper()

    task1 = int(
        not (
            is_all_c(mt_v_label)
            and is_all_c(pr_mt_labels)
            and is_all_c(pr_v_label)
        )
    )
    task2 = str(int(is_empty_or_all_c(pr_mt_labels))) if pr_v == "I" else ""
    task3 = str(int(is_empty_or_all_c(pr_mt_labels))) if pr_v == "S" else ""

    labeled = dict(row)
    labeled["task1"] = str(task1)
    labeled["task2"] = task2
    labeled["task3"] = task3
    return labeled


def label_table(input_path: Path, output_path: Path) -> tuple[int, dict[str, int]]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} has no header")
        missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"{input_path} is missing required columns: {', '.join(missing)}")

        fieldnames = [name for name in reader.fieldnames if name not in TASK_COLUMNS]
        fieldnames.extend(TASK_COLUMNS)

        rows = [add_task_labels(row) for row in reader]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = {name: sum(1 for row in rows if row[name] == "1") for name in TASK_COLUMNS}
    return len(rows), counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append task1/task2/task3 labels to a speaker_pr_mt_vx label table CSV."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input label table CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output labeled CSV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    row_count, counts = label_table(args.input, args.output)
    print(f"Rows: {row_count}")
    print(f"task1=1: {counts['task1']}")
    print(f"task2=1: {counts['task2']}")
    print(f"task3=1: {counts['task3']}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
