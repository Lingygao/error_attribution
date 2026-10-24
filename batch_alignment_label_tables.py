from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from batch_power_alignment import DEFAULT_OUTPUT_ROOT
from build_alignment_label_table import rows_for_speaker_variant


TASK_TO_VARIANT = {
    "pr_mt_v2": "v2",
    "pr_mt_v3": "v3",
}

FIELDNAMES = [
    "speaker",
    "variant",
    "asr_index",
    "asr_word",
    "mt_words",
    "pr_mt_prompt_words",
    "pr_v_prompt_words",
    "mt_v_label",
    "pr_mt_labels",
    "pr_v_label",
    "source_mt_v_json",
    "source_pr_mt_json",
    "source_pr_v_json",
]

ERROR_FIELDS = [
    "task",
    "speaker",
    "stage",
    "reason",
    "output_path",
]


def speaker_name(path: Path) -> str:
    return path.name.removesuffix("_txt")


def existing_speaker_dirs(alignment_root: Path, task_dir: str) -> set[str]:
    root = alignment_root / task_dir
    if not root.exists():
        return set()
    return {speaker_name(path) for path in root.iterdir() if path.is_dir()}


def discover_speakers(
    alignment_root: Path,
    selected_tasks: list[str],
    only_speaker: str | None,
) -> list[str]:
    if only_speaker:
        return [only_speaker.removesuffix("_txt")]

    speakers: set[str] = set()
    for task in selected_tasks:
        variant = TASK_TO_VARIANT[task]
        speakers |= existing_speaker_dirs(alignment_root, "pr_mt")
        speakers |= existing_speaker_dirs(alignment_root, f"mt_{variant}")
        speakers |= existing_speaker_dirs(alignment_root, f"pr_{variant}")
    return sorted(speakers)


def missing_inputs(alignment_root: Path, speaker: str, variant: str) -> list[str]:
    missing = []
    for task_dir in ("pr_mt", f"mt_{variant}", f"pr_{variant}"):
        speaker_dir = alignment_root / task_dir / speaker
        if not speaker_dir.exists():
            missing.append(f"missing directory: {speaker_dir}")
            continue
        if not list(speaker_dir.glob("*.power.json")):
            missing.append(f"missing *.power.json in: {speaker_dir}")
    return missing


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch build token-level label tables from pr_mt, mt_v2/mt_v3, "
            "and pr_v2/pr_v3 POWER alignments."
        )
    )
    parser.add_argument(
        "--alignment-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root containing pr_mt/, mt_v2/, mt_v3/, pr_v2/, pr_v3/.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("all",) + tuple(TASK_TO_VARIANT),
        default=["all"],
        help="Tasks to build. Default: all.",
    )
    parser.add_argument(
        "--speaker",
        help="Run only one speaker, e.g. fn000049 or fn000049_txt.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also write one combined CSV per task.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    alignment_root = args.alignment_root.resolve()
    selected_tasks = list(TASK_TO_VARIANT if "all" in args.tasks else args.tasks)
    speakers = discover_speakers(alignment_root, selected_tasks, args.speaker)

    total_ok = 0
    total_failed = 0
    total_rows = 0

    for task in selected_tasks:
        variant = TASK_TO_VARIANT[task]
        output_root = alignment_root / task
        output_root.mkdir(parents=True, exist_ok=True)
        error_path = output_root / f"{task}_errors.csv"
        summary_path = output_root / f"{task}_summary.csv"
        combined_rows: list[dict[str, object]] = []

        with error_path.open("w", encoding="utf-8", newline="") as error_file, summary_path.open(
            "w", encoding="utf-8", newline=""
        ) as summary_file:
            error_writer = csv.DictWriter(error_file, fieldnames=ERROR_FIELDS)
            summary_writer = csv.DictWriter(
                summary_file,
                fieldnames=["task", "speaker", "status", "rows", "output_path"],
            )
            error_writer.writeheader()
            summary_writer.writeheader()

            for speaker in speakers:
                output_path = output_root / speaker / f"{speaker}_{task}_label_table.csv"
                missing = missing_inputs(alignment_root, speaker, variant)
                if missing:
                    total_failed += 1
                    reason = " | ".join(missing)
                    error_writer.writerow(
                        {
                            "task": task,
                            "speaker": speaker,
                            "stage": "discover",
                            "reason": reason,
                            "output_path": output_path,
                        }
                    )
                    summary_writer.writerow(
                        {
                            "task": task,
                            "speaker": speaker,
                            "status": "skipped",
                            "rows": 0,
                            "output_path": output_path,
                        }
                    )
                    continue

                try:
                    rows = rows_for_speaker_variant(speaker, variant, alignment_root)
                    write_rows(output_path, rows)
                except Exception as error:
                    total_failed += 1
                    error_writer.writerow(
                        {
                            "task": task,
                            "speaker": speaker,
                            "stage": "build",
                            "reason": repr(error),
                            "output_path": output_path,
                        }
                    )
                    summary_writer.writerow(
                        {
                            "task": task,
                            "speaker": speaker,
                            "status": "failed",
                            "rows": 0,
                            "output_path": output_path,
                        }
                    )
                    continue

                total_ok += 1
                total_rows += len(rows)
                combined_rows.extend(rows)
                summary_writer.writerow(
                    {
                        "task": task,
                        "speaker": speaker,
                        "status": "ok",
                        "rows": len(rows),
                        "output_path": output_path,
                    }
                )

        if args.combined:
            write_rows(output_root / f"{task}_all_speakers_label_table.csv", combined_rows)

    print(f"Alignment root: {alignment_root}")
    print(f"Speakers discovered: {len(speakers)}")
    print(f"Completed speaker/tasks: {total_ok}")
    print(f"Failed or skipped speaker/tasks: {total_failed}")
    print(f"Rows written: {total_rows}")
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
