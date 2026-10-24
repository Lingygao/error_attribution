from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from batch_power_alignment import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RAW_ROOT,
    DEFAULT_V2_ROOT,
    DEFAULT_V3_ROOT,
    AlignmentTask,
    find_output_transcript,
    find_raw_transcript,
    resolve_lexicon,
    resolve_power_dir,
    run_task,
    write_error,
)


TASKS = ("mt_v2", "mt_v3")


def discover_speakers(raw_root: Path, only_speaker: str | None) -> list[str]:
    if only_speaker:
        return [only_speaker.removesuffix("_txt")]

    speakers = []
    for path in raw_root.iterdir():
        if not path.is_dir():
            continue
        if not path.name.endswith("_txt"):
            continue
        speaker = path.name[:-4]
        if find_raw_transcript(path, speaker, "mt") is not None:
            speakers.append(speaker)
    return sorted(speakers)


def build_tasks(
    speaker: str,
    raw_root: Path,
    v2_root: Path,
    v3_root: Path,
    output_root: Path,
    selected_tasks: set[str],
) -> tuple[list[AlignmentTask], list[tuple[str, str]]]:
    speaker_dir = raw_root / f"{speaker}_txt"
    if not speaker_dir.exists():
        speaker_dir = raw_root / speaker

    mt_path = find_raw_transcript(speaker_dir, speaker, "mt")
    tasks: list[AlignmentTask] = []
    missing: list[tuple[str, str]] = []

    if mt_path is None:
        return [], [(task, "missing mt transcript") for task in selected_tasks]

    if "mt_v2" in selected_tasks:
        v2_path = find_output_transcript(v2_root, speaker)
        if v2_path is None:
            missing.append(("mt_v2", "missing v2 transcript"))
        else:
            tasks.append(
                AlignmentTask(
                    name="mt_v2",
                    ref_path=mt_path,
                    hyp_path=v2_path,
                    output_dir=output_root / "mt_v2" / speaker,
                    max_many_to_one=3,
                    max_one_to_many=3,
                    max_multiword_char_ratio=1.4,
                    max_power_region_words=25,
                )
            )

    if "mt_v3" in selected_tasks:
        v3_path = find_output_transcript(v3_root, speaker)
        if v3_path is None:
            missing.append(("mt_v3", "missing v3 transcript"))
        else:
            tasks.append(
                AlignmentTask(
                    name="mt_v3",
                    ref_path=mt_path,
                    hyp_path=v3_path,
                    output_dir=output_root / "mt_v3" / speaker,
                    max_many_to_one=3,
                    max_one_to_many=3,
                    max_multiword_char_ratio=1.4,
                    max_power_region_words=25,
                )
            )

    return tasks, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch Dutch POWER alignments for JASMIN mt/v2 and mt/v3 outputs."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--v3-root", type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--power-dir", type=Path, help="Directory containing power.py.")
    parser.add_argument("--lexicon", type=Path, help="POWER JSON or word<TAB>phones lexicon.")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--speaker", help="Run only one speaker, e.g. fn000049 or fn000049_txt.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("all",) + TASKS,
        default=["all"],
        help="Tasks to run. Default: all.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_tasks = set(TASKS if "all" in args.tasks else args.tasks)
    power_dir = resolve_power_dir(args.power_dir)
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    lexicon = resolve_lexicon(args.lexicon, args.output_root)

    error_path = args.output_root / "alignment_errors_mt.csv"
    summary_path = args.output_root / "alignment_summary_mt.csv"
    error_fields = [
        "task",
        "speaker",
        "stage",
        "reason",
        "ref_path",
        "hyp_path",
        "output_dir",
        "returncode",
        "elapsed_sec",
        "stderr_tail",
    ]
    summary_fields = ["task", "speaker", "status", "elapsed_sec", "output_dir"]

    speakers = discover_speakers(args.raw_root, args.speaker)
    with error_path.open("w", encoding="utf-8", newline="") as error_file, summary_path.open(
        "w", encoding="utf-8", newline=""
    ) as summary_file:
        error_writer = csv.DictWriter(error_file, fieldnames=error_fields)
        summary_writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
        error_writer.writeheader()
        summary_writer.writeheader()

        for speaker in speakers:
            tasks, missing = build_tasks(
                speaker,
                args.raw_root,
                args.v2_root,
                args.v3_root,
                args.output_root,
                selected_tasks,
            )
            for task_name, reason in missing:
                write_error(error_writer, task_name, speaker, "discover", reason)
                summary_writer.writerow(
                    {
                        "task": task_name,
                        "speaker": speaker,
                        "status": "skipped",
                        "elapsed_sec": "",
                        "output_dir": "",
                    }
                )

            for task in tasks:
                try:
                    ok, elapsed, reason, err_tail, returncode = run_task(
                        task,
                        speaker,
                        power_dir,
                        lexicon,
                        args.timeout,
                        args.python_bin,
                    )
                except Exception as error:
                    ok = False
                    elapsed = 0.0
                    reason = "exception"
                    err_tail = repr(error)
                    returncode = None

                summary_writer.writerow(
                    {
                        "task": task.name,
                        "speaker": speaker,
                        "status": "ok" if ok else "failed",
                        "elapsed_sec": f"{elapsed:.3f}",
                        "output_dir": str(task.output_dir),
                    }
                )
                if not ok:
                    write_error(
                        error_writer,
                        task.name,
                        speaker,
                        "run",
                        reason,
                        ref_path=task.ref_path,
                        hyp_path=task.hyp_path,
                        output_dir=task.output_dir,
                        returncode="" if returncode is None else returncode,
                        elapsed_sec=f"{elapsed:.3f}",
                        stderr_tail=err_tail,
                    )

    print(f"Speakers: {len(speakers)}")
    print(f"Power dir: {power_dir}")
    print(f"Lexicon: {lexicon}")
    print(f"Summary: {summary_path.resolve()}")
    print(f"Errors: {error_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
