from __future__ import annotations

import argparse
import sys
from pathlib import Path

from batch_mt_v_alignment import TASKS, build_tasks
from batch_power_alignment import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RAW_ROOT,
    DEFAULT_V2_ROOT,
    DEFAULT_V3_ROOT,
    resolve_lexicon,
    resolve_power_dir,
    run_task,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one difficult mt_v alignment with custom POWER multiword limits. "
            "This is intended for timeout cases such as fn000051 mt_v2."
        )
    )
    parser.add_argument("--speaker", required=True, help="Speaker id, e.g. fn000051.")
    parser.add_argument("--task", choices=TASKS, required=True, help="Run mt_v2 or mt_v3.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--v3-root", type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--power-dir", type=Path, help="Directory containing power.py.")
    parser.add_argument("--lexicon", type=Path, help="POWER JSON or word<TAB>phones lexicon.")
    parser.add_argument("--timeout", type=int, default=1500)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--max-many-to-one", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--max-one-to-many", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--max-power-region-words", type=int, default=7)
    parser.add_argument("--max-multiword-char-ratio", type=float, default=1.4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    speaker = args.speaker.removesuffix("_txt")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    power_dir = resolve_power_dir(args.power_dir)
    lexicon = resolve_lexicon(args.lexicon, output_root)

    tasks, missing = build_tasks(
        speaker,
        args.raw_root,
        args.v2_root,
        args.v3_root,
        output_root,
        {args.task},
    )
    if missing:
        for task_name, reason in missing:
            print(f"{task_name},{speaker},discover,{reason}", file=sys.stderr)
        return 1
    if len(tasks) != 1:
        print(f"Expected one task for {speaker} {args.task}, got {len(tasks)}", file=sys.stderr)
        return 1

    task = tasks[0]
    task.max_many_to_one = args.max_many_to_one
    task.max_one_to_many = args.max_one_to_many
    task.max_power_region_words = args.max_power_region_words
    task.max_multiword_char_ratio = args.max_multiword_char_ratio

    ok, elapsed, reason, err_tail, returncode = run_task(
        task,
        speaker,
        power_dir,
        lexicon,
        args.timeout,
        args.python_bin,
    )
    print(f"Task: {task.name}")
    print(f"Speaker: {speaker}")
    print(f"Output: {task.output_dir}")
    print(f"Elapsed: {elapsed:.3f}s")
    print(f"Status: {'ok' if ok else 'failed'}")
    if not ok:
        print(f"Reason: {reason}")
        print(f"Return code: {returncode}")
        if err_tail:
            print("stderr tail:")
            print(err_tail)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
