from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from text_normalization import text_norm_all


DEFAULT_RAW_ROOT = Path(
    "/vol/tensusers/lgao/uncertainty_whisper/"
    "0_uncertainty_attribution_whisper/data/jasmin/raw"
)
DEFAULT_V2_ROOT = Path(
    "/vol/tensusers/lgao/uncertainty_whisper/"
    "0_uncertainty_attribution_whisper/data/jasmin/v2_segment"
)
DEFAULT_V3_ROOT = Path(
    "/vol/tensusers/lgao/uncertainty_whisper/"
    "0_uncertainty_attribution_whisper/data/jasmin/v3_segment"
)
DEFAULT_OUTPUT_ROOT = Path("output/alignment")
TASKS = ("pr_mt", "pr_v2", "pr_v3")


@dataclass
class AlignmentTask:
    name: str
    ref_path: Path
    hyp_path: Path
    output_dir: Path
    max_many_to_one: int
    max_one_to_many: int
    max_multiword_char_ratio: float | None
    max_power_region_words: int | None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_norm(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text_norm_all(text) + "\n", encoding="utf-8")


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists() and path.is_file():
            return path
    return None


def find_raw_transcript(speaker_dir: Path, speaker: str, suffix: str) -> Path | None:
    candidates = [
        speaker_dir / f"{speaker}_{suffix}.txt",
        speaker_dir / f"{speaker_dir.name}_{suffix}.txt",
        speaker_dir / f"{suffix}.txt",
    ]
    exact = first_existing(candidates)
    if exact is not None:
        return exact

    matches = sorted(speaker_dir.glob(f"*_{suffix}.txt"))
    if matches:
        return matches[0]
    return None


def find_output_transcript(output_root: Path, speaker: str) -> Path | None:
    candidates = [
        output_root / speaker / f"{speaker}_full.txt",
        output_root / speaker / f"{speaker}_txt_full.txt",
        output_root / speaker / "full.txt",
        output_root / f"{speaker}_full.txt",
        output_root / speaker / f"{speaker}.txt",
        output_root / speaker / "output.txt",
        output_root / f"{speaker}.txt",
    ]
    exact = first_existing(candidates)
    if exact is not None:
        return exact

    speaker_dir = output_root / speaker
    if speaker_dir.exists():
        matches = sorted(
            path
            for path in speaker_dir.glob("*.txt")
            if not path.name.endswith("_norm.txt")
        )
        if matches:
            return matches[0]
    return None


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
        if find_raw_transcript(path, speaker, "pr") is not None:
            speakers.append(speaker)
    return sorted(speakers)


def resolve_power_dir(power_dir: Path | None) -> Path:
    candidates = []
    if power_dir is not None:
        candidates.append(power_dir)
    candidates.extend(
        [
            REPO_DIR / "scripts" / "power-asr",
            REPO_DIR / "scripts" / "power-asr-master",
            REPO_DIR / "power-asr-master",
            Path.cwd() / "scripts" / "power-asr",
            Path.cwd() / "scripts" / "power-asr-master",
            Path.cwd() / "power-asr-master",
        ]
    )

    for candidate in candidates:
        if (candidate / "power.py").exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not find power.py. Pass --power-dir explicitly.")


def parse_tab_lexicon(path: Path) -> dict[str, str]:
    lexicon: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split("\t", 1)
            if len(fields) != 2:
                raise ValueError(f"{path}: line {line_number} must contain word<TAB>phones")

            word, phones = fields
            word = word.strip().lower()
            phones = " ".join(phones.strip().split())
            if not word:
                raise ValueError(f"{path}: line {line_number} has an empty word")
            if not phones:
                raise ValueError(f"{path}: line {line_number} has empty phones for {word}")
            lexicon[word] = phones
    return lexicon


def validate_json_lexicon(path: Path) -> None:
    if path.stat().st_size == 0:
        raise ValueError(f"Lexicon JSON is empty: {path}")
    with path.open("r", encoding="utf-8") as fin:
        data = json.load(fin)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Lexicon JSON must be a non-empty object: {path}")


def convert_tab_lexicon(path: Path, output_root: Path) -> Path:
    lexicon = parse_tab_lexicon(path)
    if not lexicon:
        raise ValueError(f"Tab lexicon has no entries: {path}")

    output_root.mkdir(parents=True, exist_ok=True)
    converted = output_root / f"{path.stem}.power.json"
    with converted.open("w", encoding="utf-8") as fout:
        json.dump(lexicon, fout, ensure_ascii=False, indent=2, sort_keys=True)
        fout.write("\n")
    return converted.resolve()


def prepare_lexicon(path: Path, output_root: Path) -> Path:
    suffix = path.suffix.lower()
    if suffix == ".json":
        validate_json_lexicon(path)
        return path.resolve()
    if suffix in {".dict", ".txt", ".lex"}:
        return convert_tab_lexicon(path, output_root)
    raise ValueError(
        f"Unsupported lexicon format: {path}. Use POWER JSON or word<TAB>phones .dict/.txt."
    )


def resolve_lexicon(path: Path | None, output_root: Path) -> Path:
    candidates = []
    if path is not None:
        candidates.append(path)
    candidates.extend(
        [
            REPO_DIR / "power_runs" / "merged_lexicon.json",
            REPO_DIR / "merged_lexicon.json",
            REPO_DIR / "merged_lexicon.dict",
            Path.cwd() / "power_runs" / "merged_lexicon.json",
            Path.cwd() / "merged_lexicon.json",
            Path.cwd() / "merged_lexicon.dict",
        ]
    )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return prepare_lexicon(candidate, output_root)
    raise FileNotFoundError("Could not find merged lexicon. Pass --lexicon.")


def stderr_tail(text: str, limit: int = 1200) -> str:
    text = text.strip()
    return text[-limit:] if len(text) > limit else text


def write_error(writer: csv.DictWriter, task: str, speaker: str, stage: str, reason: str, **kwargs: object) -> None:
    row = {
        "task": task,
        "speaker": speaker,
        "stage": stage,
        "reason": reason,
        "ref_path": kwargs.get("ref_path", ""),
        "hyp_path": kwargs.get("hyp_path", ""),
        "output_dir": kwargs.get("output_dir", ""),
        "returncode": kwargs.get("returncode", ""),
        "elapsed_sec": kwargs.get("elapsed_sec", ""),
        "stderr_tail": kwargs.get("stderr_tail", ""),
    }
    writer.writerow(row)


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
    pr_path = find_raw_transcript(speaker_dir, speaker, "pr")
    mt_path = find_raw_transcript(speaker_dir, speaker, "mt")
    tasks: list[AlignmentTask] = []
    missing: list[tuple[str, str]] = []

    if pr_path is None:
        return [], [(task, "missing pr transcript") for task in selected_tasks]

    if "pr_mt" in selected_tasks:
        if mt_path is None:
            missing.append(("pr_mt", "missing mt transcript"))
        else:
            tasks.append(
                AlignmentTask(
                    name="pr_mt",
                    ref_path=pr_path,
                    hyp_path=mt_path,
                    output_dir=output_root / "pr_mt" / speaker,
                    max_many_to_one=1,
                    max_one_to_many=1,
                    max_multiword_char_ratio=None,
                    max_power_region_words=25,
                )
            )

    if "pr_v2" in selected_tasks:
        v2_path = find_output_transcript(v2_root, speaker)
        if v2_path is None:
            missing.append(("pr_v2", "missing v2 transcript"))
        else:
            tasks.append(
                AlignmentTask(
                    name="pr_v2",
                    ref_path=pr_path,
                    hyp_path=v2_path,
                    output_dir=output_root / "pr_v2" / speaker,
                    max_many_to_one=3,
                    max_one_to_many=3,
                    max_multiword_char_ratio=1.4,
                    max_power_region_words=25,
                )
            )

    if "pr_v3" in selected_tasks:
        v3_path = find_output_transcript(v3_root, speaker)
        if v3_path is None:
            missing.append(("pr_v3", "missing v3 transcript"))
        else:
            tasks.append(
                AlignmentTask(
                    name="pr_v3",
                    ref_path=pr_path,
                    hyp_path=v3_path,
                    output_dir=output_root / "pr_v3" / speaker,
                    max_many_to_one=3,
                    max_one_to_many=3,
                    max_multiword_char_ratio=1.4,
                    max_power_region_words=25,
                )
            )

    return tasks, missing


def run_task(
    task: AlignmentTask,
    speaker: str,
    power_dir: Path,
    lexicon: Path,
    timeout: int,
    python_bin: str,
) -> tuple[bool, float, str, str, int | None]:
    task.output_dir.mkdir(parents=True, exist_ok=True)
    norm_ref = task.output_dir / "ref_norm.txt"
    norm_hyp = task.output_dir / "hyp_norm.txt"
    stdout_path = task.output_dir / "power.stdout.txt"
    stderr_path = task.output_dir / "power.stderr.txt"
    command_path = task.output_dir / "power.command.txt"
    config_path = task.output_dir / "task_config.json"
    output_prefix = task.output_dir / f"{speaker}_{task.name}"

    write_norm(norm_ref, read_text(task.ref_path))
    write_norm(norm_hyp, read_text(task.hyp_path))

    cmd = [
        python_bin,
        "power.py",
        "--ref",
        str(norm_ref),
        "--hyp",
        str(norm_hyp),
        "--lexicon",
        str(lexicon),
        "--lexicon-fallback",
        "strict",
        "--alignment-mode",
        "auto",
        "--max-many-to-one",
        str(task.max_many_to_one),
        "--max-one-to-many",
        str(task.max_one_to_many),
        "-f",
        "snt",
        "json",
        "-o",
        str(output_prefix),
    ]
    if task.max_multiword_char_ratio is not None:
        cmd.extend(["--max-multiword-char-ratio", str(task.max_multiword_char_ratio)])
    if task.max_power_region_words is not None:
        cmd.extend(["--max-power-region-words", str(task.max_power_region_words)])

    command_path.write_text(" ".join(cmd) + "\n", encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "task": task.name,
                "speaker": speaker,
                "ref_path": str(task.ref_path),
                "hyp_path": str(task.hyp_path),
                "normalized_ref": str(norm_ref),
                "normalized_hyp": str(norm_hyp),
                "output_prefix": str(output_prefix),
                "alignment_mode": "auto",
                "max_many_to_one": task.max_many_to_one,
                "max_one_to_many": task.max_one_to_many,
                "max_multiword_char_ratio": task.max_multiword_char_ratio,
                "max_power_region_words": task.max_power_region_words,
                "lexicon": str(lexicon),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    start = time.monotonic()
    try:
        completed = subprocess.run(
            cmd,
            cwd=power_dir,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        elapsed = time.monotonic() - start
        stdout_path.write_text(error.stdout or "", encoding="utf-8")
        stderr_path.write_text(error.stderr or "", encoding="utf-8")
        return False, elapsed, "timeout", stderr_tail(error.stderr or ""), None

    elapsed = time.monotonic() - start
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        return False, elapsed, "power_failed", stderr_tail(completed.stderr), completed.returncode

    return True, elapsed, "", "", completed.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch Dutch POWER alignments for JASMIN pr/mt and pr/v2/v3 outputs."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--v3-root", type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--power-dir", type=Path, help="Directory containing power.py.")
    parser.add_argument("--lexicon", type=Path, help="POWER JSON lexicon, e.g. merged_lexicon.json.")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--speaker", help="Run only one speaker.")
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

    error_path = args.output_root / "alignment_errors.csv"
    summary_path = args.output_root / "alignment_summary.csv"
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
