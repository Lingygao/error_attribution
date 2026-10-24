from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from batch_power_alignment import DEFAULT_OUTPUT_ROOT, DEFAULT_RAW_ROOT


TASKS = ("v2", "v3")


@dataclass(frozen=True)
class TokenTag:
    index: int
    word: str
    label: str
    slot_index: int
    token_index_in_slot: int
    ref_slot: str
    hyp_slot: str
    ref_tokens: tuple[str, ...]
    hyp_tokens: tuple[str, ...]


def split_words(text: str) -> list[str]:
    return text.split() if text else []


def load_power_json(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as fin:
        data: Any = json.load(fin)
    alignments = data.get("alignments")
    if not isinstance(alignments, list):
        raise ValueError(f"{path} has no list field: alignments")

    slots: list[dict[str, str]] = []
    for index, slot in enumerate(alignments):
        if not isinstance(slot, dict):
            raise ValueError(f"{path}: alignment slot {index} is not an object")
        slots.append(
            {
                "align": str(slot.get("align", "")),
                "ref": str(slot.get("ref", "")),
                "hyp": str(slot.get("hyp", "")),
            }
        )
    return slots


def expand_side(slots: list[dict[str, str]], side: str) -> list[TokenTag]:
    if side not in {"ref", "hyp"}:
        raise ValueError(f"side must be ref or hyp, got {side}")

    tags: list[TokenTag] = []
    token_index = 0
    for slot_index, slot in enumerate(slots):
        ref_tokens = tuple(split_words(slot["ref"]))
        hyp_tokens = tuple(split_words(slot["hyp"]))
        side_tokens = ref_tokens if side == "ref" else hyp_tokens
        for token_index_in_slot, word in enumerate(side_tokens):
            tags.append(
                TokenTag(
                    index=token_index,
                    word=word,
                    label=slot["align"],
                    slot_index=slot_index,
                    token_index_in_slot=token_index_in_slot,
                    ref_slot=slot["ref"],
                    hyp_slot=slot["hyp"],
                    ref_tokens=ref_tokens,
                    hyp_tokens=hyp_tokens,
                )
            )
            token_index += 1
    return tags


def prompt_label_for_hyp_token(tag: TokenTag) -> str:
    if not tag.ref_tokens:
        return ""
    if len(tag.ref_tokens) == len(tag.hyp_tokens):
        return tag.ref_tokens[tag.token_index_in_slot]
    if len(tag.ref_tokens) == 1:
        suffix = "*" if tag.token_index_in_slot > 0 else ""
        return f"{tag.ref_tokens[0]}{suffix}"
    return " ".join(tag.ref_tokens)


def slot_words_for_side(tag: TokenTag, side: str) -> str:
    words = tag.ref_tokens if side == "ref" else tag.hyp_tokens
    return " ".join(words)


def find_power_json(task_dir: Path, speaker: str, task_name: str) -> Path | None:
    preferred = task_dir / f"{speaker}_{task_name}.power.json"
    if preferred.exists():
        return preferred

    matches = sorted(task_dir.glob("*.power.json"))
    if len(matches) == 1:
        return matches[0]
    if matches:
        named = [path for path in matches if task_name in path.stem]
        if len(named) == 1:
            return named[0]
    return None


def require_alignment(
    output_root: Path,
    task_dir_name: str,
    speaker: str,
    file_task_name: str,
) -> Path:
    task_dir = output_root / task_dir_name / speaker
    if not task_dir.exists():
        raise FileNotFoundError(f"missing alignment directory: {task_dir}")
    path = find_power_json(task_dir, speaker, file_task_name)
    if path is None:
        raise FileNotFoundError(f"missing or ambiguous *.power.json in {task_dir}")
    return path


def join_tag_values(tags: list[TokenTag], values: list[str]) -> str:
    if not tags:
        return ""
    return " | ".join(values)


def rows_for_speaker_variant(
    speaker: str,
    variant: str,
    output_root: Path,
    pr_v_override: Path | None = None,
) -> list[dict[str, object]]:
    mt_v_task_dir = f"mt_{variant}"
    pr_v_task_dir = f"pr_{variant}"

    mt_v_path = require_alignment(output_root, mt_v_task_dir, speaker, mt_v_task_dir)
    pr_mt_path = require_alignment(output_root, "pr_mt", speaker, "pr_mt")
    pr_v_path = pr_v_override or require_alignment(output_root, pr_v_task_dir, speaker, pr_v_task_dir)

    mt_v_slots = load_power_json(mt_v_path)
    pr_mt_slots = load_power_json(pr_mt_path)
    pr_v_slots = load_power_json(pr_v_path)

    mt_v_hyp_tags = expand_side(mt_v_slots, "hyp")
    mt_v_ref_tags = expand_side(mt_v_slots, "ref")
    pr_mt_hyp_tags = expand_side(pr_mt_slots, "hyp")
    pr_v_hyp_tags = expand_side(pr_v_slots, "hyp")

    if len(mt_v_ref_tags) != len(pr_mt_hyp_tags):
        raise ValueError(
            "MT token count mismatch: "
            f"mt_{variant}.ref={len(mt_v_ref_tags)} vs pr_mt.hyp={len(pr_mt_hyp_tags)}"
        )
    if len(mt_v_hyp_tags) != len(pr_v_hyp_tags):
        raise ValueError(
            "ASR token count mismatch: "
            f"mt_{variant}.hyp={len(mt_v_hyp_tags)} vs pr_{variant}.hyp={len(pr_v_hyp_tags)}"
        )

    mt_to_pr_mt = {tag.index: pr_mt_hyp_tags[tag.index] for tag in mt_v_ref_tags}
    asr_to_pr_v = {tag.index: pr_v_hyp_tags[tag.index] for tag in mt_v_hyp_tags}

    rows: list[dict[str, object]] = []
    for asr_tag in mt_v_hyp_tags:
        mt_tags_for_slot = [
            tag for tag in mt_v_ref_tags if tag.slot_index == asr_tag.slot_index
        ]
        pr_mt_tags = [mt_to_pr_mt[tag.index] for tag in mt_tags_for_slot]
        pr_v_tag = asr_to_pr_v[asr_tag.index]

        rows.append(
            {
                "speaker": speaker,
                "variant": variant,
                "asr_index": asr_tag.index,
                "asr_word": asr_tag.word,
                "mt_v_label": asr_tag.label,
                "mt_words": join_tag_values(
                    mt_tags_for_slot, [tag.word for tag in mt_tags_for_slot]
                ),
                "pr_mt_labels": join_tag_values(
                    pr_mt_tags, [tag.label for tag in pr_mt_tags]
                ),
                "pr_mt_prompt_words": join_tag_values(
                    pr_mt_tags, [prompt_label_for_hyp_token(tag) for tag in pr_mt_tags]
                ),
                "pr_v_label": pr_v_tag.label,
                "pr_v_prompt_words": prompt_label_for_hyp_token(pr_v_tag),
                "source_mt_v_json": str(mt_v_path),
                "source_pr_mt_json": str(pr_mt_path),
                "source_pr_v_json": str(pr_v_path),
            }
        )
    return rows


def discover_speakers(output_root: Path, only_speaker: str | None) -> list[str]:
    if only_speaker:
        return [only_speaker.removesuffix("_txt")]

    speakers: set[str] = set()
    for task_name in ("mt_v2", "mt_v3"):
        task_root = output_root / task_name
        if not task_root.exists():
            continue
        for path in task_root.iterdir():
            if path.is_dir():
                speakers.add(path.name.removesuffix("_txt"))
    return sorted(speakers)


def build_demo(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    ]
    rows = [
        {
            "speaker": "demo",
            "variant": "v2",
            "asr_index": 0,
            "asr_word": "I",
            "mt_words": "I",
            "pr_mt_prompt_words": "I",
            "pr_v_prompt_words": "I",
            "mt_v_label": "C",
            "pr_mt_labels": "C",
            "pr_v_label": "C",
        },
        {
            "speaker": "demo",
            "variant": "v2",
            "asr_index": 1,
            "asr_word": "eat",
            "mt_words": "ate",
            "pr_mt_prompt_words": "ate",
            "pr_v_prompt_words": "ate",
            "mt_v_label": "S",
            "pr_mt_labels": "C",
            "pr_v_label": "S",
        },
        {
            "speaker": "demo",
            "variant": "v2",
            "asr_index": 2,
            "asr_word": "a",
            "mt_words": "an",
            "pr_mt_prompt_words": "red",
            "pr_v_prompt_words": "",
            "mt_v_label": "S",
            "pr_mt_labels": "S",
            "pr_v_label": "I",
        },
        {
            "speaker": "demo",
            "variant": "v2",
            "asr_index": 3,
            "asr_word": "red",
            "mt_words": "ed",
            "pr_mt_prompt_words": "apples",
            "pr_v_prompt_words": "red",
            "mt_v_label": "S",
            "pr_mt_labels": "S",
            "pr_v_label": "C",
        },
        {
            "speaker": "demo",
            "variant": "v2",
            "asr_index": 4,
            "asr_word": "red",
            "mt_words": "",
            "pr_mt_prompt_words": "",
            "pr_v_prompt_words": "",
            "mt_v_label": "I",
            "pr_mt_labels": "",
            "pr_v_label": "I",
        },
        {
            "speaker": "demo",
            "variant": "v2",
            "asr_index": 5,
            "asr_word": "apple",
            "mt_words": "apple",
            "pr_mt_prompt_words": "apple",
            "pr_v_prompt_words": "apples",
            "mt_v_label": "C",
            "pr_mt_labels": "C",
            "pr_v_label": "S",
        },
    ]
    with output_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a token-level label table using mt_v alignment as the row baseline. "
            "Rows are ASR output words from mt_v hyp."
        )
    )
    parser.add_argument("--alignment-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--speaker", help="Run one speaker, e.g. fn000049 or fn000049_txt.")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("all",) + TASKS,
        default=["all"],
        help="Which mt_v/pr_v variants to combine. Default: all.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "alignment_label_table.csv",
    )
    parser.add_argument(
        "--errors",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "alignment_label_table_errors.csv",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Write the small PR/MT/AO example table and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.demo:
        build_demo(args.output)
        print(f"Demo table: {args.output}")
        return 0

    selected_variants = list(TASKS if "all" in args.variants else args.variants)
    alignment_root = args.alignment_root.resolve()
    speakers = discover_speakers(alignment_root, args.speaker)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.errors.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
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
    error_fields = ["speaker", "variant", "reason"]

    ok_tasks = 0
    failed_tasks = 0
    row_count = 0
    with args.output.open("w", encoding="utf-8", newline="") as out_file, args.errors.open(
        "w", encoding="utf-8", newline=""
    ) as err_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        error_writer = csv.DictWriter(err_file, fieldnames=error_fields)
        writer.writeheader()
        error_writer.writeheader()

        for speaker in speakers:
            for variant in selected_variants:
                try:
                    rows = rows_for_speaker_variant(speaker, variant, alignment_root)
                except Exception as error:
                    failed_tasks += 1
                    error_writer.writerow(
                        {
                            "speaker": speaker,
                            "variant": variant,
                            "reason": repr(error),
                        }
                    )
                    continue

                writer.writerows(rows)
                row_count += len(rows)
                ok_tasks += 1

    print(f"Speakers: {len(speakers)}")
    print(f"Completed speaker/variant tasks: {ok_tasks}")
    print(f"Failed speaker/variant tasks: {failed_tasks}")
    print(f"Rows: {row_count}")
    print(f"Output: {args.output}")
    print(f"Errors: {args.errors}")
    return 0 if failed_tasks == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
