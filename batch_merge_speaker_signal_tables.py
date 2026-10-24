from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
import traceback


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from merge_speaker_signal_tables import (
    SIGNAL_SUFFIXES,
    has_expected_files,
    merge_speaker_dir,
)


@dataclass(frozen=True)
class BatchRow:
    speaker_id: str
    status: str
    output_csv: str
    skipped_log_csv: str
    rows: int
    skipped_anchor_positions: int
    message: str


def missing_signal_files(speaker_dir: Path) -> list[str]:
    speaker_id = speaker_dir.name
    missing: list[str] = []
    for suffix in SIGNAL_SUFFIXES:
        path = speaker_dir / f"{speaker_id}_{suffix}.csv"
        if not path.exists():
            missing.append(path.name)
    return missing


def discover_speaker_dirs(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.iterdir() if path.is_dir())


def batch_merge(
    input_root: Path,
    mode: str = "rough",
    output_name: str | None = None,
    manifest_csv: Path | None = None,
    skip_incomplete: bool = True,
) -> list[BatchRow]:
    input_root = input_root.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {input_root}")

    manifest_csv = manifest_csv or input_root / "batch_merge_speaker_signal_tables.csv"
    rows: list[BatchRow] = []

    for speaker_dir in discover_speaker_dirs(input_root):
        speaker_id = speaker_dir.name
        if not has_expected_files(speaker_dir):
            missing = missing_signal_files(speaker_dir)
            if skip_incomplete:
                rows.append(
                    BatchRow(
                        speaker_id=speaker_id,
                        status="skipped_missing_files",
                        output_csv="",
                        skipped_log_csv="",
                        rows=0,
                        skipped_anchor_positions=0,
                        message=";".join(missing),
                    )
                )
                continue
            raise FileNotFoundError(
                f"{speaker_dir} is missing required signal files: {missing}"
            )

        try:
            result = merge_speaker_dir(
                speaker_dir,
                output_name=output_name,
                mode=mode,
            )
            rows.append(
                BatchRow(
                    speaker_id=speaker_id,
                    status="ok",
                    output_csv=str(result.output_path),
                    skipped_log_csv=str(result.log_path),
                    rows=result.written_rows,
                    skipped_anchor_positions=result.skipped_anchor_positions,
                    message="",
                )
            )
        except Exception as exc:
            rows.append(
                BatchRow(
                    speaker_id=speaker_id,
                    status="error",
                    output_csv="",
                    skipped_log_csv="",
                    rows=0,
                    skipped_anchor_positions=0,
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
            traceback.print_exc()

    write_manifest(manifest_csv, rows)
    return rows


def write_manifest(path: Path, rows: list[BatchRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "speaker_id",
        "status",
        "output_csv",
        "skipped_log_csv",
        "rows",
        "skipped_anchor_positions",
        "message",
    ]
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "speaker_id": row.speaker_id,
                    "status": row.status,
                    "output_csv": row.output_csv,
                    "skipped_log_csv": row.skipped_log_csv,
                    "rows": row.rows,
                    "skipped_anchor_positions": row.skipped_anchor_positions,
                    "message": row.message,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch merge speaker signal CSVs under a root folder. Each speaker folder "
            "must contain <speaker>_cross_attention.csv, <speaker>_ctc_alignment.csv, "
            "<speaker>_decoding_consistency.csv, and <speaker>_word_scores.csv. "
            "The merged CSV is written back into the speaker folder."
        )
    )
    parser.add_argument(
        "input_root",
        type=Path,
        help=(
            "Root folder containing speaker folders, e.g. "
            "/vol/tensusers/lgao/uncertainty_whisper/0_uncertainty_attribution_whisper/"
            "data/jasmin/v2_segment/"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("rough", "strict"),
        default="rough",
        help="Merge mode passed to merge_speaker_signal_tables.py. Default: rough.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Optional same output filename in every speaker folder. Default: "
            "<speaker_id>_merged_signals.csv"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Batch summary CSV path. Default: "
            "<input_root>/batch_merge_speaker_signal_tables.csv"
        ),
    )
    parser.add_argument(
        "--fail-on-incomplete",
        action="store_true",
        help="Raise an error if a speaker folder is missing any required signal CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = batch_merge(
        args.input_root,
        mode=args.mode,
        output_name=args.output_name,
        manifest_csv=args.manifest,
        skip_incomplete=not args.fail_on_incomplete,
    )
    ok = sum(row.status == "ok" for row in rows)
    skipped = sum(row.status == "skipped_missing_files" for row in rows)
    errors = sum(row.status == "error" for row in rows)
    manifest = args.manifest or args.input_root / "batch_merge_speaker_signal_tables.csv"
    print(f"speakers_ok={ok}")
    print(f"speakers_skipped_missing_files={skipped}")
    print(f"speakers_error={errors}")
    print(f"manifest={manifest}")


if __name__ == "__main__":
    main()
