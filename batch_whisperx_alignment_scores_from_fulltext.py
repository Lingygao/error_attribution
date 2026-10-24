"""Batch WhisperX CTC alignment score tables from *_full.txt files.

Input layout:

    raw/speaker/speaker.wav
    v2_segment/speaker/speaker_full.txt
    v3_segment/speaker/speaker_full.txt

Default output next to each full text file:

    v2_segment/speaker/speaker_ctc_alignment.csv
    v3_segment/speaker/speaker_ctc_alignment.csv

Output columns:

    segment_id, word_idx, asr_word, A_ctc, A_std, audio_entropy

The alignment logic mirrors whisperx_alignment_scores.ipynb:
load full text as one transcript segment, run whisperx.align with
return_char_alignments=True, then compute frame-level entropy from the same CTC
alignment model and average it inside each aligned word time span.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")
MODEL_TO_SEGMENT_DIR = {
    "large-v2": "v2_segment",
    "large-v3": "v3_segment",
}
KEEP_COLUMNS = ["segment_id", "word_idx", "asr_word", "A_ctc", "A_std", "audio_entropy"]
DEFAULT_ALIGN_MODEL = "GroNLP/wav2vec2-dutch-large-ft-cgn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate WhisperX CTC alignment score tables for speaker wavs and *_full.txt files."
    )
    parser.add_argument(
        "--raw-root",
        required=True,
        type=Path,
        help="Raw audio root, e.g. .../data/jasmin/raw.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_TO_SEGMENT_DIR),
        default=["large-v2", "large-v3"],
        help="Transcript roots to process. Default: large-v2 large-v3.",
    )
    parser.add_argument("--segment-root-large-v2", type=Path, default=None)
    parser.add_argument("--segment-root-large-v3", type=Path, default=None)
    parser.add_argument("--text-suffix", default="_full.txt")
    parser.add_argument("--output-suffix", default="_ctc_alignment.csv")
    parser.add_argument("--language", default="nl")
    parser.add_argument("--align-model", default=DEFAULT_ALIGN_MODEL)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="WhisperX alignment model cache dir, e.g. /vol/.../hf_cache.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for WhisperX alignment model, e.g. cuda, cuda:0, cuda:1, cpu.",
    )
    parser.add_argument("--audio-ext", nargs="+", default=list(AUDIO_EXTENSIONS))
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", action="store_false", dest="overwrite")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=None,
        help="Optional CSV log path. Default: <raw-root-parent>/whisperx_alignment_scores_log.csv.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def iter_audio_files(raw_root: Path, extensions: Iterable[str]) -> list[Path]:
    normalized_exts = tuple(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions)
    audio_files = []
    for path in raw_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in normalized_exts:
            continue
        relative_parts = path.relative_to(raw_root).parts
        if any(part.endswith("_txt") for part in relative_parts[:-1]):
            continue
        audio_files.append(path)
    return sorted(audio_files)


def segment_root_for_model(args: argparse.Namespace, model_name: str) -> Path:
    if model_name == "large-v2" and args.segment_root_large_v2 is not None:
        return args.segment_root_large_v2.expanduser().resolve()
    if model_name == "large-v3" and args.segment_root_large_v3 is not None:
        return args.segment_root_large_v3.expanduser().resolve()
    return (args.raw_root.expanduser().resolve().parent / MODEL_TO_SEGMENT_DIR[model_name]).resolve()


def text_path_for_audio(audio_path: Path, raw_root: Path, segment_root: Path, suffix: str) -> Path:
    relative = audio_path.relative_to(raw_root)
    return segment_root / relative.parent / f"{audio_path.stem}{suffix}"


def output_path_for_audio(text_path: Path, audio_path: Path, suffix: str) -> Path:
    return text_path.with_name(f"{audio_path.stem}{suffix}")


def audio_duration(audio_path: Path, whisperx_module: Any, sample_rate: int) -> float:
    audio = whisperx_module.load_audio(str(audio_path))
    return len(audio) / sample_rate


def load_asr_output(path: Path) -> dict[str, Any] | list[dict[str, Any]] | pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8").strip()
        return {"segments": [{"start": 0.0, "text": text}]}
    raise ValueError(f"Unsupported ASR output format: {path.suffix}")


def dataframe_to_segments(
    df: pd.DataFrame,
    *,
    audio_path: Path,
    whisperx_module: Any,
    sample_rate: int,
) -> list[dict[str, Any]]:
    df = df.copy()
    text_col = next((column for column in ["text", "asr", "word"] if column in df.columns), None)
    if text_col is None:
        raise ValueError("CSV must contain one of: text, asr, word")

    duration = audio_duration(audio_path, whisperx_module, sample_rate)
    if "text" in df.columns:
        start = df["start"] if "start" in df.columns else 0.0
        end = df["end"].clip(upper=duration) if "end" in df.columns else duration
        return pd.DataFrame({"start": start, "end": end, "text": df["text"]}).to_dict("records")

    text = " ".join(df[text_col].dropna().astype(str).str.strip())
    start = float(df["start"].min()) if "start" in df.columns else 0.0
    end = float(df["end"].max()) if "end" in df.columns else duration
    return [{"start": start, "end": min(end, duration), "text": text}]


def asr_to_segments(
    asr_output: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame,
    *,
    audio_path: Path,
    whisperx_module: Any,
    sample_rate: int,
) -> list[dict[str, Any]]:
    duration = audio_duration(audio_path, whisperx_module, sample_rate)

    if isinstance(asr_output, pd.DataFrame):
        segments = dataframe_to_segments(
            asr_output,
            audio_path=audio_path,
            whisperx_module=whisperx_module,
            sample_rate=sample_rate,
        )
    elif isinstance(asr_output, dict):
        segments = asr_output.get("segments") if "segments" in asr_output else [asr_output]
    elif isinstance(asr_output, list):
        segments = asr_output
    else:
        raise TypeError(f"Unsupported ASR output type: {type(asr_output)}")

    clean = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text and seg.get("words"):
            text = " ".join(str(word.get("word", "")).strip() for word in seg["words"] if str(word.get("word", "")).strip())
        if not text:
            continue

        start = max(0.0, float(seg.get("start", 0.0)))
        end = float(seg.get("end", duration))
        if end > duration:
            end = duration
        if end <= start:
            continue

        clean.append({"start": start, "end": end, "text": text})

    if not clean:
        raise ValueError("No usable ASR segments found.")
    return clean


def compute_ctc_frame_entropy(
    audio_path: Path,
    align_model: Any,
    device: str,
    *,
    whisperx_module: Any,
    torch_module: Any,
    sample_rate: int,
) -> tuple[np.ndarray, float]:
    audio = whisperx_module.load_audio(str(audio_path))
    waveform = torch_module.from_numpy(audio).unsqueeze(0).to(device)

    logits = align_model(waveform).logits[0]
    probs = torch_module.softmax(logits.float(), dim=-1)
    entropy = (-(probs * probs.clamp(min=1e-8).log()).sum(dim=-1)).cpu().numpy()

    duration = len(audio) / sample_rate
    return entropy, duration


def time_to_frame(t: float, duration: float, n_frames: int) -> int:
    return int(np.clip(t / duration * n_frames, 0, n_frames - 1))


def word_char_scores(word: dict[str, Any], chars: list[dict[str, Any]]) -> list[float]:
    if "start" not in word or "end" not in word:
        return []
    return [
        char["score"]
        for char in chars
        if "score" in char
        and "start" in char
        and "end" in char
        and char["start"] >= word["start"]
        and char["end"] <= word["end"]
    ]


def alignment_to_word_rows(
    aligned_result: dict[str, Any],
    frame_entropy: np.ndarray,
    audio_duration_value: float,
) -> list[dict[str, Any]]:
    rows = []
    for segment_id, seg in enumerate(aligned_result.get("segments", [])):
        for word_idx, word in enumerate(seg.get("words", [])):
            score = word.get("score")
            chars = seg.get("chars") or []
            char_scores = word_char_scores(word, chars)

            if len(char_scores) > 1:
                a_std = float(np.std(char_scores))
            elif len(char_scores) == 1:
                a_std = 0.0
            else:
                a_std = np.nan

            audio_entropy = np.nan
            if "start" in word and "end" in word:
                fs = time_to_frame(word["start"], audio_duration_value, len(frame_entropy))
                fe = time_to_frame(word["end"], audio_duration_value, len(frame_entropy))
                if fe > fs:
                    audio_entropy = float(np.mean(frame_entropy[fs:fe]))

            rows.append(
                {
                    "segment_id": segment_id,
                    "word_idx": word_idx,
                    "asr_word": word.get("word"),
                    "A_ctc": score,
                    "A_std": a_std,
                    "audio_entropy": audio_entropy,
                }
            )
    return rows


def write_score_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=KEEP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_log_header(log_path: Path) -> None:
    if log_path.exists():
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "audio", "text", "output_csv", "rows", "status", "duration_sec", "error"],
        )
        writer.writeheader()


def append_log(log_path: Path, row: dict[str, Any]) -> None:
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "audio", "text", "output_csv", "rows", "status", "duration_sec", "error"],
        )
        writer.writerow(row)


def process_one(
    *,
    audio_path: Path,
    text_path: Path,
    output_csv: Path,
    align_model: Any,
    align_metadata: dict[str, Any],
    device: str,
    whisperx_module: Any,
    torch_module: Any,
    sample_rate: int,
) -> int:
    asr_output = load_asr_output(text_path)
    segments = asr_to_segments(
        asr_output,
        audio_path=audio_path,
        whisperx_module=whisperx_module,
        sample_rate=sample_rate,
    )
    aligned = whisperx_module.align(
        transcript=segments,
        model=align_model,
        align_model_metadata=align_metadata,
        audio=str(audio_path),
        device=device,
        return_char_alignments=True,
    )
    frame_entropy, duration = compute_ctc_frame_entropy(
        audio_path,
        align_model,
        device,
        whisperx_module=whisperx_module,
        torch_module=torch_module,
        sample_rate=sample_rate,
    )
    rows = alignment_to_word_rows(aligned, frame_entropy, duration)
    write_score_csv(output_csv, rows)
    return len(rows)


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    if not raw_root.exists():
        print(f"[ERROR] raw root does not exist: {raw_root}", file=sys.stderr)
        return 2

    log_path = (args.log_csv or raw_root.parent / "whisperx_alignment_scores_log.csv").expanduser().resolve()
    write_log_header(log_path)

    audio_files = iter_audio_files(raw_root, args.audio_ext)
    if args.limit is not None:
        audio_files = audio_files[: args.limit]

    import torch
    import whisperx
    from whisperx.audio import SAMPLE_RATE

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = None if args.model_dir is None else str(args.model_dir.expanduser())

    print(f"[INFO] raw_root={raw_root}")
    print(f"[INFO] found_audio={len(audio_files)}")
    print(f"[INFO] models={' '.join(args.models)}")
    print(f"[INFO] align_model={args.align_model}")
    print(f"[INFO] device={device}")
    print(f"[INFO] log={log_path}")
    print("[INFO] loading WhisperX alignment model")

    align_model, align_metadata = whisperx.load_align_model(
        language_code=args.language,
        device=device,
        model_name=args.align_model,
        model_dir=model_dir,
    )

    for model_name in args.models:
        segment_root = segment_root_for_model(args, model_name)
        print(f"[INFO] text_root for {model_name}: {segment_root}")

        for index, audio_path in enumerate(audio_files, start=1):
            text_path = text_path_for_audio(audio_path, raw_root, segment_root, args.text_suffix)
            output_csv = output_path_for_audio(text_path, audio_path, args.output_suffix)
            started = time.time()

            if not text_path.exists():
                print(f"[{model_name} {index}/{len(audio_files)}] missing text: {text_path}")
                append_log(
                    log_path,
                    {
                        "model": model_name,
                        "audio": str(audio_path),
                        "text": str(text_path),
                        "output_csv": str(output_csv),
                        "rows": 0,
                        "status": "missing_text",
                        "duration_sec": "0.000",
                        "error": "",
                    },
                )
                continue

            if output_csv.exists() and not args.overwrite:
                print(f"[{model_name} {index}/{len(audio_files)}] skip existing: {output_csv}")
                append_log(
                    log_path,
                    {
                        "model": model_name,
                        "audio": str(audio_path),
                        "text": str(text_path),
                        "output_csv": str(output_csv),
                        "rows": "",
                        "status": "skipped",
                        "duration_sec": "0.000",
                        "error": "",
                    },
                )
                continue

            print(f"[{model_name} {index}/{len(audio_files)}] aligning: {audio_path}")
            try:
                with torch.inference_mode():
                    n_rows = process_one(
                        audio_path=audio_path,
                        text_path=text_path,
                        output_csv=output_csv,
                        align_model=align_model,
                        align_metadata=align_metadata,
                        device=device,
                        whisperx_module=whisperx,
                        torch_module=torch,
                        sample_rate=SAMPLE_RATE,
                    )
                duration = time.time() - started
                append_log(
                    log_path,
                    {
                        "model": model_name,
                        "audio": str(audio_path),
                        "text": str(text_path),
                        "output_csv": str(output_csv),
                        "rows": n_rows,
                        "status": "ok",
                        "duration_sec": f"{duration:.3f}",
                        "error": "",
                    },
                )
                if args.verbose:
                    print(f"  rows={n_rows}")
            except Exception as exc:
                duration = time.time() - started
                print(f"[ERROR] {model_name} failed for {audio_path}: {exc}", file=sys.stderr)
                append_log(
                    log_path,
                    {
                        "model": model_name,
                        "audio": str(audio_path),
                        "text": str(text_path),
                        "output_csv": str(output_csv),
                        "rows": "",
                        "status": "error",
                        "duration_sec": f"{duration:.3f}",
                        "error": repr(exc),
                    },
                )

    print(f"[INFO] done. log={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
