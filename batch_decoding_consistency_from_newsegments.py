"""Batch decoding consistency tables from generated *_newsegment.txt files.

Input layout:

    raw/speaker/speaker.wav
    v2_segment/speaker/speaker_newsegment.txt
    v3_segment/speaker/speaker_newsegment.txt

Default output next to each newsegment file:

    v2_segment/speaker/speaker_decoding_consistency.csv
    v3_segment/speaker/speaker_decoding_consistency.csv

Output columns:

    segment_id, word_idx, asr_word, decoding_consistency, aligned_alternative_words

The main words are read from *_newsegment.txt instead of running the main beam
decode again. Alternative decodes follow whisper_decoding_consistency_uncertainty.ipynb.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from transformers.generation.utils import GenerationMixin

import transcribe_vad_segments_beam as segtx


AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")
MODEL_TO_SEGMENT_DIR = {
    "large-v2": "v2_segment",
    "large-v3": "v3_segment",
}
KEEP_COLUMNS = [
    "segment_id",
    "word_idx",
    "asr_word",
    "decoding_consistency",
    "aligned_alternative_words",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate decoding consistency tables for speaker wavs and *_newsegment.txt files."
    )
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_TO_SEGMENT_DIR),
        default=["large-v2", "large-v3"],
    )
    parser.add_argument("--segment-root-large-v2", type=Path, default=None)
    parser.add_argument("--segment-root-large-v3", type=Path, default=None)
    parser.add_argument("--segment-suffix", default="_newsegment.txt")
    parser.add_argument("--output-suffix", default="_decoding_consistency.csv")
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--language", choices=["nl", "en"], default="nl")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe")
    parser.add_argument("--initial-prompt", default=segtx.DEFAULT_INITIAL_PROMPT)
    parser.add_argument("--initial-prompt-file", type=Path, default=None)
    parser.add_argument("--no-initial-prompt", action="store_true")
    parser.add_argument("--alternative-temperatures", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--sample-num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens-cap", type=int, default=128)
    parser.add_argument("--new-token-buffer", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no-seed", action="store_true", help="Disable deterministic per-alternative seeds.")
    parser.add_argument("--none-penalty", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--audio-ext", nargs="+", default=list(AUDIO_EXTENSIONS))
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", action="store_false", dest="overwrite")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=None,
        help="Optional CSV log path. Default: <raw-root-parent>/decoding_consistency_log.csv.",
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


def segment_path_for_audio(audio_path: Path, raw_root: Path, segment_root: Path, suffix: str) -> Path:
    relative = audio_path.relative_to(raw_root)
    return segment_root / relative.parent / f"{audio_path.stem}{suffix}"


def output_path_for_audio(segment_path: Path, audio_path: Path, suffix: str) -> Path:
    return segment_path.with_name(f"{audio_path.stem}{suffix}")


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.no_initial_prompt:
        return ""
    if args.initial_prompt_file is not None:
        return args.initial_prompt_file.read_text(encoding="utf-8").strip()
    return args.initial_prompt.strip()


def resolve_dtype(torch_module, dtype_name: str, device: str):
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name == "float32":
        return torch_module.float32
    return torch_module.float16 if str(device).startswith("cuda") else torch_module.float32


def segment_audio(audio: Any, segment: dict[str, Any]) -> tuple[Any, float, float]:
    start_t = float(segment.get("start", 0.0) or 0.0)
    end_t = segment.get("end")
    if end_t is None:
        end_t = len(audio) / segtx.SAMPLE_RATE
    end_t = float(end_t)
    return audio[int(start_t * segtx.SAMPLE_RATE) : int(end_t * segtx.SAMPLE_RATE)], start_t, end_t


def build_generation_inputs(
    *,
    audio_chunk: Any,
    feature_extractor: Any,
    prefix_ids: list[int],
    device: str,
    dtype: Any,
    torch_module: Any,
):
    inputs = feature_extractor(
        audio_chunk,
        sampling_rate=segtx.SAMPLE_RATE,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_features = inputs.input_features.to(device, dtype=dtype)
    attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, "attention_mask") else None
    decoder_input_ids = torch_module.tensor([prefix_ids], device=device, dtype=torch_module.long)
    return input_features, attention_mask, decoder_input_ids


def max_new_tokens_for_segment(
    *,
    segment: dict[str, Any],
    tokenizer: Any,
    model: Any,
    prefix_ids: list[int],
    max_new_tokens_cap: int,
    new_token_buffer: int,
) -> int:
    available_new_tokens = int(model.config.max_target_positions) - len(prefix_ids)
    if available_new_tokens <= 0:
        raise ValueError("prefix_ids are too long for this Whisper model; shorten initial prompt")
    text = str(segment.get("text", "") or "").strip()
    wanted = len(tokenizer.encode(text, add_special_tokens=False)) + new_token_buffer if text else max_new_tokens_cap
    return max(1, min(int(wanted), int(max_new_tokens_cap), available_new_tokens))


def token_piece(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=True, clean_up_tokenization_spaces=False)


def token_groups_to_words(tokenizer: Any, token_ids: list[int]) -> list[dict[str, Any]]:
    groups = []
    current = None
    for token_id in token_ids:
        piece = token_piece(tokenizer, token_id)
        if piece == "":
            continue
        starts_new_word = piece.startswith(" ") or current is None
        if starts_new_word:
            if current is not None and current["word"].strip():
                groups.append(current)
            current = {"word": piece}
        else:
            current["word"] += piece
    if current is not None and current["word"].strip():
        groups.append(current)
    return groups


def words_from_token_ids(tokenizer: Any, token_ids: list[int]) -> list[str]:
    return [group["word"].strip() for group in token_groups_to_words(tokenizer, token_ids) if group["word"].strip()]


def words_from_text(tokenizer: Any, text: str) -> list[str]:
    token_ids = tokenizer.encode(str(text or "").strip(), add_special_tokens=False)
    return words_from_token_ids(tokenizer, [int(token_id) for token_id in token_ids])


def clean_generated_ids(tokenizer: Any, gen_ids: list[int]) -> list[int]:
    eot_id = tokenizer.eos_token_id
    clean = []
    for token_id in gen_ids:
        token_id = int(token_id)
        if token_id == eot_id:
            break
        clean.append(token_id)
    return clean


def generate_segment_token_ids(
    *,
    audio: Any,
    segment: dict[str, Any],
    temperature: float,
    num_beams: int,
    feature_extractor: Any,
    tokenizer: Any,
    model: Any,
    prefix_ids: list[int],
    device: str,
    dtype: Any,
    torch_module: Any,
    max_new_tokens_cap: int,
    new_token_buffer: int,
    seed: int | None,
) -> list[int]:
    audio_chunk, _, _ = segment_audio(audio, segment)
    if len(audio_chunk) == 0:
        return []

    if seed is not None:
        torch_module.manual_seed(int(seed))
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(int(seed))

    input_features, attention_mask, decoder_input_ids = build_generation_inputs(
        audio_chunk=audio_chunk,
        feature_extractor=feature_extractor,
        prefix_ids=prefix_ids,
        device=device,
        dtype=dtype,
        torch_module=torch_module,
    )
    gen_config = deepcopy(model.generation_config)
    gen_config.forced_decoder_ids = None

    do_sample = float(temperature) > 0.0
    use_generic_generate = do_sample and int(num_beams) > 1
    generate_kwargs = dict(
        input_features=input_features,
        decoder_input_ids=decoder_input_ids,
        generation_config=gen_config,
        num_beams=int(num_beams),
        num_return_sequences=1,
        do_sample=do_sample,
        return_dict_in_generate=True,
        max_new_tokens=max_new_tokens_for_segment(
            segment=segment,
            tokenizer=tokenizer,
            model=model,
            prefix_ids=prefix_ids,
            max_new_tokens_cap=max_new_tokens_cap,
            new_token_buffer=new_token_buffer,
        ),
    )
    if not use_generic_generate:
        generate_kwargs["force_unique_generate_call"] = True
    if do_sample:
        generate_kwargs["temperature"] = float(temperature)
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask

    if use_generic_generate:
        output = GenerationMixin.generate(model, **generate_kwargs)
    else:
        output = model.generate(**generate_kwargs)

    full_ids = output.sequences[0].detach().cpu().tolist()
    gen_ids = full_ids[len(prefix_ids) :]
    return clean_generated_ids(tokenizer, gen_ids)


def align_alt_to_main_words(main_words: list[str], alt_words: list[str]) -> list[str | None]:
    aligned: list[str | None] = [None] * len(main_words)
    matcher = difflib.SequenceMatcher(a=main_words, b=alt_words, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                aligned[i1 + offset] = alt_words[j1 + offset]
        elif tag == "replace":
            span = min(i2 - i1, j2 - j1)
            for offset in range(span):
                aligned[i1 + offset] = alt_words[j1 + offset]
    return aligned


def compute_rows_for_segments(
    *,
    audio: Any,
    segments: list[dict[str, Any]],
    feature_extractor: Any,
    tokenizer: Any,
    model: Any,
    prefix_ids: list[int],
    device: str,
    dtype: Any,
    torch_module: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = []
    alt_temperatures = tuple(float(value) for value in args.alternative_temperatures)
    n_alternatives = len(alt_temperatures)
    seed_base = None if args.no_seed else int(args.seed)

    with torch_module.inference_mode():
        for segment_id, segment in enumerate(segments):
            main_words = words_from_text(tokenizer, str(segment.get("text", "") or ""))
            aligned_word_lists = []

            for alt_idx, temperature in enumerate(alt_temperatures):
                alt_seed = None if seed_base is None else seed_base + alt_idx + 1
                alt_ids = generate_segment_token_ids(
                    audio=audio,
                    segment=segment,
                    temperature=temperature,
                    num_beams=args.sample_num_beams,
                    feature_extractor=feature_extractor,
                    tokenizer=tokenizer,
                    model=model,
                    prefix_ids=prefix_ids,
                    device=device,
                    dtype=dtype,
                    torch_module=torch_module,
                    max_new_tokens_cap=args.max_new_tokens_cap,
                    new_token_buffer=args.new_token_buffer,
                    seed=alt_seed,
                )
                alt_words = words_from_token_ids(tokenizer, alt_ids)
                aligned_word_lists.append(align_alt_to_main_words(main_words, alt_words))

            for word_idx, word in enumerate(main_words):
                aligned = [aligned_words[word_idx] for aligned_words in aligned_word_lists]
                matches = [int(alt_word == word) for alt_word in aligned]
                n_none = int(sum(alt_word is None for alt_word in aligned))
                raw_score = 1.0 + float(sum(matches)) - float(args.none_penalty) * n_none
                consistency = max(0.0, min(1.0, raw_score / float(n_alternatives + 1)))
                rows.append(
                    {
                        "segment_id": segment_id,
                        "word_idx": word_idx,
                        "asr_word": word,
                        "decoding_consistency": consistency,
                        "aligned_alternative_words": aligned,
                    }
                )
            if args.verbose:
                print(f"  segment {segment_id}: {len(main_words)} words")
    return rows


def write_score_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=KEEP_COLUMNS)
        writer.writeheader()
        for row in rows:
            clean = dict(row)
            clean["aligned_alternative_words"] = json.dumps(clean["aligned_alternative_words"], ensure_ascii=False)
            writer.writerow(clean)


def write_log_header(log_path: Path) -> None:
    if log_path.exists():
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "audio", "segment", "output_csv", "rows", "status", "duration_sec", "error"],
        )
        writer.writeheader()


def append_log(log_path: Path, row: dict[str, Any]) -> None:
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "audio", "segment", "output_csv", "rows", "status", "duration_sec", "error"],
        )
        writer.writerow(row)


def process_one(
    *,
    audio_path: Path,
    segment_path: Path,
    output_csv: Path,
    feature_extractor: Any,
    tokenizer: Any,
    model: Any,
    prefix_ids: list[int],
    device: str,
    dtype: Any,
    torch_module: Any,
    args: argparse.Namespace,
) -> int:
    import whisperx

    audio = whisperx.load_audio(str(audio_path))
    segments = segtx.load_segments(segment_path)
    rows = compute_rows_for_segments(
        audio=audio,
        segments=segments,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        model=model,
        prefix_ids=prefix_ids,
        device=device,
        dtype=dtype,
        torch_module=torch_module,
        args=args,
    )
    write_score_csv(output_csv, rows)
    return len(rows)


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    if not raw_root.exists():
        print(f"[ERROR] raw root does not exist: {raw_root}", file=sys.stderr)
        return 2

    log_path = (args.log_csv or raw_root.parent / "decoding_consistency_log.csv").expanduser().resolve()
    write_log_header(log_path)

    audio_files = iter_audio_files(raw_root, args.audio_ext)
    if args.limit is not None:
        audio_files = audio_files[: args.limit]

    print(f"[INFO] raw_root={raw_root}")
    print(f"[INFO] found_audio={len(audio_files)}")
    print(f"[INFO] models={' '.join(args.models)}")
    print(f"[INFO] alternative_temperatures={args.alternative_temperatures}")
    print(f"[INFO] sample_num_beams={args.sample_num_beams}")
    print(f"[INFO] log={log_path}")

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, WhisperFeatureExtractor, WhisperTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(torch, args.dtype, device)
    cache_dir = None if args.model_cache_dir is None else str(args.model_cache_dir.expanduser())
    initial_prompt = resolve_prompt(args)

    for model_name in args.models:
        model_id = segtx.MODEL_IDS[model_name]
        segment_root = segment_root_for_model(args, model_name)
        print(f"[INFO] loading {model_name}: {model_id}")
        print(f"[INFO] segment_root={segment_root}")

        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=args.local_files_only,
        )
        tokenizer = WhisperTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            language=args.language,
            task=args.task,
            local_files_only=args.local_files_only,
        )
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            local_files_only=args.local_files_only,
        ).to(device).eval()
        prefix_ids = segtx.build_prefix_ids(tokenizer, initial_prompt)

        for index, audio_path in enumerate(audio_files, start=1):
            segment_path = segment_path_for_audio(audio_path, raw_root, segment_root, args.segment_suffix)
            output_csv = output_path_for_audio(segment_path, audio_path, args.output_suffix)
            started = time.time()

            if not segment_path.exists():
                print(f"[{model_name} {index}/{len(audio_files)}] missing segment: {segment_path}")
                append_log(
                    log_path,
                    {
                        "model": model_name,
                        "audio": str(audio_path),
                        "segment": str(segment_path),
                        "output_csv": str(output_csv),
                        "rows": 0,
                        "status": "missing_segment",
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
                        "segment": str(segment_path),
                        "output_csv": str(output_csv),
                        "rows": "",
                        "status": "skipped",
                        "duration_sec": "0.000",
                        "error": "",
                    },
                )
                continue

            print(f"[{model_name} {index}/{len(audio_files)}] scoring: {audio_path}")
            try:
                n_rows = process_one(
                    audio_path=audio_path,
                    segment_path=segment_path,
                    output_csv=output_csv,
                    feature_extractor=feature_extractor,
                    tokenizer=tokenizer,
                    model=model,
                    prefix_ids=prefix_ids,
                    device=device,
                    dtype=dtype,
                    torch_module=torch,
                    args=args,
                )
                duration = time.time() - started
                append_log(
                    log_path,
                    {
                        "model": model_name,
                        "audio": str(audio_path),
                        "segment": str(segment_path),
                        "output_csv": str(output_csv),
                        "rows": n_rows,
                        "status": "ok",
                        "duration_sec": f"{duration:.3f}",
                        "error": "",
                    },
                )
            except Exception as exc:
                duration = time.time() - started
                print(f"[ERROR] {model_name} failed for {audio_path}: {exc}", file=sys.stderr)
                append_log(
                    log_path,
                    {
                        "model": model_name,
                        "audio": str(audio_path),
                        "segment": str(segment_path),
                        "output_csv": str(output_csv),
                        "rows": "",
                        "status": "error",
                        "duration_sec": f"{duration:.3f}",
                        "error": repr(exc),
                    },
                )

        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    print(f"[INFO] done. log={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
