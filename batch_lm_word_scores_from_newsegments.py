"""Batch LM word scoring from generated *_newsegment.txt files.

Input layout:

    raw/speaker/speaker.wav
    v2_segment/speaker/speaker_newsegment.txt
    v3_segment/speaker/speaker_newsegment.txt

Default output next to each newsegment file:

    v2_segment/speaker/speaker_word_scores.csv
    v3_segment/speaker/speaker_word_scores.csv

Output columns:

    segment_id, word_idx, asr_word, lm_confidence, lm_entropy, lm_margin, token_pieces

The scoring logic mirrors whisper_lm_word_scores.ipynb::score_segment_beam:
beam search with output_scores=True, no score renormalization, and token grouping
by decoded token pieces.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

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
    "lm_confidence",
    "lm_entropy",
    "lm_margin",
    "token_pieces",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate word-level LM confidence tables for speaker wavs and *_newsegment.txt files."
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
        help="Models to process. Default: large-v2 large-v3.",
    )
    parser.add_argument(
        "--segment-root-large-v2",
        type=Path,
        default=None,
        help="Override segment root for large-v2. Default: <raw-root-parent>/v2_segment.",
    )
    parser.add_argument(
        "--segment-root-large-v3",
        type=Path,
        default=None,
        help="Override segment root for large-v3. Default: <raw-root-parent>/v3_segment.",
    )
    parser.add_argument(
        "--segment-suffix",
        default="_newsegment.txt",
        help="Segment filename suffix. Default: _newsegment.txt.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_word_scores.csv",
        help="Output CSV filename suffix. Default: _word_scores.csv.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer/feature extractor only from local cache.",
    )
    parser.add_argument("--language", choices=["nl", "en"], default="nl")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe")
    parser.add_argument(
        "--initial-prompt",
        default=segtx.DEFAULT_INITIAL_PROMPT,
        help="Initial prompt text. Defaults to the project prompt used in the notebook.",
    )
    parser.add_argument("--initial-prompt-file", type=Path, default=None)
    parser.add_argument("--no-initial-prompt", action="store_true")
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-new-tokens-cap", type=int, default=128)
    parser.add_argument("--new-token-buffer", type=int, default=16)
    parser.add_argument(
        "--device",
        default=None,
        help="Device for Transformers model, e.g. cuda, cuda:0, cpu. Default: cuda if available else cpu.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "float32"],
        default="auto",
        help="Tensor dtype. auto uses float16 on CUDA and float32 on CPU.",
    )
    parser.add_argument(
        "--audio-ext",
        nargs="+",
        default=list(AUDIO_EXTENSIONS),
        help="Audio extensions to process. Default: .wav .flac .mp3 .m4a .ogg",
    )
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", action="store_false", dest="overwrite")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=None,
        help="Optional CSV log path. Default: <raw-root-parent>/lm_word_scores_log.csv.",
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


def token_piece(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def token_groups_to_words(
    tokenizer: Any,
    token_ids: list[int],
    token_logprobs: list[float],
    token_entropies: list[float],
    token_margins: list[float],
) -> list[dict[str, Any]]:
    groups = []
    current = None

    for token_id, logprob, ent, margin in zip(token_ids, token_logprobs, token_entropies, token_margins):
        piece = token_piece(tokenizer, token_id)
        if piece == "":
            continue

        starts_new_word = piece.startswith(" ") or current is None
        if starts_new_word:
            if current is not None and current["word"].strip():
                groups.append(current)
            current = {"word": piece, "logprob": [], "entropy": [], "margin": [], "pieces": []}
        else:
            current["word"] += piece

        current["pieces"].append(piece)
        current["logprob"].append(float(logprob))
        current["entropy"].append(float(ent))
        current["margin"].append(float(margin))

    if current is not None and current["word"].strip():
        groups.append(current)
    return groups


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def summarize_group(group: dict[str, Any]) -> dict[str, Any]:
    logprob = [float(value) for value in group["logprob"]]
    entropy = [float(value) for value in group["entropy"]]
    margin = [float(value) for value in group["margin"]]
    return {
        "word": group["word"].strip(),
        "token_pieces": group["pieces"],
        "lm_confidence": float(math.exp(mean(logprob))),
        "lm_entropy": mean(entropy),
        "lm_margin": mean(margin),
    }


def score_segment_beam(
    *,
    audio: Any,
    segment: dict[str, Any],
    seg_id: int,
    feature_extractor: Any,
    tokenizer: Any,
    model: Any,
    prefix_ids: list[int],
    sample_rate: int,
    device: str,
    dtype: Any,
    torch_module: Any,
    num_beams: int,
    max_new_tokens_cap: int,
    new_token_buffer: int,
) -> list[dict[str, Any]]:
    start_t = float(segment["start"])
    end_t = segment.get("end")
    if end_t is None:
        end_t = len(audio) / sample_rate
    end_t = float(end_t)

    start_sample = int(start_t * sample_rate)
    end_sample = int(end_t * sample_rate)
    audio_chunk = audio[start_sample:end_sample]
    if len(audio_chunk) == 0:
        return []

    inputs = feature_extractor(
        audio_chunk,
        sampling_rate=sample_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_features = inputs.input_features.to(device, dtype=dtype)
    attention_mask = None
    if hasattr(inputs, "attention_mask"):
        attention_mask = inputs.attention_mask.to(device)

    decoder_input_ids = torch_module.tensor([prefix_ids], device=device, dtype=torch_module.long)

    vocab_size = model.config.vocab_size
    if int(decoder_input_ids.min()) < 0 or int(decoder_input_ids.max()) >= vocab_size:
        raise ValueError(
            "prefix_ids contain invalid token ids: "
            f"min={int(decoder_input_ids.min())}, "
            f"max={int(decoder_input_ids.max())}, "
            f"vocab_size={vocab_size}"
        )

    prefix_len = decoder_input_ids.shape[-1]
    available_new_tokens = model.config.max_target_positions - prefix_len
    if available_new_tokens <= 0:
        raise ValueError(
            f"prefix_ids too long: prefix_len={prefix_len}, "
            f"max_target_positions={model.config.max_target_positions}"
        )

    text = segment.get("text", "")
    text = "" if text is None else str(text).strip()
    if text:
        wanted_new_tokens = len(tokenizer.encode(text, add_special_tokens=False)) + new_token_buffer
    else:
        wanted_new_tokens = max_new_tokens_cap

    max_new_tokens = min(
        int(wanted_new_tokens),
        int(max_new_tokens_cap),
        int(available_new_tokens),
    )
    max_new_tokens = max(1, max_new_tokens)

    gen_config = deepcopy(model.generation_config)
    gen_config.forced_decoder_ids = None

    generate_kwargs = dict(
        input_features=input_features,
        decoder_input_ids=decoder_input_ids,
        generation_config=gen_config,
        num_beams=num_beams,
        num_return_sequences=1,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
        max_new_tokens=max_new_tokens,
        force_unique_generate_call=True,
    )
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask

    output = model.generate(**generate_kwargs)
    if output.scores is None or len(output.scores) == 0:
        raise RuntimeError("model.generate() did not return scores")

    full_ids = output.sequences[0].detach().cpu().tolist()
    gen_ids = full_ids[prefix_len:]
    n_steps = min(len(output.scores), len(gen_ids))
    if n_steps == 0:
        return []

    eot_id = tokenizer.eos_token_id
    clean_gen_ids: list[int] = []
    token_logprobs: list[float] = []
    token_entropies: list[float] = []
    token_margins: list[float] = []

    for step in range(n_steps):
        token_id = int(gen_ids[step])
        if token_id == eot_id:
            break

        score_t = output.scores[step].detach().float().cpu()
        if score_t.ndim != 2:
            raise ValueError(f"Unexpected score_t shape at step {step}: {score_t.shape}")

        if score_t.shape[0] == 1:
            lp_vec = score_t[0]
        else:
            if output.beam_indices is None:
                beam_row = 0
            else:
                beam_indices = output.beam_indices[0].detach().cpu()
                if beam_indices.numel() == len(gen_ids):
                    beam_row = int(beam_indices[step])
                elif beam_indices.numel() == len(full_ids):
                    beam_row = int(beam_indices[prefix_len + step])
                elif beam_indices.numel() == n_steps:
                    beam_row = int(beam_indices[step])
                else:
                    raise ValueError(
                        f"Unexpected beam_indices length: {beam_indices.numel()}, "
                        f"full_len={len(full_ids)}, gen_len={len(gen_ids)}, n_steps={n_steps}"
                    )
            if beam_row < 0 or beam_row >= score_t.shape[0]:
                raise ValueError(f"Invalid beam_row={beam_row} at step {step}")
            lp_vec = score_t[beam_row]

        finite = torch_module.isfinite(lp_vec)
        if not bool(finite[token_id]):
            selected_lp = float("nan")
        else:
            safe_lp = torch_module.where(finite, lp_vec, torch_module.zeros_like(lp_vec))
            safe_p = torch_module.where(finite, lp_vec.exp(), torch_module.zeros_like(lp_vec))
            selected_lp = float(lp_vec[token_id])
            entropy = float(-(safe_p * safe_lp).sum())
            top2 = safe_p.topk(2).values
            margin = float(top2[0] - top2[1])

            clean_gen_ids.append(token_id)
            token_logprobs.append(selected_lp)
            token_entropies.append(entropy)
            token_margins.append(margin)

    if not clean_gen_ids:
        return []

    groups = token_groups_to_words(
        tokenizer,
        clean_gen_ids,
        token_logprobs,
        token_entropies,
        token_margins,
    )

    rows = []
    source_words = segment.get("words") or []
    for word_idx, group in enumerate(groups):
        summary = summarize_group(group)
        generated_word = summary["word"]
        if word_idx < len(source_words):
            source = source_words[word_idx]
            asr_word = str(source.get("word", source.get("text", generated_word))).strip()
        else:
            asr_word = generated_word
        rows.append(
            {
                "segment_id": seg_id,
                "word_idx": word_idx,
                "asr_word": asr_word,
                "lm_confidence": summary["lm_confidence"],
                "lm_entropy": summary["lm_entropy"],
                "lm_margin": summary["lm_margin"],
                "token_pieces": summary["token_pieces"],
            }
        )
    return rows


def write_score_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=KEEP_COLUMNS)
        writer.writeheader()
        for row in rows:
            clean = dict(row)
            clean["token_pieces"] = json.dumps(clean["token_pieces"], ensure_ascii=False)
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
    all_rows: list[dict[str, Any]] = []
    with torch_module.inference_mode():
        for seg_id, segment in enumerate(segments):
            rows = score_segment_beam(
                audio=audio,
                segment=segment,
                seg_id=seg_id,
                feature_extractor=feature_extractor,
                tokenizer=tokenizer,
                model=model,
                prefix_ids=prefix_ids,
                sample_rate=segtx.SAMPLE_RATE,
                device=device,
                dtype=dtype,
                torch_module=torch_module,
                num_beams=args.num_beams,
                max_new_tokens_cap=args.max_new_tokens_cap,
                new_token_buffer=args.new_token_buffer,
            )
            all_rows.extend(rows)
            if args.verbose:
                print(f"  segment {seg_id}: {len(rows)} words")
    write_score_csv(output_csv, all_rows)
    return len(all_rows)


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    if not raw_root.exists():
        print(f"[ERROR] raw root does not exist: {raw_root}", file=sys.stderr)
        return 2

    log_path = (args.log_csv or raw_root.parent / "lm_word_scores_log.csv").expanduser().resolve()
    write_log_header(log_path)

    audio_files = iter_audio_files(raw_root, args.audio_ext)
    if args.limit is not None:
        audio_files = audio_files[: args.limit]

    print(f"[INFO] raw_root={raw_root}")
    print(f"[INFO] found_audio={len(audio_files)}")
    print(f"[INFO] models={' '.join(args.models)}")
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
