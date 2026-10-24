"""Batch cross-attention entropy tables from generated *_newsegment.txt files.

Input layout:

    raw/speaker/speaker.wav
    v2_segment/speaker/speaker_newsegment.txt
    v3_segment/speaker/speaker_newsegment.txt

Default output next to each newsegment file:

    v2_segment/speaker/speaker_cross_attention.csv
    v3_segment/speaker/speaker_cross_attention.csv

Output columns:

    segment_id, word_idx, asr_word, attn_entropy_mean

The attention logic mirrors whisper_cross_attention_uncertainty_v3.ipynb:
generate token ids first, then compute normalised cross-attention entropy from
decoder layer 22 with padding frames excluded.
"""

from __future__ import annotations

import argparse
import csv
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
KEEP_COLUMNS = ["segment_id", "word_idx", "asr_word", "attn_entropy_mean"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate word-level cross-attention entropy tables for speaker wavs and *_newsegment.txt files."
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
    parser.add_argument("--output-suffix", default="_cross_attention.csv")
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--language", choices=["nl", "en"], default="nl")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe")
    parser.add_argument("--initial-prompt", default=segtx.DEFAULT_INITIAL_PROMPT)
    parser.add_argument("--initial-prompt-file", type=Path, default=None)
    parser.add_argument("--no-initial-prompt", action="store_true")
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-new-tokens-cap", type=int, default=128)
    parser.add_argument("--new-token-buffer", type=int, default=16)
    parser.add_argument("--attention-layer", type=int, default=22)
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
        help="Optional CSV log path. Default: <raw-root-parent>/cross_attention_log.csv.",
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


def load_whisper_model(
    model_class: Any,
    model_id: str,
    *,
    cache_dir: str | None,
    dtype: Any,
    local_files_only: bool,
    device: str,
):
    kwargs = dict(
        cache_dir=cache_dir,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        attn_implementation="eager",
    )
    try:
        return model_class.from_pretrained(model_id, **kwargs).to(device).eval()
    except TypeError:
        kwargs.pop("attn_implementation")
        return model_class.from_pretrained(model_id, **kwargs).to(device).eval()


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
    feature_extractor: Any,
    tokenizer: Any,
    model: Any,
    prefix_ids: list[int],
    device: str,
    dtype: Any,
    torch_module: Any,
    num_beams: int,
    max_new_tokens_cap: int,
    new_token_buffer: int,
) -> tuple[list[int], Any, float, float]:
    audio_chunk, start_t, end_t = segment_audio(audio, segment)
    if len(audio_chunk) == 0:
        return [], audio_chunk, start_t, end_t

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

    generate_kwargs = dict(
        input_features=input_features,
        decoder_input_ids=decoder_input_ids,
        generation_config=gen_config,
        num_beams=int(num_beams),
        num_return_sequences=1,
        do_sample=False,
        return_dict_in_generate=True,
        max_new_tokens=max_new_tokens_for_segment(
            segment=segment,
            tokenizer=tokenizer,
            model=model,
            prefix_ids=prefix_ids,
            max_new_tokens_cap=max_new_tokens_cap,
            new_token_buffer=new_token_buffer,
        ),
        force_unique_generate_call=True,
    )
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask

    output = model.generate(**generate_kwargs)
    full_ids = output.sequences[0].detach().cpu().tolist()
    gen_ids = full_ids[len(prefix_ids) :]
    return clean_generated_ids(tokenizer, gen_ids), audio_chunk, start_t, end_t


def mel_attention_mask_to_encoder_frames(attention_mask: Any):
    t_mel = attention_mask.shape[-1]
    t_enc = t_mel // 2
    mask = attention_mask[:, : t_enc * 2].reshape(1, t_enc, 2)
    return mask.min(dim=-1).values


def token_cross_attention_entropies(
    *,
    audio_chunk: Any,
    token_ids: list[int],
    feature_extractor: Any,
    model: Any,
    prefix_ids: list[int],
    device: str,
    dtype: Any,
    torch_module: Any,
    attention_layer: int,
) -> list[float]:
    if not token_ids:
        return []

    input_features, attention_mask, _ = build_generation_inputs(
        audio_chunk=audio_chunk,
        feature_extractor=feature_extractor,
        prefix_ids=prefix_ids,
        device=device,
        dtype=dtype,
        torch_module=torch_module,
    )
    full_decoder_ids = torch_module.tensor(
        [prefix_ids + list(map(int, token_ids))],
        device=device,
        dtype=torch_module.long,
    )

    forward_kwargs = dict(
        input_features=input_features,
        decoder_input_ids=full_decoder_ids,
        output_attentions=True,
        return_dict=True,
    )
    if attention_mask is not None:
        forward_kwargs["attention_mask"] = attention_mask

    output = model(**forward_kwargs)
    if output.cross_attentions is None:
        raise RuntimeError(
            "model did not return cross_attentions. Reload the model with "
            "attn_implementation='eager' or use a Transformers version that supports "
            "Whisper cross-attention outputs."
        )
    cross_attentions = output.cross_attentions
    layer_index = int(attention_layer)
    if layer_index < 0:
        layer_index = len(cross_attentions) + layer_index
    if layer_index < 0 or layer_index >= len(cross_attentions):
        raise ValueError(f"attention_layer={attention_layer} out of range for {len(cross_attentions)} layers")

    final_cross = cross_attentions[layer_index][0].detach().float().cpu()
    token_cross = final_cross[:, len(prefix_ids) : len(prefix_ids) + len(token_ids), :]
    if token_cross.shape[1] != len(token_ids):
        raise ValueError(
            f"Unexpected cross-attention shape: {tuple(token_cross.shape)} for {len(token_ids)} tokens"
        )

    t_enc = token_cross.shape[-1]
    if attention_mask is not None:
        enc_mask = mel_attention_mask_to_encoder_frames(attention_mask.cpu())
        real_mask = enc_mask[0].bool()
        if real_mask.shape[0] > t_enc:
            real_mask = real_mask[:t_enc]
        elif real_mask.shape[0] < t_enc:
            pad = torch_module.zeros(t_enc - real_mask.shape[0], dtype=torch_module.bool)
            real_mask = torch_module.cat([real_mask, pad])
        t_real = int(real_mask.sum().item())
    else:
        real_mask = None
        t_real = t_enc

    eps = 1e-12
    if real_mask is not None and t_real > 1:
        masked = token_cross * real_mask.float()
        masked = masked.clamp_min(eps)
        masked = masked / masked.sum(dim=-1, keepdim=True)
        denom = math.log(t_real)
        per_head_entropy = -(masked * masked.log()).sum(dim=-1) / denom
    else:
        denom = math.log(t_enc) if t_enc > 1 else 1.0
        attn = token_cross.clamp_min(eps)
        per_head_entropy = -(attn * attn.log()).sum(dim=-1) / denom

    return per_head_entropy.mean(dim=0).numpy().astype(float).tolist()


def token_piece(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=True, clean_up_tokenization_spaces=False)


def token_groups_to_words(tokenizer: Any, token_ids: list[int], attn_entropies: list[float]) -> list[dict[str, Any]]:
    groups = []
    current = None
    for token_id, entropy in zip(token_ids, attn_entropies):
        piece = token_piece(tokenizer, token_id)
        if piece == "":
            continue
        starts_new_word = piece.startswith(" ") or current is None
        if starts_new_word:
            if current is not None and current["word"].strip():
                groups.append(current)
            current = {"word": piece, "attn_entropy": []}
        else:
            current["word"] += piece
        current["attn_entropy"].append(float(entropy))
    if current is not None and current["word"].strip():
        groups.append(current)
    return groups


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


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
    with torch_module.inference_mode():
        for segment_id, segment in enumerate(segments):
            token_ids, audio_chunk, _, _ = generate_segment_token_ids(
                audio=audio,
                segment=segment,
                feature_extractor=feature_extractor,
                tokenizer=tokenizer,
                model=model,
                prefix_ids=prefix_ids,
                device=device,
                dtype=dtype,
                torch_module=torch_module,
                num_beams=args.num_beams,
                max_new_tokens_cap=args.max_new_tokens_cap,
                new_token_buffer=args.new_token_buffer,
            )
            token_entropies = token_cross_attention_entropies(
                audio_chunk=audio_chunk,
                token_ids=token_ids,
                feature_extractor=feature_extractor,
                model=model,
                prefix_ids=prefix_ids,
                device=device,
                dtype=dtype,
                torch_module=torch_module,
                attention_layer=args.attention_layer,
            )
            groups = token_groups_to_words(tokenizer, token_ids, token_entropies)
            for word_idx, group in enumerate(groups):
                rows.append(
                    {
                        "segment_id": segment_id,
                        "word_idx": word_idx,
                        "asr_word": group["word"].strip(),
                        "attn_entropy_mean": mean(group["attn_entropy"]),
                    }
                )
            if args.verbose:
                print(f"  segment {segment_id}: {len(groups)} words")
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

    log_path = (args.log_csv or raw_root.parent / "cross_attention_log.csv").expanduser().resolve()
    write_log_header(log_path)

    audio_files = iter_audio_files(raw_root, args.audio_ext)
    if args.limit is not None:
        audio_files = audio_files[: args.limit]

    print(f"[INFO] raw_root={raw_root}")
    print(f"[INFO] found_audio={len(audio_files)}")
    print(f"[INFO] models={' '.join(args.models)}")
    print(f"[INFO] attention_layer={args.attention_layer}")
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
        model = load_whisper_model(
            AutoModelForSpeechSeq2Seq,
            model_id,
            cache_dir=cache_dir,
            dtype=dtype,
            local_files_only=args.local_files_only,
            device=device,
        )
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
