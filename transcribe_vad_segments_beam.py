"""Transcribe fixed VAD segments with the same generation setup as score_segment_beam.

Input:
    speaker.wav
    speaker_segment.txt

Output by default in the same folder as speaker_segment.txt:
    speaker.txt
    speaker_newsegment.txt

The segment file may be JSONL like:
    # audio: speaker.wav
    {"start": 0.0, "end": 3.2}
    {"text": "old hypothesis", "start": 4.0, "end": 8.0}

This script keeps the model.generate inputs and decoding parameters aligned with
whisper_lm_word_scores.ipynb::score_segment_beam, but skips output_scores and all
LM uncertainty scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


SAMPLE_RATE = 16000

DEFAULT_INITIAL_PROMPT = (
    "eerst ee rst eer st ehij geeft ze allemaal een naam eerste was hhij geeft ze allema "
    "ehij geeft ze allemaal een naamt et p lagened plad gen flagen plagen dat vond maar "
    "tje maar maarte maart t maartje wel leuk maar l eu k loek luk maar naar maar toen "
    "het pesten pe te sten werd is ze mevrouw fre f r e d d e r fre dder fledder vledder "
    "gaan hel pen h e l p e n gaan helpen hand gebaren handgebaren"
)

MODEL_IDS = {
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe one audio file using supplied VAD segments and the same "
            "Transformers Whisper beam generation setup as score_segment_beam."
        )
    )
    parser.add_argument("audio", type=Path, help="Input audio path, e.g. speaker.wav.")
    parser.add_argument(
        "segments",
        type=Path,
        help="Input VAD/segment file, e.g. speaker_segment.txt.",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=None,
        help="Transcript output path. Default: <audio_stem>.txt next to the segment file.",
    )
    parser.add_argument(
        "--output-segments",
        type=Path,
        default=None,
        help="Generated segment JSONL output path. Default: <audio_stem>_newsegment.txt next to the segment file.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_IDS),
        default="large-v2",
        help="Whisper model size/version.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional full Hugging Face model id. Overrides --model.",
    )
    parser.add_argument(
        "--language",
        choices=["nl", "en"],
        default="nl",
        help="Whisper language code.",
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Whisper task. score_segment_beam used transcribe.",
    )
    parser.add_argument(
        "--initial-prompt",
        default=DEFAULT_INITIAL_PROMPT,
        help="Initial prompt text. Defaults to the project prompt from the notebook.",
    )
    parser.add_argument(
        "--initial-prompt-file",
        type=Path,
        default=None,
        help="Read initial prompt from a UTF-8 text file.",
    )
    parser.add_argument(
        "--no-initial-prompt",
        action="store_true",
        help="Disable initial prompt.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=5,
        help="Beam count. Default 5 uses beam search with do_sample=False.",
    )
    parser.add_argument(
        "--max-new-tokens-cap",
        type=int,
        default=128,
        help="Maximum new tokens per segment. score_segment_beam default is 128.",
    )
    parser.add_argument(
        "--new-token-buffer",
        type=int,
        default=16,
        help="Extra tokens when input segment has text. score_segment_beam default is 16.",
    )
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite output files if they already exist. This is the default.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        help="Refuse to run when output files already exist.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each generated segment while running.",
    )
    return parser.parse_args()


def load_segments(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    data_lines = [line for line in lines if not line.startswith("#")]

    jsonl_segments: list[dict[str, Any]] = []
    jsonl_ok = True
    for line in data_lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            jsonl_ok = False
            break
        if not isinstance(obj, dict):
            jsonl_ok = False
            break
        jsonl_segments.append(obj)

    if jsonl_ok and jsonl_segments:
        return normalize_segments(jsonl_segments)

    obj = json.loads(raw)
    if isinstance(obj, dict) and "segments" in obj:
        return normalize_segments(obj["segments"])
    if isinstance(obj, list):
        return normalize_segments(obj)
    if isinstance(obj, dict):
        return normalize_segments([obj])

    raise ValueError(f"Unsupported segment file format: {path}")


def normalize_segments(segments: Any) -> list[dict[str, Any]]:
    if not isinstance(segments, list):
        raise ValueError("segments must be a list")

    out: list[dict[str, Any]] = []
    for idx, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"segment {idx} is not a JSON object")
        if "start" not in segment:
            raise ValueError(f"segment {idx} is missing start")

        start = float(segment["start"])
        end = segment.get("end")
        end = None if end is None else float(end)

        new_segment = dict(segment)
        new_segment["id"] = int(segment.get("id", idx))
        new_segment["start"] = start
        new_segment["end"] = end
        if "text" in new_segment and new_segment["text"] is None:
            new_segment["text"] = ""
        out.append(new_segment)

    return out


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.no_initial_prompt:
        return ""
    if args.initial_prompt_file is not None:
        return args.initial_prompt_file.read_text(encoding="utf-8").strip()
    return args.initial_prompt.strip()


def build_prefix_ids(
    tokenizer: Any,
    prompt_text: str,
) -> list[int]:
    prompt_ids = (
        tokenizer.encode(" " + prompt_text.strip(), add_special_tokens=False)
        if prompt_text
        else []
    )
    sot_prev_id = tokenizer.convert_tokens_to_ids("<|startofprev|>")
    prompt_prefix_ids = [sot_prev_id] + prompt_ids if prompt_ids else []
    return prompt_prefix_ids + list(tokenizer.prefix_tokens)


def output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    audio_path = args.audio.expanduser().resolve()
    segments_path = args.segments.expanduser().resolve()
    output_txt = args.output_txt or segments_path.with_name(f"{audio_path.stem}.txt")
    output_segments = args.output_segments or segments_path.with_name(f"{audio_path.stem}_newsegment.txt")
    return output_txt.expanduser().resolve(), output_segments.expanduser().resolve()


def decode_generated_ids(tokenizer: Any, token_ids: list[int]) -> str:
    eot_id = tokenizer.eos_token_id
    if eot_id in token_ids:
        token_ids = token_ids[: token_ids.index(eot_id)]
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def transcribe_segment_beam(
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
) -> dict[str, Any]:
    start_t = float(segment["start"])
    end_t = segment.get("end")
    if end_t is None:
        end_t = len(audio) / sample_rate
    end_t = float(end_t)

    start_sample = int(start_t * sample_rate)
    end_sample = int(end_t * sample_rate)
    audio_chunk = audio[start_sample:end_sample]

    if len(audio_chunk) == 0:
        return {"id": seg_id, "start": start_t, "end": end_t, "text": ""}

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

    decoder_input_ids = torch_module.tensor(
        [prefix_ids],
        device=device,
        dtype=torch_module.long,
    )

    vocab_size = model.config.vocab_size
    if int(decoder_input_ids.min()) < 0 or int(decoder_input_ids.max()) >= vocab_size:
        raise ValueError(
            "prefix_ids contain invalid token ids: "
            f"min={int(decoder_input_ids.min())}, "
            f"max={int(decoder_input_ids.max())}, "
            f"vocab_size={vocab_size}"
        )

    prefix_len = decoder_input_ids.shape[-1]
    max_target_positions = model.config.max_target_positions
    available_new_tokens = max_target_positions - prefix_len
    if available_new_tokens <= 0:
        raise ValueError(
            f"prefix_ids too long: prefix_len={prefix_len}, "
            f"max_target_positions={max_target_positions}"
        )

    source_text = segment.get("text", "")
    source_text = "" if source_text is None else str(source_text).strip()
    if source_text:
        text_token_ids = tokenizer.encode(source_text, add_special_tokens=False)
        wanted_new_tokens = len(text_token_ids) + new_token_buffer
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
        max_new_tokens=max_new_tokens,
        force_unique_generate_call=True,
    )
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask

    sequences = model.generate(**generate_kwargs)
    full_ids = sequences[0].detach().cpu().tolist()
    gen_ids = full_ids[prefix_len:]
    text = decode_generated_ids(tokenizer, gen_ids)

    return {
        "id": seg_id,
        "start": start_t,
        "end": end_t,
        "text": text,
    }


def write_outputs(
    *,
    audio_path: Path,
    output_txt: Path,
    output_segments: Path,
    generated_segments: list[dict[str, Any]],
) -> None:
    transcript = " ".join(
        str(segment.get("text", "")).strip()
        for segment in generated_segments
        if str(segment.get("text", "")).strip()
    ).strip()

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_segments.parent.mkdir(parents=True, exist_ok=True)

    output_txt.write_text(transcript + "\n", encoding="utf-8")
    with output_segments.open("w", encoding="utf-8") as handle:
        handle.write(f"# audio: {audio_path.name}\n")
        for segment in generated_segments:
            handle.write(json.dumps(segment, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    audio_path = args.audio.expanduser().resolve()
    segments_path = args.segments.expanduser().resolve()
    output_txt, output_segments = output_paths(args)

    if not audio_path.exists():
        print(f"[ERROR] audio does not exist: {audio_path}", file=sys.stderr)
        return 2
    if not segments_path.exists():
        print(f"[ERROR] segments file does not exist: {segments_path}", file=sys.stderr)
        return 2
    if not args.overwrite and (output_txt.exists() or output_segments.exists()):
        print(
            "[ERROR] output exists; use --overwrite or choose different output paths: "
            f"{output_txt}, {output_segments}",
            file=sys.stderr,
        )
        return 2

    import torch
    import whisperx
    from transformers import AutoModelForSpeechSeq2Seq, WhisperFeatureExtractor, WhisperTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float32":
        dtype = torch.float32
    else:
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

    model_id = args.model_id or MODEL_IDS[args.model]
    cache_dir = None if args.model_cache_dir is None else str(args.model_cache_dir.expanduser())
    initial_prompt = resolve_prompt(args)

    print(f"[INFO] audio={audio_path}")
    print(f"[INFO] segments={segments_path}")
    print(f"[INFO] model_id={model_id}")
    print(f"[INFO] language={args.language}, task={args.task}, device={device}, dtype={dtype}")
    print(f"[INFO] initial_prompt={'yes' if initial_prompt else 'no'}")

    segments = load_segments(segments_path)
    audio = whisperx.load_audio(str(audio_path))

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

    prefix_ids = build_prefix_ids(tokenizer, initial_prompt)

    generated_segments: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, segment in enumerate(segments):
            generated = transcribe_segment_beam(
                audio=audio,
                segment=segment,
                seg_id=index,
                feature_extractor=feature_extractor,
                tokenizer=tokenizer,
                model=model,
                prefix_ids=prefix_ids,
                sample_rate=SAMPLE_RATE,
                device=device,
                dtype=dtype,
                torch_module=torch,
                num_beams=args.num_beams,
                max_new_tokens_cap=args.max_new_tokens_cap,
                new_token_buffer=args.new_token_buffer,
            )
            generated_segments.append(generated)
            if args.verbose:
                print(
                    f"[{index + 1}/{len(segments)}] "
                    f"{generated['start']:.3f}-{generated['end']:.3f}: {generated['text']}"
                )

    write_outputs(
        audio_path=audio_path,
        output_txt=output_txt,
        output_segments=output_segments,
        generated_segments=generated_segments,
    )

    print(f"[INFO] wrote transcript: {output_txt}")
    print(f"[INFO] wrote segments: {output_segments}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
