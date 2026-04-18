#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


DEFAULT_SOURCE = "CohereLabs/cohere-transcribe-03-2026"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Cohere frontend metadata and best-effort waveform frontend assets."
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Model name or local Hugging Face snapshot path.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for exported frontend assets.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Load device for Cohere model metadata: auto, cpu, cuda, or cuda:N.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Prompt language used for sample prompt metadata.",
    )
    parser.add_argument(
        "--sample-audio-seconds",
        type=float,
        default=15.0,
        help="Sample waveform length used for probe/export.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset for best-effort frontend export.",
    )
    parser.add_argument(
        "--skip-onnx-export",
        action="store_true",
        help="Only save metadata and probe outputs; do not attempt frontend.onnx export.",
    )
    return parser.parse_args()


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def sanitize_inputs(tokenizer: Any, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = dict(inputs)
    pad_token_id = tokenizer.pad_token_id
    if "input_ids" in out and "decoder_input_ids" not in out:
        out["decoder_input_ids"] = out.pop("input_ids")
    if "decoder_input_ids" in out and "decoder_attention_mask" not in out:
        if pad_token_id is None:
            out["decoder_attention_mask"] = torch.ones(
                out["decoder_input_ids"].shape,
                dtype=torch.long,
            )
        else:
            out["decoder_attention_mask"] = out["decoder_input_ids"].ne(pad_token_id).long()
    return out


def build_prompt_text(model: Any, language: str) -> str:
    build_prompt = getattr(model, "build_prompt", None)
    if callable(build_prompt):
        return str(build_prompt(language=language, punctuation=True))
    return ""


def probe_processor(
    processor: Any,
    tokenizer: Any,
    prompt_text: str,
    sample_rate: int,
    seconds: float,
) -> dict[str, Any]:
    waveform = np.zeros((max(1, int(round(sample_rate * seconds))),), dtype=np.float32)
    kwargs: dict[str, Any] = {
        "audio": [waveform],
        "sampling_rate": sample_rate,
        "return_tensors": "pt",
    }
    if prompt_text:
        kwargs["text"] = [prompt_text]

    try:
        inputs = processor(**kwargs)
        used_text = bool(prompt_text)
    except TypeError:
        kwargs.pop("text", None)
        inputs = processor(**kwargs)
        used_text = False

    inputs = sanitize_inputs(tokenizer, inputs)
    return {
        "used_text_prompt": used_text,
        "input_keys": sorted(inputs.keys()),
        "input_shapes": {
            key: list(value.shape) for key, value in inputs.items() if hasattr(value, "shape")
        },
        "prompt_token_ids": (
            inputs["decoder_input_ids"][0].detach().cpu().tolist()
            if "decoder_input_ids" in inputs
            else []
        ),
    }


def dotted_getattr(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        value = getattr(value, part)
    return value


def detect_frontend_candidate(processor: Any, model: Any) -> tuple[str | None, Any, list[dict[str, Any]]]:
    candidates = [
        ("processor.feature_extractor", processor),
        ("processor.audio_processor", processor),
        ("processor.feature_extractor.feature_extractor", processor),
        ("model.preprocessor", model),
        ("model.feature_extractor", model),
        ("model.audio_processor", model),
    ]
    inspected: list[dict[str, Any]] = []
    for path, root in candidates:
        try:
            candidate = dotted_getattr(root, path.split(".", 1)[1])
        except Exception as error:
            inspected.append({"path": path, "error": str(error)})
            continue
        inspected.append(
            {
                "path": path,
                "type": type(candidate).__name__,
                "module": isinstance(candidate, nn.Module),
                "callable": callable(candidate),
            }
        )
        if isinstance(candidate, nn.Module) or callable(candidate):
            return path, candidate, inspected
    return None, None, inspected


def call_frontend(frontend: Any, audio: torch.Tensor, audio_len: torch.Tensor) -> Any:
    attempts = [
        lambda: frontend(input_signal=audio, length=audio_len),
        lambda: frontend(audio_signal=audio, length=audio_len),
        lambda: frontend(audio=audio, audio_len=audio_len),
        lambda: frontend(waveform=audio, length=audio_len),
        lambda: frontend(input_values=audio, length=audio_len),
        lambda: frontend(audio, audio_len),
        lambda: frontend(audio),
    ]
    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except Exception as error:
            last_error = error
    raise RuntimeError(f"no supported frontend call signature: {last_error}")


def normalize_frontend_outputs(outputs: Any, audio_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    features: torch.Tensor | None = None
    feature_len: torch.Tensor | None = None

    if isinstance(outputs, dict):
        candidate = outputs.get("input_features") or outputs.get("features")
        if isinstance(candidate, torch.Tensor):
            features = candidate
        candidate_len = (
            outputs.get("length")
            or outputs.get("feature_lengths")
            or outputs.get("input_lengths")
        )
        if isinstance(candidate_len, torch.Tensor):
            feature_len = candidate_len
    elif isinstance(outputs, (list, tuple)):
        if outputs and isinstance(outputs[0], torch.Tensor):
            features = outputs[0]
        if len(outputs) > 1 and isinstance(outputs[1], torch.Tensor):
            feature_len = outputs[1]
    elif isinstance(outputs, torch.Tensor):
        features = outputs

    if features is None:
        raise RuntimeError(f"unsupported frontend output type {type(outputs).__name__}")

    if feature_len is None:
        batch = int(features.shape[0]) if features.ndim > 0 else 1
        frames = int(features.shape[-1]) if features.ndim > 0 else 0
        feature_len = torch.full(
            (batch,),
            frames,
            dtype=torch.int64,
            device=features.device,
        )

    return features, feature_len.to(dtype=torch.int64)


class CohereFrontendWrapper(nn.Module):
    def __init__(self, frontend: Any):
        super().__init__()
        self.frontend = frontend

    def forward(self, audio: torch.Tensor, audio_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = call_frontend(self.frontend, audio, audio_len)
        return normalize_frontend_outputs(outputs, audio_len)


def export_frontend_onnx(
    frontend: Any,
    out_path: Path,
    sample_rate: int,
    sample_audio_seconds: float,
    device: torch.device,
    opset: int,
) -> dict[str, Any]:
    wrapper = CohereFrontendWrapper(frontend).eval()
    if isinstance(frontend, nn.Module):
        wrapper = wrapper.to(device)

    audio = torch.zeros(
        (1, max(1, int(round(sample_rate * sample_audio_seconds)))),
        dtype=torch.float32,
        device=device,
    )
    audio_len = torch.tensor([audio.shape[1]], dtype=torch.int64, device=device)

    with torch.inference_mode():
        features, feature_len = wrapper(audio, audio_len)
        dynamic_axes = {
            "audio": {0: "batch", 1: "samples"},
            "audio_len": {0: "batch"},
            "feature_len": {0: "batch"},
        }
        if features.ndim >= 3:
            dynamic_axes["input_features"] = {0: "batch", features.ndim - 1: "frames"}
        elif features.ndim >= 2:
            dynamic_axes["input_features"] = {0: "batch", 1: "frames"}
        else:
            dynamic_axes["input_features"] = {0: "batch"}

        torch.onnx.export(
            wrapper,
            (audio, audio_len),
            str(out_path),
            opset_version=opset,
            export_params=True,
            input_names=["audio", "audio_len"],
            output_names=["input_features", "feature_len"],
            dynamic_axes=dynamic_axes,
        )

    return {
        "feature_shape": list(features.shape),
        "feature_len_shape": list(feature_len.shape),
    }


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.source, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.source, trust_remote_code=True)
    model = model.to(device).eval()
    tokenizer = processor.tokenizer
    sample_rate = int(processor.feature_extractor.sampling_rate)
    prompt_text = build_prompt_text(model, args.language)

    processor.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)

    report: dict[str, Any] = {
        "format_version": 1,
        "source": args.source,
        "device": str(device),
        "language": args.language,
        "sample_rate": sample_rate,
        "sample_audio_seconds": args.sample_audio_seconds,
        "prompt_text": prompt_text,
        "tokenizer_class": type(tokenizer).__name__,
        "processor_class": type(processor).__name__,
        "feature_extractor_class": type(processor.feature_extractor).__name__,
    }
    report["processor_probe"] = probe_processor(
        processor=processor,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        sample_rate=sample_rate,
        seconds=args.sample_audio_seconds,
    )

    candidate_path, candidate, inspected = detect_frontend_candidate(processor, model)
    report["frontend_candidates"] = inspected
    report["frontend_candidate_path"] = candidate_path

    if candidate is None:
        report["frontend_export"] = {
            "status": "no_candidate",
            "message": "no callable frontend candidate was found; processor may remain Python-only",
        }
    elif args.skip_onnx_export:
        report["frontend_export"] = {
            "status": "skipped",
            "message": "frontend export skipped by flag",
        }
    else:
        out_path = output_dir / "frontend.onnx"
        try:
            export_info = export_frontend_onnx(
                frontend=candidate,
                out_path=out_path,
                sample_rate=sample_rate,
                sample_audio_seconds=args.sample_audio_seconds,
                device=device,
                opset=args.opset,
            )
            report["frontend_export"] = {
                "status": "ok",
                "path": out_path.name,
                **export_info,
            }
        except Exception as error:
            report["frontend_export"] = {
                "status": "error",
                "message": str(error),
            }

    (output_dir / "frontend_export.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
