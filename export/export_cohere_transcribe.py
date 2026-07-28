#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
import torch.nn as nn
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.convert_slow_tokenizer import SpmConverter
from transformers.modeling_outputs import BaseModelOutput

from processor_compat import save_processor_pretrained


class CohereTokenizerConverter(SpmConverter):
    handle_byte_fallback = True

    def pre_tokenizer(self, replacement: str, add_prefix_space: bool) -> Any:
        return super().pre_tokenizer(replacement, True)

    def decoder(self, replacement: str, add_prefix_space: bool) -> Any:
        return super().decoder(replacement, True)


def save_runtime_tokenizer(processor: Any, output_dir: Path) -> None:
    tokenizer_model = output_dir / "tokenizer.model"
    if not tokenizer_model.is_file():
        raise FileNotFoundError(f"processor export did not create {tokenizer_model}")

    tokenizer = processor.tokenizer
    had_vocab_file = hasattr(tokenizer, "vocab_file")
    previous_vocab_file = getattr(tokenizer, "vocab_file", None)
    tokenizer.vocab_file = str(tokenizer_model)
    try:
        runtime_tokenizer = CohereTokenizerConverter(tokenizer).converted()
    finally:
        if had_vocab_file:
            tokenizer.vocab_file = previous_vocab_file
        else:
            del tokenizer.vocab_file

    sample_text = "Audio café — hello world."
    sample_ids = tokenizer.encode(sample_text, add_special_tokens=False)
    expected_text = tokenizer.decode(sample_ids, skip_special_tokens=True)
    actual_text = runtime_tokenizer.decode(sample_ids, skip_special_tokens=True)
    if actual_text != expected_text:
        raise ValueError(
            "the exported runtime tokenizer does not match the source tokenizer"
        )
    runtime_tokenizer.save(str(output_dir / "tokenizer.json"))


def normalize_preprocessor_config(output_dir: Path) -> None:
    config_path = output_dir / "preprocessor_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    filterbank = config.get("_fb_config", {})
    fields = {
        "dither": "dither",
        "feature_size": "nfilt",
        "n_fft": "n_fft",
        "n_window_size": "n_window_size",
        "n_window_stride": "n_window_stride",
        "normalize": "normalize",
        "padding_value": "pad_value",
        "sampling_rate": "sample_rate",
        "window": "window",
    }
    missing = []
    for output_name, filterbank_name in fields.items():
        value = config.get(output_name, filterbank.get(filterbank_name))
        if value is None:
            missing.append(output_name)
        else:
            config[output_name] = value
    if missing:
        raise ValueError(
            "preprocessor export is missing runtime fields: "
            + ", ".join(sorted(missing))
        )
    config_path.write_text(
        f"{json.dumps(config, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _cache_layer_tensors(cache: DynamicCache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    return cache.key_cache[layer_idx], cache.value_cache[layer_idx]


def _dynamic_cache_from_tensors(keys: list[torch.Tensor], values: list[torch.Tensor]) -> DynamicCache:
    if hasattr(DynamicCache, "from_legacy_cache"):
        legacy_cache = tuple((key, value) for key, value in zip(keys, values))
        return DynamicCache.from_legacy_cache(legacy_cache)

    cache = DynamicCache()
    cache.key_cache = list(keys)
    cache.value_cache = list(values)
    return cache


class CohereDecoderOnlyWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        length: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.model(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
            length=length,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return out.logits[:, -1, :]


class CohereDecoderPrefillWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.decoder_config = model.config.transf_decoder["config_dict"]
        self.num_layers = int(self.decoder_config["num_layers"])

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        length: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if self.model.encoder_decoder_proj is not None:
            encoder_hidden_states = self.model.encoder_decoder_proj(encoder_hidden_states)

        dtype = encoder_hidden_states.dtype
        batch_size, target_steps = decoder_input_ids.shape
        positions = torch.arange(target_steps, device=decoder_input_ids.device).unsqueeze(0).expand(batch_size, -1)

        query_positions = torch.arange(target_steps, device=decoder_input_ids.device)[:, None]
        key_positions = torch.arange(target_steps, device=decoder_input_ids.device)[None, :]
        causal_bool = key_positions > query_positions
        self_attention_mask = torch.zeros(
            (batch_size, 1, target_steps, target_steps),
            device=decoder_input_ids.device,
            dtype=dtype,
        )
        self_attention_mask.masked_fill_(causal_bool[None, None, :, :], float("-inf"))
        decoder_mask = align_decoder_attention_mask(decoder_attention_mask, total_kv_len=target_steps)
        key_padding = (1.0 - decoder_mask[:, None, None, :].to(dtype=dtype)) * -1e9
        self_attention_mask = self_attention_mask + key_padding

        encoder_lengths = self.model._infer_encoder_lengths_from_raw(length)
        source_steps = encoder_hidden_states.shape[1]
        enc_positions = torch.arange(source_steps, device=encoder_hidden_states.device)[None, :]
        valid = enc_positions < encoder_lengths.to(device=encoder_hidden_states.device)[:, None]
        cross_attention_mask = (1.0 - valid[:, None, None, :].to(dtype=dtype)) * -1e9

        cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
        outputs, updated_cache = self.model.transf_decoder(
            input_ids=decoder_input_ids,
            positions=positions,
            encoder_hidden_states=encoder_hidden_states,
            self_attention_mask=self_attention_mask,
            cross_attention_mask=cross_attention_mask,
            past_key_values=cache,
            cache_position=None,
            kv_seq_len=None,
        )
        logits = self.model.log_softmax(outputs)[:, -1, :]

        flat_outputs: list[torch.Tensor] = [logits]
        for layer_idx in range(self.num_layers):
            self_key, self_value = _cache_layer_tensors(updated_cache.self_attention_cache, layer_idx)
            cross_key, cross_value = _cache_layer_tensors(updated_cache.cross_attention_cache, layer_idx)
            flat_outputs.extend(
                [
                    self_key,
                    self_value,
                    cross_key,
                    cross_value,
                ]
            )
        return tuple(flat_outputs)


class CohereDecoderCachedStepWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.decoder_config = model.config.transf_decoder["config_dict"]
        self.num_layers = int(self.decoder_config["num_layers"])

    def _build_cache(
        self,
        self_keys: list[torch.Tensor],
        self_values: list[torch.Tensor],
        cross_keys: list[torch.Tensor],
        cross_values: list[torch.Tensor],
    ) -> EncoderDecoderCache:
        self_cache = _dynamic_cache_from_tensors(self_keys, self_values)
        cross_cache = _dynamic_cache_from_tensors(cross_keys, cross_values)

        cache = EncoderDecoderCache(self_cache, cross_cache)
        for layer_idx in range(self.num_layers):
            cache.is_updated[layer_idx] = True
        return cache

    def forward(
        self,
        encoded_length: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        *flat_cache_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        cache_inputs = list(flat_cache_inputs)
        self_keys: list[torch.Tensor] = []
        self_values: list[torch.Tensor] = []
        cross_keys: list[torch.Tensor] = []
        cross_values: list[torch.Tensor] = []
        for layer_idx in range(self.num_layers):
            base = layer_idx * 4
            self_keys.append(cache_inputs[base + 0])
            self_values.append(cache_inputs[base + 1])
            cross_keys.append(cache_inputs[base + 2])
            cross_values.append(cache_inputs[base + 3])

        batch_size, target_steps = decoder_input_ids.shape
        past_steps = self_keys[0].shape[-2] if self_keys else 0
        positions = torch.arange(
            past_steps,
            past_steps + target_steps,
            device=decoder_input_ids.device,
        ).unsqueeze(0).expand(batch_size, -1)
        hidden_states = self.model.transf_decoder._embedding(decoder_input_ids, positions)
        dtype = hidden_states.dtype

        total_kv_len = past_steps + target_steps
        query_positions = torch.arange(past_steps, past_steps + target_steps, device=decoder_input_ids.device)[:, None]
        key_positions = torch.arange(total_kv_len, device=decoder_input_ids.device)[None, :]
        causal_bool = key_positions > query_positions
        self_attention_mask = torch.zeros(
            (batch_size, 1, target_steps, total_kv_len),
            device=decoder_input_ids.device,
            dtype=dtype,
        )
        self_attention_mask.masked_fill_(causal_bool[None, None, :, :], float("-inf"))

        source_steps = cross_keys[0].shape[-2] if cross_keys else 0
        enc_positions = torch.arange(source_steps, device=decoder_input_ids.device)[None, :]
        valid = enc_positions < encoded_length.to(device=decoder_input_ids.device)[:, None]
        cross_attention_mask = (1.0 - valid[:, None, None, :].to(dtype=dtype)) * -1e9

        cache = self._build_cache(self_keys, self_values, cross_keys, cross_values)
        outputs, updated_cache = self.model.transf_decoder._decoder(
            hidden_states,
            encoder_hidden_states=None,
            self_attention_mask=self_attention_mask,
            cross_attention_mask=cross_attention_mask,
            past_key_values=cache,
            cache_position=None,
            kv_seq_len=None,
        )
        logits = self.model.log_softmax(outputs)[:, -1, :]

        flat_outputs: list[torch.Tensor] = [logits]
        for layer_idx in range(self.num_layers):
            self_key, self_value = _cache_layer_tensors(updated_cache.self_attention_cache, layer_idx)
            flat_outputs.extend(
                [
                    self_key,
                    self_value,
                ]
            )
        return tuple(flat_outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="CohereLabs/cohere-transcribe-03-2026",
        help="Model name or local Hugging Face snapshot path",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for exported assets")
    parser.add_argument("--device", default="auto", help="Export device: auto, cpu, cuda, or cuda:N")
    parser.add_argument("--language", default="en", help="Prompt language for example inputs")
    parser.add_argument("--sample-audio-seconds", type=float, default=15.0)
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def cache_tensor_names(num_layers: int, include_cross: bool) -> list[str]:
    names: list[str] = []
    for layer_idx in range(num_layers):
        names.append(f"self_key_{layer_idx}")
        names.append(f"self_value_{layer_idx}")
        if include_cross:
            names.append(f"cross_key_{layer_idx}")
            names.append(f"cross_value_{layer_idx}")
    return names


def prefill_output_names(num_layers: int) -> list[str]:
    return ["last_token_logits", *cache_tensor_names(num_layers=num_layers, include_cross=True)]


def cached_step_input_names(num_layers: int) -> list[str]:
    return ["encoded_length", "decoder_input_ids", *cache_tensor_names(num_layers=num_layers, include_cross=True)]


def cached_step_output_names(num_layers: int) -> list[str]:
    names = ["last_token_logits"]
    for layer_idx in range(num_layers):
        names.append(f"self_key_out_{layer_idx}")
        names.append(f"self_value_out_{layer_idx}")
    return names


def align_decoder_attention_mask(decoder_attention_mask: torch.Tensor, total_kv_len: int) -> torch.Tensor:
    current_len = int(decoder_attention_mask.shape[-1])
    if current_len < total_kv_len:
        pad = torch.ones(
            (decoder_attention_mask.shape[0], total_kv_len - current_len),
            device=decoder_attention_mask.device,
            dtype=decoder_attention_mask.dtype,
        )
        return torch.cat([decoder_attention_mask, pad], dim=-1)
    if current_len > total_kv_len:
        return decoder_attention_mask[:, -total_kv_len:]
    return decoder_attention_mask


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


def export_onnx(
    module: nn.Module,
    path: Path,
    inputs: tuple[Any, ...],
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    opset: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=path.parent,
        prefix=f".{path.stem}-",
    ) as temporary_directory:
        temporary_path = Path(temporary_directory) / path.name
        with torch.inference_mode():
            torch.onnx.export(
                module,
                inputs,
                str(temporary_path),
                opset_version=opset,
                dynamo=False,
                export_params=True,
                external_data=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

        exported_model = onnx.load_model(temporary_path, load_external_data=True)
        data_path = path.with_name(f"{path.name}.data")
        path.unlink(missing_ok=True)
        data_path.unlink(missing_ok=True)
        onnx.save_model(
            exported_model,
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_path.name,
            size_threshold=0,
        )


def export_encoder_onnx(
    encoder: nn.Module,
    out_path: Path,
    input_features: torch.Tensor,
    length: torch.Tensor,
    opset: int,
) -> None:
    export_onnx(
        encoder.eval(),
        out_path,
        (input_features, length),
        ["input_features", "length"],
        ["encoder_hidden_states", "encoded_length"],
        {
            "input_features": {0: "batch", 2: "frames"},
            "length": {0: "batch"},
            "encoder_hidden_states": {0: "batch", 1: "encoded_frames"},
            "encoded_length": {0: "batch"},
        },
        opset,
    )


def export_decoder_onnx(
    model: nn.Module,
    out_path: Path,
    encoder_hidden_states: torch.Tensor,
    length: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    decoder_attention_mask: torch.Tensor,
    opset: int,
) -> None:
    export_onnx(
        CohereDecoderOnlyWrapper(model).eval(),
        out_path,
        (encoder_hidden_states, length, decoder_input_ids, decoder_attention_mask),
        [
            "encoder_hidden_states",
            "length",
            "decoder_input_ids",
            "decoder_attention_mask",
        ],
        ["last_token_logits"],
        {
            "encoder_hidden_states": {0: "batch", 1: "encoded_frames"},
            "length": {0: "batch"},
            "decoder_input_ids": {0: "batch", 1: "target_steps"},
            "decoder_attention_mask": {0: "batch", 1: "target_steps"},
            "last_token_logits": {0: "batch"},
        },
        opset,
    )


def export_decoder_prefill_onnx(
    model: nn.Module,
    out_path: Path,
    encoder_hidden_states: torch.Tensor,
    length: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    decoder_attention_mask: torch.Tensor,
    opset: int,
) -> None:
    num_layers = int(model.config.transf_decoder["config_dict"]["num_layers"])
    dynamic_axes: dict[str, dict[int, str]] = {
        "encoder_hidden_states": {0: "batch", 1: "encoded_frames"},
        "length": {0: "batch"},
        "decoder_input_ids": {0: "batch", 1: "target_steps"},
        "decoder_attention_mask": {0: "batch", 1: "target_steps"},
        "last_token_logits": {0: "batch"},
    }
    for layer_idx in range(num_layers):
        dynamic_axes[f"self_key_{layer_idx}"] = {0: "batch", 2: "self_steps"}
        dynamic_axes[f"self_value_{layer_idx}"] = {0: "batch", 2: "self_steps"}
        dynamic_axes[f"cross_key_{layer_idx}"] = {0: "batch", 2: "encoded_frames"}
        dynamic_axes[f"cross_value_{layer_idx}"] = {0: "batch", 2: "encoded_frames"}

    export_onnx(
        CohereDecoderPrefillWrapper(model).eval(),
        out_path,
        (encoder_hidden_states, length, decoder_input_ids, decoder_attention_mask),
        [
            "encoder_hidden_states",
            "length",
            "decoder_input_ids",
            "decoder_attention_mask",
        ],
        prefill_output_names(num_layers),
        dynamic_axes,
        opset,
    )


def export_decoder_cached_step_onnx(
    model: nn.Module,
    out_path: Path,
    encoded_length: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    self_keys: list[torch.Tensor],
    self_values: list[torch.Tensor],
    cross_keys: list[torch.Tensor],
    cross_values: list[torch.Tensor],
    opset: int,
) -> None:
    num_layers = int(model.config.transf_decoder["config_dict"]["num_layers"])
    dynamic_axes: dict[str, dict[int, str]] = {
        "encoded_length": {0: "batch"},
        "decoder_input_ids": {0: "batch", 1: "target_steps"},
        "last_token_logits": {0: "batch"},
    }
    for layer_idx in range(num_layers):
        dynamic_axes[f"self_key_{layer_idx}"] = {0: "batch", 2: "self_steps"}
        dynamic_axes[f"self_value_{layer_idx}"] = {0: "batch", 2: "self_steps"}
        dynamic_axes[f"cross_key_{layer_idx}"] = {0: "batch", 2: "encoded_frames"}
        dynamic_axes[f"cross_value_{layer_idx}"] = {0: "batch", 2: "encoded_frames"}
        dynamic_axes[f"self_key_out_{layer_idx}"] = {0: "batch", 2: "self_steps_out"}
        dynamic_axes[f"self_value_out_{layer_idx}"] = {0: "batch", 2: "self_steps_out"}

    flat_inputs: list[torch.Tensor] = [encoded_length, decoder_input_ids]
    for layer_idx in range(num_layers):
        flat_inputs.extend(
            [
                self_keys[layer_idx],
                self_values[layer_idx],
                cross_keys[layer_idx],
                cross_values[layer_idx],
            ]
        )

    export_onnx(
        CohereDecoderCachedStepWrapper(model).eval(),
        out_path,
        tuple(flat_inputs),
        cached_step_input_names(num_layers),
        cached_step_output_names(num_layers),
        dynamic_axes,
        opset,
    )


def prepare_sample_inputs(
    model: Any,
    processor: Any,
    language: str,
    seconds: float,
    device: torch.device,
) -> tuple[str, dict[str, torch.Tensor]]:
    sample_rate = int(processor.feature_extractor.sampling_rate)
    waveform = np.zeros((max(1, int(round(sample_rate * seconds))),), dtype=np.float32)
    prompt_text = model.build_prompt(language=language, punctuation=True)
    sample = processor(
        audio=[waveform],
        text=[prompt_text],
        sampling_rate=sample_rate,
        return_tensors="pt",
    )
    sample = sanitize_inputs(processor.tokenizer, sample)
    return prompt_text, {name: value.to(device) for name, value in sample.items()}


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.source, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.source, trust_remote_code=True).to(device).eval()
    save_processor_pretrained(processor, output_dir)
    normalize_preprocessor_config(output_dir)
    model.config.save_pretrained(output_dir)
    model.generation_config.save_pretrained(output_dir)
    save_runtime_tokenizer(processor, output_dir)
    prompt_text, sample = prepare_sample_inputs(
        model=model,
        processor=processor,
        language=args.language,
        seconds=args.sample_audio_seconds,
        device=device,
    )

    with torch.inference_mode():
        encoder_hidden_states, encoded_length = model.get_encoder()(sample["input_features"], sample["length"])
        prefill_outputs = list(
            CohereDecoderPrefillWrapper(model).eval()(
                encoder_hidden_states,
                sample["length"],
                sample["decoder_input_ids"],
                sample["decoder_attention_mask"],
            )
        )

    decoder_num_layers = int(model.config.transf_decoder["config_dict"]["num_layers"])
    prefill_logits = prefill_outputs[0]
    self_keys: list[torch.Tensor] = []
    self_values: list[torch.Tensor] = []
    cross_keys: list[torch.Tensor] = []
    cross_values: list[torch.Tensor] = []
    for layer_idx in range(decoder_num_layers):
        base = 1 + (layer_idx * 4)
        self_keys.append(prefill_outputs[base + 0])
        self_values.append(prefill_outputs[base + 1])
        cross_keys.append(prefill_outputs[base + 2])
        cross_values.append(prefill_outputs[base + 3])
    next_token_ids = torch.argmax(prefill_logits, dim=-1, keepdim=True)

    export_encoder_onnx(
        encoder=model.get_encoder(),
        out_path=output_dir / "encoder.onnx",
        input_features=sample["input_features"],
        length=sample["length"],
        opset=args.opset,
    )
    export_decoder_onnx(
        model=model,
        out_path=output_dir / "decoder_last_token.onnx",
        encoder_hidden_states=encoder_hidden_states,
        length=sample["length"],
        decoder_input_ids=sample["decoder_input_ids"],
        decoder_attention_mask=sample["decoder_attention_mask"],
        opset=args.opset,
    )
    export_decoder_prefill_onnx(
        model=model,
        out_path=output_dir / "decoder_prefill.onnx",
        encoder_hidden_states=encoder_hidden_states,
        length=sample["length"],
        decoder_input_ids=sample["decoder_input_ids"],
        decoder_attention_mask=sample["decoder_attention_mask"],
        opset=args.opset,
    )
    export_decoder_cached_step_onnx(
        model=model,
        out_path=output_dir / "decoder_cached_step.onnx",
        encoded_length=encoded_length,
        decoder_input_ids=next_token_ids,
        self_keys=self_keys,
        self_values=self_values,
        cross_keys=cross_keys,
        cross_values=cross_values,
        opset=args.opset,
    )

    files = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file()
    )
    metadata = {
        "format_version": 1,
        "model_family": "cohere-transcribe-seq2seq",
        "source": args.source,
        "device": str(device),
        "language": args.language,
        "prompt_text": prompt_text,
        "prompt_token_ids": sample["decoder_input_ids"][0].detach().cpu().tolist(),
        "bos_token_id": processor.tokenizer.bos_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "decoder_start_token_id": getattr(model.config, "decoder_start_token_id", None),
        "sample_audio_seconds": args.sample_audio_seconds,
        "opset": args.opset,
        "decoder_num_layers": decoder_num_layers,
        "sample_rate": int(processor.feature_extractor.sampling_rate),
        "files": files,
    }
    (output_dir / "export.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
