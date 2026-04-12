#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from nemo.collections.asr.models import ASRModel


class EncoderWrapper(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor):
        fn = getattr(self.encoder, "forward_for_export", self.encoder.forward)
        return fn(audio_signal=audio_signal, length=length)


class DecoderWrapper(nn.Module):
    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        targets: torch.Tensor,
        target_length: torch.Tensor,
        state_h: torch.Tensor,
        state_c: torch.Tensor,
    ):
        outputs, prednet_lengths, (state_h_out, state_c_out) = self.decoder(
            targets=targets,
            target_length=target_length,
            states=(state_h, state_c),
        )
        return outputs, prednet_lengths, state_h_out, state_c_out


class JointEncoderWrapper(nn.Module):
    def __init__(self, joint: nn.Module):
        super().__init__()
        self.joint = joint

    def forward(self, encoder_outputs: torch.Tensor):
        return self.joint.project_encoder(encoder_outputs)


class JointPredictorWrapper(nn.Module):
    def __init__(self, joint: nn.Module):
        super().__init__()
        self.joint = joint

    def forward(self, decoder_outputs: torch.Tensor):
        return self.joint.project_prednet(decoder_outputs)


class JointNetWrapper(nn.Module):
    def __init__(self, joint: nn.Module):
        super().__init__()
        self.joint = joint

    def forward(
        self,
        joint_encoder_outputs: torch.Tensor,
        joint_predictor_outputs: torch.Tensor,
    ):
        return self.joint.joint_after_projection(
            joint_encoder_outputs, joint_predictor_outputs
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-name", help="NeMo/Hugging Face pretrained model name")
    group.add_argument("--model-path", help="Path to a local .nemo checkpoint")
    parser.add_argument("--output-dir", required=True, help="Directory for exported assets")
    parser.add_argument(
        "--device",
        default="auto",
        help="Export device: auto, cpu, cuda, or cuda:N",
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--encoder-frames",
        type=int,
        default=256,
        help="Example encoder time dimension for export tracing",
    )
    return parser.parse_args()


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def move_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, (list, tuple)):
        moved = [move_to_device(item, device) for item in value]
        return type(value)(moved)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def prepare_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False
    prepare = getattr(module, "_prepare_for_export", None)
    if callable(prepare):
        prepare()


def load_model(args: argparse.Namespace, device: torch.device):
    if args.model_name:
        model = ASRModel.from_pretrained(model_name=args.model_name, map_location=device)
        source = args.model_name
    else:
        model = ASRModel.restore_from(args.model_path, map_location=device)
        source = args.model_path

    model.eval()
    if hasattr(model, "freeze"):
        model.freeze()
    model = model.to(device)

    for attr in ("encoder", "decoder", "joint"):
        if not hasattr(model, attr):
            raise RuntimeError(f"Loaded model is missing required `{attr}` module")

    return model, source


def export_onnx(
    module: nn.Module,
    path: Path,
    inputs: tuple,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    opset: int,
) -> None:
    with torch.inference_mode(), torch.no_grad():
        torch.onnx.export(
            module,
            inputs,
            str(path),
            opset_version=opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )


def extract_vocab(model) -> list[str]:
    vocabulary = getattr(getattr(model, "joint", None), "vocabulary", None)
    if vocabulary:
        return list(vocabulary)

    tokenizer = getattr(model, "tokenizer", None)
    tokenizer_impl = getattr(tokenizer, "tokenizer", tokenizer)
    if tokenizer_impl is not None:
        if hasattr(tokenizer_impl, "vocab_size") and hasattr(tokenizer_impl, "id_to_piece"):
            return [tokenizer_impl.id_to_piece(i) for i in range(tokenizer_impl.vocab_size())]
        if hasattr(tokenizer_impl, "get_vocab"):
            vocab = tokenizer_impl.get_vocab()
            return [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]

    raise RuntimeError("Could not extract vocabulary from model")


def write_vocab_files(output_dir: Path, vocab: Iterable[str]) -> int:
    labels = list(vocab)
    (output_dir / "vocab.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    tokens = labels + ["<blk>"]
    (output_dir / "tokens.txt").write_text("\n".join(tokens) + "\n", encoding="utf-8")
    return len(labels)


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, source = load_model(args, device)

    prepare_module(model.encoder)
    prepare_module(model.decoder)
    prepare_module(model.joint)

    encoder = EncoderWrapper(model.encoder).to(device).eval()
    decoder = DecoderWrapper(model.decoder).to(device).eval()
    joint_enc = JointEncoderWrapper(model.joint).to(device).eval()
    joint_pred = JointPredictorWrapper(model.joint).to(device).eval()
    joint_net = JointNetWrapper(model.joint).to(device).eval()

    encoder_inputs = move_to_device(
        tuple(model.encoder.input_example(max_batch=1, max_dim=args.encoder_frames)), device
    )
    decoder_inputs = move_to_device(tuple(model.decoder.input_example(max_batch=1, max_dim=1)), device)
    targets, target_length, states = decoder_inputs
    if len(states) != 2:
        raise RuntimeError(f"Expected 2 decoder states, found {len(states)}")
    state_h, state_c = states

    with torch.inference_mode(), torch.no_grad():
        encoder_outputs, encoded_lengths = encoder(*encoder_inputs)
        decoder_outputs, prednet_lengths, state_h_out, state_c_out = decoder(
            targets, target_length, state_h, state_c
        )
        joint_encoder_inputs = encoder_outputs.transpose(1, 2).contiguous()
        joint_predictor_inputs = decoder_outputs.transpose(1, 2).contiguous()
        joint_encoder_outputs = joint_enc(joint_encoder_inputs)
        joint_predictor_outputs = joint_pred(joint_predictor_inputs)
        _ = joint_net(joint_encoder_outputs, joint_predictor_outputs)

    export_onnx(
        encoder,
        output_dir / "encoder.onnx",
        encoder_inputs,
        ["audio_signal", "length"],
        ["outputs", "encoded_lengths"],
        {
            "audio_signal": {0: "batch", 2: "time"},
            "length": {0: "batch"},
            "outputs": {0: "batch", 2: "time_encoded"},
            "encoded_lengths": {0: "batch"},
        },
        args.opset,
    )
    export_onnx(
        decoder,
        output_dir / "decoder.onnx",
        (targets, target_length, state_h, state_c),
        ["targets", "target_length", "state_h", "state_c"],
        ["decoder_outputs", "prednet_lengths", "state_h_out", "state_c_out"],
        {
            "targets": {0: "batch", 1: "target_steps"},
            "target_length": {0: "batch"},
            "state_h": {1: "batch"},
            "state_c": {1: "batch"},
            "decoder_outputs": {0: "batch", 2: "target_steps"},
            "prednet_lengths": {0: "batch"},
            "state_h_out": {1: "batch"},
            "state_c_out": {1: "batch"},
        },
        args.opset,
    )
    export_onnx(
        joint_enc,
        output_dir / "joint.enc.onnx",
        (joint_encoder_inputs,),
        ["encoder_outputs"],
        ["joint_encoder_outputs"],
        {
            "encoder_outputs": {0: "batch", 1: "time_encoded"},
            "joint_encoder_outputs": {0: "batch", 1: "time_encoded"},
        },
        args.opset,
    )
    export_onnx(
        joint_pred,
        output_dir / "joint.pred.onnx",
        (joint_predictor_inputs,),
        ["decoder_outputs"],
        ["joint_predictor_outputs"],
        {
            "decoder_outputs": {0: "batch", 1: "target_steps"},
            "joint_predictor_outputs": {0: "batch", 1: "target_steps"},
        },
        args.opset,
    )
    export_onnx(
        joint_net,
        output_dir / "joint.joint_net.onnx",
        (joint_encoder_outputs, joint_predictor_outputs),
        ["joint_encoder_outputs", "joint_predictor_outputs"],
        ["logits"],
        {
            "joint_encoder_outputs": {0: "batch", 1: "time_encoded"},
            "joint_predictor_outputs": {0: "batch", 1: "target_steps"},
            "logits": {0: "batch", 1: "time_encoded", 2: "target_steps"},
        },
        args.opset,
    )

    blank_idx = write_vocab_files(output_dir, extract_vocab(model))
    metadata = {
        "format_version": 1,
        "source": source,
        "device": str(device),
        "decoder_type": type(model.decoder).__name__,
        "encoder_type": type(model.encoder).__name__,
        "joint_type": type(model.joint).__name__,
        "blank_idx": blank_idx,
        "pred_hidden": getattr(model.decoder, "pred_hidden", None),
        "pred_rnn_layers": getattr(model.decoder, "pred_rnn_layers", None),
        "joint_hidden": getattr(model.joint, "joint_hidden", None),
        "num_classes_with_blank": getattr(model.joint, "num_classes_with_blank", blank_idx + 1),
        "files": [
            "encoder.onnx",
            "decoder.onnx",
            "joint.enc.onnx",
            "joint.pred.onnx",
            "joint.joint_net.onnx",
            "tokens.txt",
            "vocab.txt",
        ],
    }
    (output_dir / "export.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
