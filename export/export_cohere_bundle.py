#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = "CohereLabs/cohere-transcribe-03-2026"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a unified Cohere bundle with ONNX graphs, metadata, frontend assets, and optional featurizer traces."
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Model name or local Hugging Face snapshot path.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the staged Cohere bundle.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for the ONNX export scripts: auto, cpu, cuda, or cuda:N.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Prompt language used for export probes.",
    )
    parser.add_argument(
        "--sample-audio-seconds",
        type=float,
        default=15.0,
        help="Dummy waveform length used for export probes and tracing.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset for Cohere exports.",
    )
    parser.add_argument(
        "--num-devices",
        type=int,
        default=0,
        help="Number of CUDA devices to trace for the optional featurizer step.",
    )
    parser.add_argument(
        "--skip-frontend",
        action="store_true",
        help="Skip frontend metadata / frontend.onnx export.",
    )
    parser.add_argument(
        "--skip-featurizer",
        action="store_true",
        help="Skip best-effort TorchScript featurizer tracing.",
    )
    parser.add_argument(
        "--skip-sha256sums",
        action="store_true",
        help="Skip writing SHA256SUMS after staging the bundle.",
    )
    parser.add_argument(
        "--asr-torch-dir",
        default=None,
        help="Optional path to the sibling asr-torch repo. Defaults to ../asr-torch relative to this script.",
    )
    return parser.parse_args()


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def asr_torch_dir(raw: str | None) -> Path:
    if raw:
        return Path(raw).resolve()
    return (script_dir().parent.parent / "../asr-torch").resolve()


def run_step(label: str, cmd: list[str]) -> dict[str, Any]:
    print(f"[cohere-bundle] {label}: {' '.join(cmd)}")
    completed = subprocess.run(cmd, check=False)
    return {
        "label": label,
        "command": cmd,
        "returncode": completed.returncode,
        "status": "ok" if completed.returncode == 0 else "error",
    }


def write_sha256sums(output_dir: Path) -> None:
    lines: list[str] = []
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file()):
        if path.name == "SHA256SUMS":
            continue
        rel = path.relative_to(output_dir).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {rel}")
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_dir = script_dir()
    torch_dir = asr_torch_dir(args.asr_torch_dir)
    trace_script = torch_dir / "trace_cohere_featurizer.py"
    if not trace_script.exists() and not args.skip_featurizer:
        raise SystemExit(f"missing trace script: {trace_script}")

    steps: list[dict[str, Any]] = []
    steps.append(
        run_step(
            "export-transcribe",
            [
                sys.executable,
                str(onnx_dir / "export_cohere_transcribe.py"),
                "--source",
                args.source,
                "--output-dir",
                str(output_dir),
                "--device",
                args.device,
                "--language",
                args.language,
                "--sample-audio-seconds",
                str(args.sample_audio_seconds),
                "--opset",
                str(args.opset),
            ],
        )
    )

    if not args.skip_frontend:
        steps.append(
            run_step(
                "export-frontend",
                [
                    sys.executable,
                    str(onnx_dir / "export_cohere_frontend.py"),
                    "--source",
                    args.source,
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    args.device,
                    "--language",
                    args.language,
                    "--sample-audio-seconds",
                    str(args.sample_audio_seconds),
                    "--opset",
                    str(args.opset),
                ],
            )
        )

    if not args.skip_featurizer:
        steps.append(
            run_step(
                "trace-featurizer",
                [
                    sys.executable,
                    str(trace_script),
                    "--source",
                    args.source,
                    "--output-dir",
                    str(output_dir),
                    "--num-devices",
                    str(args.num_devices),
                    "--sample-audio-seconds",
                    str(args.sample_audio_seconds),
                ],
            )
        )

    status = "ok" if all(step["status"] == "ok" for step in steps) else "error"
    summary = {
        "format_version": 1,
        "source": args.source,
        "output_dir": str(output_dir),
        "device": args.device,
        "language": args.language,
        "sample_audio_seconds": args.sample_audio_seconds,
        "opset": args.opset,
        "num_devices": args.num_devices,
        "steps": steps,
        "status": status,
    }
    (output_dir / "bundle_export.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not args.skip_sha256sums:
        write_sha256sums(output_dir)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
