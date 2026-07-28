from pathlib import Path
from typing import Any


def save_processor_pretrained(processor: Any, output_dir: Path) -> None:
    """Save custom processors with Transformers releases that expect an audio tokenizer."""
    if not hasattr(processor, "audio_tokenizer"):
        processor.audio_tokenizer = None
    processor.save_pretrained(output_dir)
