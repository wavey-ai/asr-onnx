import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))

from processor_compat import save_processor_pretrained


class RecordingProcessor:
    def __init__(self) -> None:
        self.saved_output = None

    def save_pretrained(self, output_dir: Path) -> None:
        self.saved_output = output_dir


class ProcessorCompatTest(unittest.TestCase):
    def test_adds_missing_audio_tokenizer_before_save(self) -> None:
        processor = RecordingProcessor()
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            save_processor_pretrained(processor, output_dir)

        self.assertIsNone(processor.audio_tokenizer)
        self.assertEqual(processor.saved_output, output_dir)

    def test_preserves_existing_audio_tokenizer(self) -> None:
        processor = RecordingProcessor()
        processor.audio_tokenizer = object()
        expected_audio_tokenizer = processor.audio_tokenizer
        with tempfile.TemporaryDirectory() as directory:
            save_processor_pretrained(processor, Path(directory))

        self.assertIs(processor.audio_tokenizer, expected_audio_tokenizer)


if __name__ == "__main__":
    unittest.main()
