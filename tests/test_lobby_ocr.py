import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from lobby_automation import LobbyAutomation
from lobby_ocr import (
    DEFAULT_LOBBY_OCR_MODEL_ID,
    LOBBY_OCR_PROMPT,
    DeepSeekOCRv2,
    extract_text_and_positions,
    grounding_to_readtext,
    lobby_ocr_settings,
)


ROOT = Path(__file__).resolve().parents[1]


class LobbyOcrTests(unittest.TestCase):
    def test_grounding_boxes_use_deepseek_999_scale(self):
        text = '<|ref|>shelly<|/ref|><|det|>[[0, 0, 999, 499]]<|/det|>'
        rows = grounding_to_readtext(text, width=200, height=100)
        self.assertEqual(len(rows), 1)
        bbox, label, confidence = rows[0]
        self.assertEqual(label, "shelly")
        self.assertEqual(confidence, 1.0)
        self.assertEqual(bbox, [[0, 0], [200, 0], [200, 49], [0, 49]])

    def test_extract_text_and_positions_uses_lobby_reader(self):
        reader = MagicMock()
        reader.readtext.return_value = [
            ([[0, 0], [10, 0], [10, 4], [0, 4]], "Shelly", 0.9),
        ]
        details = extract_text_and_positions(np.zeros((4, 10, 3), dtype=np.uint8), reader)
        self.assertIn("shelly", details)
        self.assertEqual(details["shelly"]["center"], (5, 2))
        reader.readtext.assert_called_once()

    def test_readtext_calls_deepseek_ocr_v2_infer(self):
        reader = DeepSeekOCRv2(model_id=DEFAULT_LOBBY_OCR_MODEL_ID)
        model = MagicMock()
        model.infer.return_value = '<|ref|>shelly<|/ref|><|det|>[[0, 0, 999, 999]]<|/det|>'
        reader._model = model
        reader._tokenizer = object()

        rows = reader.readtext(np.zeros((10, 20, 3), dtype=np.uint8))

        self.assertEqual(reader.model_id, "deepseek-ai/DeepSeek-OCR-2")
        args, kwargs = model.infer.call_args
        self.assertIs(args[0], reader._tokenizer)
        self.assertEqual(kwargs["prompt"], LOBBY_OCR_PROMPT)
        self.assertEqual(kwargs["base_size"], 1024)
        self.assertEqual(kwargs["image_size"], 768)
        self.assertTrue(kwargs["crop_mode"])
        self.assertFalse(kwargs["save_results"])
        self.assertTrue(kwargs["eval_mode"])
        self.assertTrue(kwargs["image_file"].endswith("lobby.png"))
        self.assertEqual(rows[0][1], "shelly")
        self.assertEqual(rows[0][0][2], [20, 10])

    def test_env_overrides_lobby_model_id_only(self):
        previous = os.environ.get("DEEPSEEK_OCR_MODEL")
        os.environ["DEEPSEEK_OCR_MODEL"] = "local/deepseek-ocr-2"
        try:
            settings = lobby_ocr_settings()
            reader = DeepSeekOCRv2.from_lobby_config()
        finally:
            if previous is None:
                os.environ.pop("DEEPSEEK_OCR_MODEL", None)
            else:
                os.environ["DEEPSEEK_OCR_MODEL"] = previous
        self.assertEqual(settings["model_id"], "local/deepseek-ocr-2")
        self.assertEqual(reader.model_id, "local/deepseek-ocr-2")
        self.assertEqual(reader.prompt, LOBBY_OCR_PROMPT)
        self.assertIsNone(reader._model)

    def test_lobby_selection_reads_with_deepseek_not_easyocr(self):
        window = MagicMock()
        window.width_ratio = 1
        window.height_ratio = 1
        window.screenshot.return_value = np.zeros((100, 120, 3), dtype=np.uint8)
        reader = MagicMock()
        reader.model_id = DEFAULT_LOBBY_OCR_MODEL_ID
        reader.readtext.return_value = [
            ([[10, 60], [50, 60], [50, 100], [10, 100]], "shelly", 1.0),
        ]
        lobby = LobbyAutomation(window, ocr_reader=reader)
        lobby.ocr_scale_down_factor = 1
        lobby.ocr_scale_up_factor = 1

        with patch("lobby_automation.time.sleep", return_value=None):
            result = lobby.select_brawler("shelly", lambda: "brawler_selection")

        self.assertEqual(result, "success")
        reader.readtext.assert_called()
        centers = []
        for call in window.click.call_args_list:
            centers.append((call.args[0], call.args[1]))
        self.assertIn((30, 30), centers)
        lobby_source = (ROOT / "lobby_automation.py").read_text(encoding="utf-8")
        ocr_source = (ROOT / "lobby_ocr.py").read_text(encoding="utf-8")
        self.assertNotIn("easyocr", lobby_source.lower())
        self.assertNotIn("easyocr", ocr_source.lower())
        self.assertIn("DeepSeekOCRv2", lobby_source)

    def test_in_game_model_is_unchanged(self):
        main_source = (ROOT / "main.py").read_text(encoding="utf-8")
        play_source = (ROOT / "play.py").read_text(encoding="utf-8")
        detect_source = (ROOT / "detect.py").read_text(encoding="utf-8")
        self.assertIn("mainInGameModel.onnx", main_source)
        self.assertIn("Detect(main_info_model", play_source)
        for source in (main_source, play_source, detect_source):
            lowered = source.lower()
            self.assertNotIn("deepseek", lowered)
            self.assertNotIn("easyocr", lowered)
            self.assertNotIn("lobby_ocr", lowered)


if __name__ == "__main__":
    unittest.main()
