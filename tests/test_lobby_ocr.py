import asyncio
import os
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from lobby_automation import LobbyAutomation
from lobby_ocr import (
    DEFAULT_LOBBY_OCR_MODEL_ID,
    LOBBY_OCR_PROMPT,
    LobbyOCRError,
    DeepSeekOCRv2,
    extract_text_and_positions,
    grounding_to_readtext,
    load_deepseek_ocr_model,
    lobby_ocr_settings,
    select_ocr_load_plan,
    validate_brawler_read,
    validate_screen_read,
)
from state_finder import find_popup_close, get_state, is_underdog


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


def _screen(screen, menu_text=None, underdog=False):
    match_result = screen[4:] if screen.startswith("end_") else None
    star_drop = screen[len("star_drop_"):] if screen.startswith("star_drop_") else None
    return {
        "screen": screen,
        "menu_text": menu_text or [],
        "brawler_name": None,
        "match_result": match_result,
        "star_drop": star_drop,
        "underdog": underdog,
        "source": "deepseek-ocr-2",
    }


class MenuStateClientTests(unittest.TestCase):
    def test_schema_accepts_screen_and_rejects_unknown(self):
        payload = validate_screen_read(_screen("end_victory", [{
            "text": "victory",
            "bbox": [0, 0, 10, 4],
            "confidence": 1.0,
        }]))
        self.assertEqual(payload["match_result"], "victory")
        brawler = validate_brawler_read({
            "brawler_name": "shelly",
            "center": [4, 8],
            "menu_text": payload["menu_text"],
            "source": "deepseek-ocr-2",
        })
        self.assertEqual(brawler["center"], [4, 8])
        broken = _screen("combat")
        with self.assertRaises(LobbyOCRError):
            validate_screen_read(broken)

    def test_async_read_screen_parses_grounding_into_schema(self):
        reader = DeepSeekOCRv2()
        reader._infer_sync = lambda image: '<|ref|>victory<|/ref|><|det|>[[0, 0, 999, 999]]<|/det|>'

        payload = asyncio.run(reader.read_screen(np.zeros((20, 40, 3), dtype=np.uint8)))

        self.assertEqual(payload["screen"], "end_victory")
        self.assertEqual(payload["match_result"], "victory")
        self.assertEqual(payload["source"], "deepseek-ocr-2")
        self.assertEqual(payload["menu_text"][0]["bbox"], [0, 0, 40, 20])
        self.assertFalse(payload["underdog"])

    def test_async_read_brawler_returns_center(self):
        reader = DeepSeekOCRv2()
        reader._infer_sync = lambda image: '<|ref|>shelly<|/ref|><|det|>[[0, 0, 499, 499]]<|/det|>'

        payload = asyncio.run(reader.read_brawler(np.zeros((10, 20, 3), dtype=np.uint8), {"shelly"}))

        self.assertEqual(payload["brawler_name"], "shelly")
        self.assertEqual(payload["center"], [4, 2])

    def test_nowait_does_not_block_and_later_returns_schema(self):
        reader = DeepSeekOCRv2(timeout_seconds=2, result_max_age=3)

        def slow(image):
            time.sleep(0.2)
            return '<|ref|>victory<|/ref|><|det|>[[0, 0, 999, 999]]<|/det|>'

        reader._infer_sync = slow
        started = time.perf_counter()
        first = reader.read_screen_nowait(np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertIsNone(first)
        self.assertLess(time.perf_counter() - started, 0.15)
        deadline = time.perf_counter() + 2
        second = None
        while time.perf_counter() < deadline:
            second = reader.read_screen_nowait(np.zeros((8, 8, 3), dtype=np.uint8))
            if second is not None:
                break
            time.sleep(0.05)
        self.assertIsNotNone(second)
        self.assertEqual(second["screen"], "end_victory")

    def test_blocking_read_times_out_without_raising_past_the_client(self):
        reader = DeepSeekOCRv2(timeout_seconds=0.05)

        def slow(image):
            time.sleep(0.4)
            return '<|ref|>shelly<|/ref|><|det|>[[0, 0, 10, 10]]<|/det|>'

        reader._infer_sync = slow
        with self.assertRaises(LobbyOCRError):
            reader.read_brawler_sync(np.zeros((8, 8, 3), dtype=np.uint8), {"shelly"})

    def test_state_uses_ocr_and_falls_back_to_templates(self):
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        client = MagicMock()
        client.read_screen_nowait.return_value = _screen("shop")
        with patch("state_finder.get_in_game_state") as fallback:
            self.assertEqual(get_state(image, reader=client), "shop")
        fallback.assert_not_called()

        client.read_screen_nowait.return_value = None
        with patch("state_finder.get_in_game_state", return_value="lobby") as fallback:
            self.assertEqual(get_state(image, reader=client), "lobby")
        fallback.assert_called_once()

        client.read_screen_nowait.return_value = _screen("unknown")
        with patch("state_finder.get_in_game_state", return_value="popup") as fallback:
            self.assertEqual(get_state(image, reader=client), "popup")
        fallback.assert_called_once()

        client.read_screen_nowait.side_effect = RuntimeError("down")
        with patch("state_finder.get_in_game_state", side_effect=RuntimeError("templates")):
            self.assertEqual(get_state(image, reader=client), "match")

    def test_underdog_and_popup_prefer_ocr_then_template(self):
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        client = MagicMock()
        client.read_screen_nowait.return_value = _screen("end_defeat", underdog=True)
        with patch("state_finder.is_underdog_template") as template:
            self.assertTrue(is_underdog(image, reader=client))
        template.assert_not_called()

        client.read_screen_nowait.return_value = None
        with patch("state_finder.is_underdog_template", return_value=True) as template:
            self.assertTrue(is_underdog(image, reader=client))
        template.assert_called_once()

        client.read_screen_nowait.return_value = _screen("popup", [{
            "text": "close",
            "bbox": [10, 20, 30, 40],
            "confidence": 1,
        }])
        self.assertEqual(find_popup_close(image, reader=client), (20, 30))
        client.read_screen_nowait.return_value = None
        self.assertIsNone(find_popup_close(image, reader=client))

    def test_brawler_selection_survives_ocr_failure(self):
        window = MagicMock()
        window.width_ratio = 1
        window.height_ratio = 1
        window.screenshot.return_value = np.zeros((40, 40, 3), dtype=np.uint8)
        reader = DeepSeekOCRv2()
        reader.read_brawler_sync = MagicMock(side_effect=LobbyOCRError("unavailable"))
        lobby = LobbyAutomation(window, ocr_reader=reader)
        with patch("lobby_automation.time.sleep", return_value=None):
            result = lobby.select_brawler("shelly", lambda: "brawler_selection")
        self.assertEqual(result, "error")


class _FakeCuda:
    def __init__(self, available=True, bf16=True):
        self._available = available
        self._bf16 = bf16

    def is_available(self):
        return self._available

    def is_bf16_supported(self):
        return self._bf16


class _FakeTorch:
    def __init__(self, hip=None, available=True, bf16=True):
        self.version = type("Version", (), {"hip": hip})()
        self.cuda = _FakeCuda(available, bf16)
        self.bfloat16 = "bfloat16"
        self.float16 = "float16"


class _FakeWeights:
    def __init__(self):
        self.dtype = None
        self.moved_to_cuda = False

    def eval(self):
        return self

    def cuda(self):
        self.moved_to_cuda = True
        return self

    def to(self, dtype):
        self.dtype = dtype
        return self


class GpuLoadPlanTests(unittest.TestCase):
    def test_nvidia_selects_flash_attention_and_bfloat16(self):
        plan = select_ocr_load_plan(_FakeTorch(hip=None, bf16=False))
        self.assertEqual(plan["backend"], "nvidia")
        self.assertEqual(plan["attention"][0], "flash_attention_2")
        self.assertEqual(plan["dtype"], "bfloat16")
        self.assertTrue(plan["device_available"])

    def test_amd_rocm_skips_flash_attention(self):
        plan = select_ocr_load_plan(_FakeTorch(hip="6.2.41133", bf16=True))
        self.assertEqual(plan["backend"], "rocm")
        self.assertEqual(plan["attention"], ["sdpa", "eager"])
        self.assertNotIn("flash_attention_2", plan["attention"])
        self.assertEqual(plan["dtype"], "bfloat16")

    def test_amd_without_bf16_selects_float16(self):
        plan = select_ocr_load_plan(_FakeTorch(hip="6.2.41133", bf16=False))
        self.assertEqual(plan["dtype"], "float16")
        self.assertNotIn("flash_attention_2", plan["attention"])

    def test_amd_sdpa_rejection_retries_eager(self):
        seen = []

        def from_pretrained(model_id, _attn_implementation, trust_remote_code, use_safetensors):
            seen.append(_attn_implementation)
            self.assertTrue(trust_remote_code)
            self.assertTrue(use_safetensors)
            self.assertEqual(model_id, DEFAULT_LOBBY_OCR_MODEL_ID)
            if _attn_implementation == "sdpa":
                raise KeyError("mha_sdpa")
            return _FakeWeights()

        model, plan = load_deepseek_ocr_model(
            DEFAULT_LOBBY_OCR_MODEL_ID,
            _FakeTorch(hip="6.2.41133", bf16=False),
            from_pretrained,
        )
        self.assertEqual(seen, ["sdpa", "eager"])
        self.assertNotIn("flash_attention_2", seen)
        self.assertEqual(plan["attention_used"], "eager")
        self.assertTrue(model.moved_to_cuda)
        self.assertEqual(model.dtype, "float16")

    def test_missing_gpu_raises_for_template_fallback(self):
        with self.assertRaises(LobbyOCRError):
            load_deepseek_ocr_model("deepseek-ai/DeepSeek-OCR-2", _FakeTorch(available=False), lambda *args, **kwargs: None)


if __name__ == "__main__":
    unittest.main()
