import ast
import asyncio
import os
import re
import tempfile
import threading
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path

import numpy as np
from PIL import Image

from utils import load_toml_as_dict, normalize_brawler_filename

# Shared menu and state OCR client. The in-game detector stays on
# models/mainInGameModel.onnx and is not invoked here.
DEFAULT_LOBBY_OCR_MODEL_ID = "deepseek-ai/DeepSeek-OCR-2"
# Official grounding prompt. model.infer returns text only when the prompt
# contains a space and eval_mode=True. DeepSeek-OCR-2 does not emit JSON;
# this client parses the grounding tags into the schemas below.
LOBBY_OCR_PROMPT = "<image>\n<|grounding|>Convert the document to markdown. "
LOBBY_OCR_BASE_SIZE = 1024
LOBBY_OCR_IMAGE_SIZE = 768
LOBBY_OCR_CROP_MODE = True
LOBBY_OCR_TIMEOUT_SECONDS = 8.0
LOBBY_OCR_RESULT_MAX_AGE = 3.0
# DeepSeek-OCR-2 det boxes are normalized to this range.
COORD_NORM = 999
OCR_SOURCE = "deepseek-ocr-2"

SCREEN_NAMES = {
    "lobby",
    "brawler_selection",
    "shop",
    "popup",
    "match_making",
    "match",
    "prestige_milestone",
    "trophy_reward",
    "star_drop_regular",
    "star_drop_angelic",
    "star_drop_demonic",
    "star_drop_starr_nova",
    "end_victory",
    "end_defeat",
    "end_draw",
    "end_trio_showdown_0",
    "end_trio_showdown_1",
    "end_trio_showdown_2",
    "end_trio_showdown_3",
    "unknown",
}
MATCH_RESULTS = {
    "victory",
    "defeat",
    "draw",
    "trio_showdown_0",
    "trio_showdown_1",
    "trio_showdown_2",
    "trio_showdown_3",
}
STAR_DROPS = {"regular", "angelic", "demonic", "starr_nova"}

_REF_DET = re.compile(
    r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>",
    re.DOTALL,
)
_CLIENT = None
_CLIENT_LOCK = threading.Lock()


class LobbyOCRError(RuntimeError):
    pass


def _positive_int(value, default):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _positive_float(value, default):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _bool(value, default):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def lobby_ocr_settings():
    section = load_toml_as_dict("cfg/lobby_config.toml").get("ocr", {})
    model_id = os.environ.get("DEEPSEEK_OCR_MODEL", "").strip() or section.get("model_id") or DEFAULT_LOBBY_OCR_MODEL_ID
    timeout = os.environ.get("DEEPSEEK_OCR_TIMEOUT", "").strip() or section.get("timeout_seconds", LOBBY_OCR_TIMEOUT_SECONDS)
    max_age = os.environ.get("DEEPSEEK_OCR_RESULT_MAX_AGE", "").strip() or section.get("result_max_age", LOBBY_OCR_RESULT_MAX_AGE)
    try:
        scale = float(section.get("scale_down_factor", 0.8))
    except (TypeError, ValueError):
        scale = 0.8
    scale = max(0.5, min(1.0, scale))
    return {
        "model_id": str(model_id),
        "prompt": LOBBY_OCR_PROMPT,
        "base_size": _positive_int(section.get("base_size"), LOBBY_OCR_BASE_SIZE),
        "image_size": _positive_int(section.get("image_size"), LOBBY_OCR_IMAGE_SIZE),
        "crop_mode": _bool(section.get("crop_mode"), LOBBY_OCR_CROP_MODE),
        "scale_down_factor": scale,
        "timeout_seconds": _positive_float(timeout, LOBBY_OCR_TIMEOUT_SECONDS),
        "result_max_age": _positive_float(max_age, LOBBY_OCR_RESULT_MAX_AGE),
    }


def get_lobby_ocr_client():
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = DeepSeekOCRv2.from_lobby_config()
        return _CLIENT


def grounding_to_readtext(text, width, height):
    """Turn DeepSeek grounding tags into (bbox, text, confidence) rows."""
    width = max(int(width), 1)
    height = max(int(height), 1)
    rows = []
    for label, det in _REF_DET.findall(text or ""):
        label = label.strip()
        if not label or label.lower() == "image":
            continue
        try:
            boxes = ast.literal_eval(det)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(boxes, list) or not boxes:
            continue
        if isinstance(boxes[0], (int, float)):
            boxes = [boxes]
        for box in boxes:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            x1, y1, x2, y2 = [float(value) for value in box]
            x1 = int(x1 / COORD_NORM * width)
            y1 = int(y1 / COORD_NORM * height)
            x2 = int(x2 / COORD_NORM * width)
            y2 = int(y2 / COORD_NORM * height)
            bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            rows.append((bbox, label, 1.0))
    return rows


def menu_text_from_grounding(text, width, height):
    items = []
    for bbox, label, confidence in grounding_to_readtext(text, width, height):
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        items.append({
            "text": label,
            "bbox": [min(xs), min(ys), max(xs), max(ys)],
            "confidence": float(confidence),
        })
    return items


def classify_menu_text(menu_text):
    joined = " ".join(str(item.get("text", "")) for item in menu_text).lower()
    if not joined.strip():
        return "unknown"
    if "1st" in joined or "1 st" in joined:
        return "end_trio_showdown_0"
    if "2nd" in joined or "2 nd" in joined:
        return "end_trio_showdown_1"
    if "3rd" in joined or "3 rd" in joined:
        return "end_trio_showdown_2"
    if "4th" in joined or "4 th" in joined:
        return "end_trio_showdown_3"
    if "victory" in joined:
        return "end_victory"
    if "defeat" in joined:
        return "end_defeat"
    if re.search(r"\bdraw\b", joined):
        return "end_draw"
    if "matchmaking" in joined or "searching" in joined:
        return "match_making"
    if "star road" in joined or "brawl pass" in joined:
        return "shop"
    if "prestige" in joined:
        return "prestige_milestone"
    if "angelic" in joined:
        return "star_drop_angelic"
    if "demonic" in joined:
        return "star_drop_demonic"
    if "starr nova" in joined or "starrnova" in joined:
        return "star_drop_starr_nova"
    if "star drop" in joined:
        return "star_drop_regular"
    if "trophy" in joined and ("claim" in joined or "reward" in joined):
        return "trophy_reward"
    if "search" in joined and "brawler" in joined:
        return "brawler_selection"
    if "power point" in joined or re.search(r"\bshop\b", joined):
        return "shop"
    if "special offer" in joined or re.search(r"\boffer\b", joined):
        return "popup"
    return "unknown"


def _screen_details(screen):
    match_result = screen[4:] if screen.startswith("end_") else None
    star_drop = screen[len("star_drop_"):] if screen.startswith("star_drop_") else None
    return match_result, star_drop


def screen_from_grounding(text, width, height):
    menu_text = menu_text_from_grounding(text, width, height)
    screen = classify_menu_text(menu_text)
    match_result, star_drop = _screen_details(screen)
    joined = " ".join(item["text"] for item in menu_text).lower()
    return {
        "screen": screen,
        "menu_text": menu_text,
        "brawler_name": None,
        "match_result": match_result,
        "star_drop": star_drop,
        "underdog": "underdog" in joined,
        "source": OCR_SOURCE,
    }


def validate_menu_text(items):
    if not isinstance(items, list):
        raise LobbyOCRError("menu_text must be a list")
    for item in items:
        if not isinstance(item, dict):
            raise LobbyOCRError("menu_text items must be objects")
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise LobbyOCRError("menu_text text must be a non-empty string")
        bbox = item.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(value, int) for value in bbox)
        ):
            raise LobbyOCRError("menu_text bbox must be four integers")
        confidence = item.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise LobbyOCRError("menu_text confidence must be a number")
        if not 0 <= float(confidence) <= 1:
            raise LobbyOCRError("menu_text confidence must be between 0 and 1")
    return items


def validate_screen_read(payload):
    if not isinstance(payload, dict):
        raise LobbyOCRError("screen read must be an object")
    screen = payload.get("screen")
    if screen not in SCREEN_NAMES:
        raise LobbyOCRError(f"Unknown screen {screen!r}")
    validate_menu_text(payload.get("menu_text"))
    for key in ("brawler_name", "match_result", "star_drop"):
        value = payload.get(key)
        if value is not None and not isinstance(value, str):
            raise LobbyOCRError(f"{key} must be a string or null")
    if payload.get("match_result") is not None and payload["match_result"] not in MATCH_RESULTS:
        raise LobbyOCRError("match_result is not a known end-screen value")
    if payload.get("star_drop") is not None and payload["star_drop"] not in STAR_DROPS:
        raise LobbyOCRError("star_drop is not a known drop")
    if not isinstance(payload.get("underdog"), bool):
        raise LobbyOCRError("underdog must be a boolean")
    if payload.get("source") != OCR_SOURCE:
        raise LobbyOCRError("screen read source must be the DeepSeek OCR client")
    return payload


def validate_brawler_read(payload):
    if not isinstance(payload, dict):
        raise LobbyOCRError("brawler read must be an object")
    name = payload.get("brawler_name")
    if name is not None and not isinstance(name, str):
        raise LobbyOCRError("brawler_name must be a string or null")
    center = payload.get("center")
    if center is not None and (
        not isinstance(center, list)
        or len(center) != 2
        or not all(isinstance(value, int) for value in center)
    ):
        raise LobbyOCRError("center must be two integers or null")
    validate_menu_text(payload.get("menu_text"))
    if payload.get("source") != OCR_SOURCE:
        raise LobbyOCRError("brawler read source must be the DeepSeek OCR client")
    return payload


def label_center(document, labels):
    if not isinstance(document, dict):
        return None
    wanted = {str(label).strip().lower() for label in labels}
    for item in document.get("menu_text") or []:
        if str(item.get("text", "")).strip().lower() not in wanted:
            continue
        bbox = item.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
    return None


def extract_text_and_positions(image, reader):
    results = reader.readtext(image)
    text_details = {}
    for bbox, text, _prob in results:
        top_left, top_right, bottom_right, bottom_left = bbox
        center = (
            (top_left[0] + top_right[0] + bottom_right[0] + bottom_left[0]) / 4,
            (top_left[1] + top_right[1] + bottom_right[1] + bottom_left[1]) / 4,
        )
        text_details[text.lower()] = {
            "top_left": top_left,
            "top_right": top_right,
            "bottom_right": bottom_right,
            "bottom_left": bottom_left,
            "center": center,
        }
    return text_details


class DeepSeekOCRv2:
    """Async DeepSeek-OCR-2 client for menu text, brawler names, and screen state."""

    def __init__(self, model_id=None, prompt=None, base_size=None, image_size=None, crop_mode=None, timeout_seconds=None, result_max_age=None):
        settings = lobby_ocr_settings()
        self.model_id = model_id or settings["model_id"]
        self.prompt = prompt or settings["prompt"]
        self.base_size = base_size if base_size is not None else settings["base_size"]
        self.image_size = image_size if image_size is not None else settings["image_size"]
        self.crop_mode = settings["crop_mode"] if crop_mode is None else crop_mode
        self.timeout_seconds = settings["timeout_seconds"] if timeout_seconds is None else timeout_seconds
        self.result_max_age = settings["result_max_age"] if result_max_age is None else result_max_age
        self._model = None
        self._tokenizer = None
        self._load_failed = False
        self._lock = threading.Lock()
        self._loop = None
        self._loop_lock = threading.Lock()
        self._flight = threading.Lock()
        self._result_lock = threading.Lock()
        self._pending = None
        self._latest = None
        self._latest_at = 0.0

    @classmethod
    def from_lobby_config(cls):
        settings = lobby_ocr_settings()
        return cls(
            model_id=settings["model_id"],
            prompt=settings["prompt"],
            base_size=settings["base_size"],
            image_size=settings["image_size"],
            crop_mode=settings["crop_mode"],
            timeout_seconds=settings["timeout_seconds"],
            result_max_age=settings["result_max_age"],
        )

    def _ensure_background_loop(self):
        if self._loop is not None and self._loop.is_running():
            return self._loop
        with self._loop_lock:
            if self._loop is not None and self._loop.is_running():
                return self._loop
            ready = threading.Event()

            def runner():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                ready.set()
                loop.run_forever()

            thread = threading.Thread(target=runner, name="deepseek-ocr", daemon=True)
            thread.start()
            if not ready.wait(5):
                raise LobbyOCRError("DeepSeek OCR v2 async loop did not start")
            return self._loop

    def _run_sync(self, coro_factory):
        loop = self._ensure_background_loop()
        future = asyncio.run_coroutine_threadsafe(coro_factory(), loop)
        try:
            return future.result(timeout=self.timeout_seconds + 1)
        except FuturesTimeoutError as exc:
            raise LobbyOCRError("DeepSeek OCR v2 timed out") from exc
        except LobbyOCRError:
            raise
        except Exception as exc:
            raise LobbyOCRError(f"DeepSeek OCR v2 failed: {exc}") from exc

    def _ensure_loaded(self):
        if self._load_failed:
            raise LobbyOCRError(f"DeepSeek OCR v2 previously failed to load {self.model_id}")
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        with self._lock:
            if self._load_failed:
                raise LobbyOCRError(f"DeepSeek OCR v2 previously failed to load {self.model_id}")
            if self._model is None or self._tokenizer is None:
                try:
                    import torch
                    from transformers import AutoModel, AutoTokenizer
                except ImportError as exc:
                    self._load_failed = True
                    raise LobbyOCRError(
                        "DeepSeek OCR v2 requires torch and transformers. "
                        f"Could not import them for {self.model_id}: {exc}"
                    ) from exc
                try:
                    tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
                    model = AutoModel.from_pretrained(
                        self.model_id,
                        _attn_implementation="flash_attention_2",
                        trust_remote_code=True,
                        use_safetensors=True,
                    )
                    model = model.eval().cuda().to(torch.bfloat16)
                except Exception as exc:
                    self._load_failed = True
                    raise LobbyOCRError(
                        f"DeepSeek OCR v2 failed to load {self.model_id}: {exc}"
                    ) from exc
                self._tokenizer = tokenizer
                self._model = model
        return self._model, self._tokenizer

    @staticmethod
    def _as_image(image_input):
        if isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            array = image_input
            if array.ndim == 2:
                image = Image.fromarray(array).convert("RGB")
            elif array.ndim == 3:
                image = Image.fromarray(array[:, :, :3]).convert("RGB")
            else:
                raise LobbyOCRError(f"Unsupported lobby image shape: {array.shape}")
        else:
            raise LobbyOCRError(f"Unsupported lobby image type: {type(image_input).__name__}")
        return image

    def _infer_sync(self, image_input):
        model, tokenizer = self._ensure_loaded()
        image = self._as_image(image_input)
        with tempfile.TemporaryDirectory(prefix="lobby-ocr-") as temp_dir:
            image_file = os.path.join(temp_dir, "lobby.png")
            output_path = os.path.join(temp_dir, "out")
            os.makedirs(output_path, exist_ok=True)
            image.save(image_file)
            result = model.infer(
                tokenizer,
                prompt=self.prompt,
                image_file=image_file,
                output_path=output_path,
                base_size=self.base_size,
                image_size=self.image_size,
                crop_mode=self.crop_mode,
                save_results=False,
                eval_mode=True,
            )
        if not isinstance(result, str) or not result.strip():
            raise LobbyOCRError("DeepSeek OCR v2 model.infer did not return text.")
        return result

    async def infer_grounding(self, image_input):
        return await asyncio.wait_for(
            asyncio.to_thread(self._infer_sync, image_input),
            timeout=self.timeout_seconds,
        )

    async def read_screen(self, image_input):
        image = self._as_image(image_input)
        raw = await self.infer_grounding(image)
        return validate_screen_read(screen_from_grounding(raw, image.size[0], image.size[1]))

    async def read_brawler(self, image_input, names):
        document = await self.read_screen(image_input)
        wanted = {normalize_brawler_filename(name) for name in names}
        matched_name = None
        center = None
        for item in document["menu_text"]:
            key = normalize_brawler_filename(item["text"])
            if key not in wanted:
                continue
            matched_name = key
            x1, y1, x2, y2 = item["bbox"]
            center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
            break
        payload = {
            "brawler_name": matched_name,
            "center": center,
            "menu_text": document["menu_text"],
            "source": OCR_SOURCE,
        }
        return validate_brawler_read(payload)

    def readtext(self, image_input):
        image = self._as_image(image_input)
        raw = self._run_sync(lambda: self.infer_grounding(image))
        return grounding_to_readtext(raw, image.size[0], image.size[1])

    def read_brawler_sync(self, image_input, names):
        return self._run_sync(lambda: self.read_brawler(image_input, names))

    def read_screen_nowait(self, image_input):
        """Start an async screen read without blocking the caller.

        Returns the latest fresh schema, or None when the model has not
        finished, failed, or the previous result is too old. Callers then
        use the template-matching fallback.
        """
        if self._load_failed:
            return None
        now = time.time()
        with self._result_lock:
            latest = self._latest
            latest_at = self._latest_at
            pending = self._pending
        fresh = latest if latest is not None and (now - latest_at) <= self.result_max_age else None
        if pending is not None and not pending.done():
            return fresh
        if not self._flight.acquire(blocking=False):
            return fresh
        try:
            loop = self._ensure_background_loop()
            snapshot = self._as_image(image_input).copy()
            future = asyncio.run_coroutine_threadsafe(self.read_screen(snapshot), loop)
        except Exception as exc:
            self._flight.release()
            print(f"DeepSeek OCR v2 screen read failed, using template fallback: {exc}")
            return fresh

        with self._result_lock:
            self._pending = future

        def _done(fut):
            try:
                result = fut.result()
            except Exception as exc:
                print(f"DeepSeek OCR v2 screen read failed, using template fallback: {exc}")
                result = None
            with self._result_lock:
                if isinstance(result, dict):
                    self._latest = result
                    self._latest_at = time.time()
                self._pending = None
            self._flight.release()

        future.add_done_callback(_done)
        return fresh
