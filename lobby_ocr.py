import ast
import os
import re
import tempfile
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from utils import load_toml_as_dict

# Lobby-only OCR. The in-game detector stays on models/mainInGameModel.onnx.
DEFAULT_LOBBY_OCR_MODEL_ID = "deepseek-ai/DeepSeek-OCR-2"
# Official grounding prompt. model.infer returns text only when the prompt
# contains a space and eval_mode=True.
LOBBY_OCR_PROMPT = "<image>\n<|grounding|>Convert the document to markdown. "
LOBBY_OCR_BASE_SIZE = 1024
LOBBY_OCR_IMAGE_SIZE = 768
LOBBY_OCR_CROP_MODE = True
# DeepSeek-OCR-2 det boxes are normalized to this range.
COORD_NORM = 999

_REF_DET = re.compile(
    r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>",
    re.DOTALL,
)


class LobbyOCRError(RuntimeError):
    pass


def _positive_int(value, default):
    try:
        parsed = int(value)
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
    }


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
    """Lobby OCR reader. Calls deepseek-ai/DeepSeek-OCR-2 through model.infer."""

    def __init__(self, model_id=None, prompt=None, base_size=None, image_size=None, crop_mode=None):
        settings = lobby_ocr_settings()
        self.model_id = model_id or settings["model_id"]
        self.prompt = prompt or settings["prompt"]
        self.base_size = base_size if base_size is not None else settings["base_size"]
        self.image_size = image_size if image_size is not None else settings["image_size"]
        self.crop_mode = settings["crop_mode"] if crop_mode is None else crop_mode
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    @classmethod
    def from_lobby_config(cls):
        settings = lobby_ocr_settings()
        return cls(
            model_id=settings["model_id"],
            prompt=settings["prompt"],
            base_size=settings["base_size"],
            image_size=settings["image_size"],
            crop_mode=settings["crop_mode"],
        )

    def _ensure_loaded(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        with self._lock:
            if self._model is None or self._tokenizer is None:
                try:
                    import torch
                    from transformers import AutoModel, AutoTokenizer
                except ImportError as exc:
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

    def readtext(self, image_input):
        model, tokenizer = self._ensure_loaded()
        image = self._as_image(image_input)
        width, height = image.size
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
        if not isinstance(result, str):
            raise LobbyOCRError("DeepSeek OCR v2 model.infer did not return text.")
        return grounding_to_readtext(result, width, height)
