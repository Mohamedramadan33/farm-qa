# -*- coding: utf-8 -*-
"""
inference_service.py
--------------------
Persistent FastAPI + Uvicorn YOLO tree detection microservice.
Loads PyTorch YOLO weights once during application lifespan and exposes:
  - GET  /health
  - POST /detect-batch
"""

from contextlib import asynccontextmanager
import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "detection.config.json"


def _load_detection_config() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "aiModelConfThresh": 0.55,
        "nmsIouThresh": 0.40,
        "classes": ["orchard_tree", "palm_tree", "windbreak_tree"],
    }
    try:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                defaults.update(data)
    except Exception:
        pass
    return defaults


DETECTION_CONFIG = _load_detection_config()
TREE_CLASSES: List[str] = list(
    DETECTION_CONFIG.get("classes", ["orchard_tree", "palm_tree", "windbreak_tree"])
)

_MODEL_STATE: Dict[str, Any] = {
    "model": None,
    "model_path": None,
    "device": "cpu",
    "load_error": None,
}


def _resolve_model_path() -> Optional[Path]:
    env_path = os.environ.get("YOLO_MODEL_PATH", "").strip()
    candidates: List[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([BASE_DIR / "models" / "best.pt", BASE_DIR / "best.pt"])

    for cand in candidates:
        try:
            resolved = cand.resolve()
            if not str(resolved).startswith(str(BASE_DIR) + os.sep) and resolved.parent != BASE_DIR:
                continue
            if resolved.exists() and resolved.is_file():
                return resolved
        except Exception:
            continue
    return None


def _select_device() -> Any:
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    model_path = _resolve_model_path()
    device = _select_device()
    _MODEL_STATE["device"] = str(device)
    _MODEL_STATE["model_path"] = str(model_path) if model_path else None

    if model_path is not None:
        try:
            from ultralytics import YOLO

            model = YOLO(str(model_path))
            _MODEL_STATE["model"] = model
            _MODEL_STATE["load_error"] = None
        except Exception as exc:
            _MODEL_STATE["model"] = None
            _MODEL_STATE["load_error"] = str(exc)
    yield
    _MODEL_STATE["model"] = None


app = FastAPI(title="Farm Tree YOLO Inference Service", version="1.0.0", lifespan=lifespan)


class TilePayload(BaseModel):
    imageBase64: str = Field(..., min_length=16, max_length=8_000_000)
    offsetX: float = 0.0
    offsetY: float = 0.0


class DetectBatchRequest(BaseModel):
    tiles: List[TilePayload] = Field(..., min_length=1, max_length=48)
    conf: float = Field(default=float(DETECTION_CONFIG.get("aiModelConfThresh", 0.55)), ge=0.05, le=0.95)
    iou: float = Field(default=float(DETECTION_CONFIG.get("nmsIouThresh", 0.40)), ge=0.10, le=0.90)


def _decode_base64_image(raw_b64: str) -> Image.Image:
    cleaned = raw_b64.split(",", 1)[1] if "," in raw_b64 and raw_b64.startswith("data:") else raw_b64
    raw_bytes = base64.b64decode(cleaned, validate=True)
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


def decode_batch_tiles(req: DetectBatchRequest) -> List[Dict[str, Any]]:
    decoded: List[Dict[str, Any]] = []
    for tile in req.tiles:
        img = _decode_base64_image(tile.imageBase64)
        decoded.append(
            {
                "image": img,
                "offsetX": float(tile.offsetX),
                "offsetY": float(tile.offsetY),
            }
        )
    return decoded


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model_loaded": _MODEL_STATE["model"] is not None,
        "model_path": os.path.basename(_MODEL_STATE["model_path"]) if _MODEL_STATE["model_path"] else None,
        "device": _MODEL_STATE["device"],
        "classes": TREE_CLASSES,
        "error": _MODEL_STATE["load_error"],
    }


@app.post("/detect-batch")
def detect_batch(req: DetectBatchRequest) -> Dict[str, Any]:
    model = _MODEL_STATE["model"]
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=_MODEL_STATE["load_error"] or "YOLO weights not loaded on inference service.",
        )

    try:
        items = decode_batch_tiles(req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid tile image payload: {exc}") from exc

    images = [item["image"] for item in items]
    device = _select_device()
    results = model.predict(
        source=images,
        conf=float(req.conf),
        iou=float(req.iou),
        device=device,
        verbose=False,
    )

    out_boxes: List[Dict[str, Any]] = []
    for idx, res in enumerate(results):
        off_x = items[idx]["offsetX"]
        off_y = items[idx]["offsetY"]
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue
        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            conf_val = float(b.conf[0].cpu().numpy()) if b.conf is not None else float(req.conf)
            cls_id = int(b.cls[0].cpu().numpy()) if b.cls is not None else 0
            cls_name = TREE_CLASSES[cls_id] if 0 <= cls_id < len(TREE_CLASSES) else "orchard_tree"
            x1 = off_x + float(xyxy[0])
            y1 = off_y + float(xyxy[1])
            x2 = off_x + float(xyxy[2])
            y2 = off_y + float(xyxy[3])
            out_boxes.append(
                {
                    "xmin": round(x1, 2),
                    "ymin": round(y1, 2),
                    "xmax": round(x2, 2),
                    "ymax": round(y2, 2),
                    "conf": round(conf_val, 4),
                    "classId": cls_id,
                    "className": cls_name,
                }
            )

    return {
        "ok": True,
        "engine": f"YOLOv8 FastAPI Batch ({_MODEL_STATE['device']})",
        "boxes": out_boxes,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("YOLO_INFER_PORT", "8001"))
    uvicorn.run("inference_service:app", host="127.0.0.1", port=port, reload=False)
