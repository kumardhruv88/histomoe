"""
HistoMoE FastAPI REST Endpoint

Provides a production-ready API for running HistoMoE predictions.

Usage (Start Server):
    uvicorn api:app --reload --port 8000

Usage (Client):
    curl -X POST "http://localhost:8000/predict/" \\
         -F "file=@patch.png" \\
         -F "cancer_type=CCRCC"
"""

import io
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms as T

from histomoe.models.histomoe_model import HistoMoE
from histomoe.data.metadata_utils import CANCER_TYPES, cancer_type_to_id
from histomoe.data.transforms import get_transforms

# ── Application ──────────────────────────────────────────────────────
app = FastAPI(
    title="HistoMoE API",
    description="REST API for Histology-Guided MoE Gene Expression Prediction",
    version="0.1.0",
)

# ── Module-level state with explicit Optional types ───────────────────
_model: Optional[HistoMoE] = None
_transform: Optional[T.Compose] = None


@app.on_event("startup")
def _load_model() -> None:
    """Load the HistoMoE model and transforms on server startup."""
    global _model, _transform

    _model = HistoMoE(
        backbone="resnet50",
        n_genes=250,
        n_experts=5,
        gating_mode="soft",
        pretrained_backbone=False,  # set True when real weights are available
    )
    _model.eval()

    _transform = get_transforms(split="test", patch_size=224)
    print("HistoMoE model and transforms loaded successfully.")


# ── Helper ────────────────────────────────────────────────────────────
def _get_model() -> HistoMoE:
    """Return the loaded model or raise a 503 if not yet ready."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return _model


def _get_transform() -> T.Compose:
    """Return the loaded transform or raise a 503 if not yet ready."""
    if _transform is None:
        raise HTTPException(status_code=503, detail="Transforms not loaded yet.")
    return _transform


# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/")
def read_root() -> dict:
    return {"message": "Welcome to the HistoMoE API — POST /predict/ to run inference."}


@app.get("/cancer_types")
def get_cancer_types() -> dict:
    """Return the list of supported cancer types."""
    return {"supported_types": CANCER_TYPES}


@app.post("/predict/")
async def predict_patch(
    file: UploadFile = File(...),
    cancer_type: str = Form(...),
) -> JSONResponse:
    """
    Run HistoMoE prediction on a single histology patch.

    - **file**: Histology image (PNG, JPG, JPEG)
    - **cancer_type**: One of [CCRCC, COAD, LUAD, PAAD, PRAD]
    """
    model = _get_model()
    transform = _get_transform()

    # 1. Validate cancer type
    cancer_type = cancer_type.upper()
    if cancer_type not in CANCER_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid cancer type '{cancer_type}'. Supported: {CANCER_TYPES}",
        )
    cancer_id = cancer_type_to_id(cancer_type)

    # 2. Read & transform image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

    img_tensor: torch.Tensor = transform(image).unsqueeze(0)   # [1, 3, 224, 224]
    cancer_id_tensor = torch.tensor([cancer_id], dtype=torch.long)

    # 3. Infer
    try:
        with torch.no_grad():
            result = model.predict_patches(img_tensor, cancer_id_tensor)

        predictions: list = result["predictions"][0].tolist()
        routing_weights: list = result["routing_weights"][0].tolist()
        dominant_expert: int = int(result["dominant_expert"][0].item())

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    # 4. Build response
    gene_names = [f"Gene_{i:03d}" for i in range(len(predictions))]
    payload = {
        "cancer_type": cancer_type,
        "dominant_expert": CANCER_TYPES[dominant_expert],
        "routing_weights": dict(zip(CANCER_TYPES, routing_weights)),
        "gene_predictions": dict(zip(gene_names, predictions)),
    }
    return JSONResponse(content=payload)
