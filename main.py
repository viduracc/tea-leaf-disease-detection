import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import numpy as np

from services.ml_service import ml_model

app = FastAPI(
    title="Tea Leaf Disease Detection API",
    description=(
        "EfficientNet-based tea leaf disease classification with "
        "OOD detection, confidence tiers, and masked Grad-CAM."
    ),
    version="5.1",
)


@app.get("/health")
def health():
    """Health check."""
    return {
        "status": "ok",
        "model_loaded": ml_model._model is not None,
        "gradcam_available": ml_model._backbone is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a tea leaf image and receive a prediction.

    Response schema:
      status      — "high" | "medium" | "low" | "rejected"
      prediction  — class name or null (if rejected)
      confidence  — float or null
      message     — human-readable status message
      disclaimer  — research tool disclaimer
      all_probs   — {class: probability} for medium/low, null for high/rejected
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()

    try:
        result = ml_model.predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {
        "status": result["status"],
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "message": result["message"],
        "disclaimer": result["disclaimer"],
        "all_probs": result["all_probs"],
    }


@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    """
    Upload a tea leaf image and receive a Grad-CAM heatmap overlay as PNG.

    The heatmap is masked to the leaf region — background activations
    are zeroed out. Returns a 300x300 PNG image.
    Prediction and confidence are included in response headers.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()

    try:
        result = ml_model.predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    if result["status"] == "rejected":
        raise HTTPException(
            status_code=422,
            detail=result["message"],
        )

    overlay = result.get("gradcam_overlay")
    if overlay is None:
        raise HTTPException(
            status_code=503,
            detail="Grad-CAM unavailable (backbone_extractor.keras may be missing).",
        )

    overlay_img = Image.fromarray(overlay.astype(np.uint8))
    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={
            "X-Status": result["status"],
            "X-Prediction": result["prediction"] or "",
            "X-Confidence": f"{result['confidence']:.4f}"
            if result["confidence"] else "0",
        },
    )
