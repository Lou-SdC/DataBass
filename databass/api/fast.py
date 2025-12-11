from pathlib import Path
from typing import Any, List
import os
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import tempfile
import librosa

from models import conv2D


app = FastAPI(
    title="DataBass Model API",
    version="1.0.0"
)

_model: Any | None = None
_label_encoder: Any | None = None


class PredictionResponse(BaseModel):
    predicted_note: str


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "DataBass conv2D model API", "docs": "/docs"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    try:
        contents = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {exc}") from exc
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
        tmp_file.write(contents)
        tmp_file.flush()
        try:
            signal, sr = librosa.load(tmp_file.name, sr=None)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to load audio: {exc}") from exc

    global _model, _label_encoder
    if _model is None:
        _model, _label_encoder = conv2D.load_model()

    predicted_note = conv2D.predict(
        signal,
        sr,
        _model,
        _label_encoder
    )

    return PredictionResponse(
        predicted_note=predicted_note,
    )
