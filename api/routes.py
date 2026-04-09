import os
import uuid
import traceback
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from config import (
    UPLOAD_DIR,
    RESULTS_DIR,
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_SIZE_MB,
    CHORD_CLASSES,
)
from services.analysis_service import AnalysisService
from services.export_service import export_json, export_lrc, export_midi

router = APIRouter(prefix="/api", tags=["analysis"])

analysis_service = None


def get_service() -> AnalysisService:
    global analysis_service
    if analysis_service is None:
        analysis_service = AnalysisService()
    return analysis_service


class AnalysisResponse(BaseModel):
    success: bool
    job_id: str
    message: str


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "FineChord API",
        "chord_classes": len(CHORD_CLASSES),
    }


@router.get("/model-info")
async def model_info():
    return {
        "model": "ChordCNNLSTM",
        "architecture": "CNN (3 layers) + BiLSTM (2 layers)",
        "input": "CQT Chroma (12 bins)",
        "chord_classes": CHORD_CLASSES,
        "num_classes": len(CHORD_CLASSES),
        "features": [
            "CQT Chroma extraction",
            "Source separation (Demucs)",
            "Beat tracking",
            "HMM smoothing",
            "Key detection",
            "Roman numeral analysis",
        ],
    }


@router.post("/analyze")
async def analyze_file(
    file: UploadFile = File(...),
    use_separation: bool = Query(False, description="Enable source separation for better accuracy"),
    rhythmic_resolution: str = Query("half", description="Rhythmic resolution: whole, half, quarter, eighth, sixteenth"),
    min_confidence: float = Query(0.4, description="Minimum confidence threshold (0.0-1.0)"),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {ext}. Allowed: {ALLOWED_EXTENSIONS}",
        )

    job_id = uuid.uuid4().hex[:12]
    temp_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    try:
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max: {MAX_UPLOAD_SIZE_MB}MB",
            )

        with open(temp_path, "wb") as f:
            f.write(content)

        service = get_service()
        result = service.analyze(
            str(temp_path),
            use_separation=use_separation,
            rhythmic_resolution=rhythmic_resolution,
            min_confidence=min_confidence,
        )

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            os.remove(temp_path)


@router.get("/result/{job_id}")
async def get_result(job_id: str):
    result_path = RESULTS_DIR / f"{job_id}.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")

    import json
    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/export/{job_id}/{format}")
async def export_result(job_id: str, format: str):
    result_path = RESULTS_DIR / f"{job_id}.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")

    import json
    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    if format == "json":
        return FileResponse(str(result_path), media_type="application/json", filename=f"{job_id}.json")
    elif format == "lrc":
        lrc_path = export_lrc(result, job_id)
        return FileResponse(lrc_path, media_type="text/plain", filename=f"{job_id}.lrc")
    elif format == "midi":
        midi_path = export_midi(result, job_id)
        return FileResponse(midi_path, media_type="audio/midi", filename=f"{job_id}.mid")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported export format: {format}")
