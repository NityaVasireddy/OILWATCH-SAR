import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .api.detection import router as detection_router
from .api.drift import router as drift_router
from .api.ais import router as ais_router
from .api.report import router as report_router
from .api.correlation import router as correlation_router


# Load environment variables
load_dotenv()


# Create FastAPI application
app = FastAPI(
    title="OILWATCH-SAR",
    version="0.1.0"
)


# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register API routers
app.include_router(
    detection_router,
    prefix="/api"
)

app.include_router(
    drift_router,
    prefix="/api"
)

app.include_router(
    ais_router,
    prefix="/api"
)

app.include_router(
    report_router,
    prefix="/api"
)

app.include_router(
    correlation_router,
    prefix="/api"
)


# Health check endpoint
@app.get("/api/health")
def health():
    p = Path(
        os.getenv(
            "MODEL_PATH",
            "backend/model/best_model.pth"
        )
    )

    loaded = False
    version = None

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        channels = 1
        size = None

        if p.exists():
            ck = torch.load(
                p,
                map_location="cpu"
            )

            loaded = True
            version = ck.get("trained_at")
            channels = ck.get(
                "input_channels",
                1
            )
            size = ck.get(
                "input_size"
            )

    except Exception:
        device = "cpu"
        channels = 1
        size = None
        loaded = False

    return {
        "backend_status": "ok",
        "model_loaded": loaded,
        "model_version": version,
        "device": device,
        "input_channels": channels,
        "input_size": size
    }


# Serve frontend
@app.get("/")
def root():
    frontend = (
        Path(__file__).resolve().parents[2]
        / "frontend"
        / "index.html"
    )

    return FileResponse(frontend)
