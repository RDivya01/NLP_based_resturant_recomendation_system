"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router


app = FastAPI(title="NLP-Based Restaurant Recommendation System")
app.include_router(router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check() -> dict[str, str]:
    """Return API health status."""

    return {"status": "ok"}
