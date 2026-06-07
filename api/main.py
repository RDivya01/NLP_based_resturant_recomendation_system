"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from api.routes import router


app = FastAPI(title="NLP-Based Restaurant Recommendation System")
app.include_router(router)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return API health status."""

    return {"status": "ok"}
