# src/medmdt/api/app.py
from fastapi import FastAPI

from medmdt.api.routes.consultation import router as consultation_router
from medmdt.api.routes.knowledge import router as knowledge_router
from medmdt.api.routes.ws import router as ws_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="MedMDT",
        description="多专家会诊医学Agent系统API",
        version="0.1.0",
    )

    @app.get("/health")
    def health_check():
        return {"status": "ok"}

    app.include_router(consultation_router)
    app.include_router(knowledge_router)
    app.include_router(ws_router)

    return app


app = create_app()
