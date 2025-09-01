# src/medmdt/api/app.py
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from medmdt.api.routes.consultation import router as consultation_router
from medmdt.api.routes.knowledge import router as knowledge_router
from medmdt.api.routes.settings import router as settings_router
from medmdt.api.routes.ws import router as ws_router
from medmdt.llm.http_clients import close_shared_http_clients


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    try:
        yield
    finally:
        await close_shared_http_clients()


def create_app() -> FastAPI:
    app = FastAPI(
        title="MedMDT",
        description="多专家会诊医学Agent系统API",
        version="0.1.0",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health_check():
        return {"status": "ok"}

    app.include_router(consultation_router)
    app.include_router(knowledge_router)
    app.include_router(settings_router)
    app.include_router(ws_router)

    return app


app = create_app()
