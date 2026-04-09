from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(
    title="FineChord API",
    version="1.0.0",
    description="API d'analyse harmonique de fichiers audio et vidéo",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "name": "FineChord API",
        "version": "1.0.0",
        "docs": "/docs",
    }
