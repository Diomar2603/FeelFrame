from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import userRoute, videoRoute, relatorioRoute

app = FastAPI(title="FeelFrame API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrinja para domínios específicos em produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auth (register / login / google / me)
app.include_router(userRoute.router)
# Vídeos (upload, progress, data, list)
app.include_router(videoRoute.router)
# Relatórios PDF
app.include_router(relatorioRoute.router)


@app.get("/")
def root():
    return {"message": "FeelFrame API Online"}
