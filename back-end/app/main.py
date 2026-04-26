from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import userRoute
from app.routes import videoRoute
from app.routes import relatorioRoute

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permite QUALQUER origem (apenas para ambiente local)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclui as rotas da API
app.include_router(userRoute.router)
app.include_router(videoRoute.router)
app.include_router(relatorioRoute.router)

# Rota inicial
@app.get("/")
def root():
    return {"message": "API Online"}
