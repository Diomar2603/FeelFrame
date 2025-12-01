from fastapi import FastAPI
from app.routes import userRoute
from app.routes import videoRoute
from app.routes import relatorioRoute

app = FastAPI()

# Inclui as rotas da API
app.include_router(userRoute.router)
app.include_router(videoRoute.router)
app.include_router(relatorioRoute.router)

# Rota inicial
@app.get("/")
def root():
    return {"message": "API Online"}
