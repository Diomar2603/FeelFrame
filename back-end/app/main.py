from fastapi import FastAPI
from app.routes import userRoute
from app.routes import videoRoute

app = FastAPI()

# Inclui as rotas da API
app.include_router(userRoute.router)
app.include_router(videoRoute.router)

# Rota inicial
@app.get("/")
def root():
    return {"message": "API Online"}
