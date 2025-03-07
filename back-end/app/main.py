from fastapi import FastAPI
from app.routes import user_routes

app = FastAPI()

# Inclui as rotas da API
app.include_router(user_routes.router)

# Rota inicial
@app.get("/")
def root():
    return {"message": "API Online"}
