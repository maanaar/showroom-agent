from fastapi import FastAPI
from api.chat import router

app = FastAPI(title="AI showroom Agent")

app.include_router(router, prefix="/api")