from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import analyze, dub, script, auth, trends, youtube

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router)
app.include_router(dub.router)
app.include_router(script.router)
app.include_router(auth.router)
app.include_router(trends.router)
app.include_router(youtube.router)

@app.get("/health")
async def health():
    return {"status": "ok", "message": "CreatorMentor API running"}
