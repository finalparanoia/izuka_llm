from fastapi import FastAPI
from izuka_llm.app import openai_compatable
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

for router in [openai_compatable]:
    app.include_router(router.router)


origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    from uvicorn import run
    run(app=app, host="0.0.0.0", port=8080)
