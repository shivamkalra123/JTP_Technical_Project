from fastapi import FastAPI
from routes import recommendation_routes
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.include_router(recommendation_routes.router)
origins = [
    "http://localhost:5173",  # React dev server origin
    # add more origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, OPTIONS...)
    allow_headers=["*"],
)