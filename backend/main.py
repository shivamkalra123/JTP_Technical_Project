from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import recommendation_routes

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",      
        "http://127.0.0.1:5173",       
        "http://host.docker.internal:5173",  
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Then include your routes
app.include_router(recommendation_routes.router)
