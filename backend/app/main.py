# main.py

from dotenv import load_dotenv
load_dotenv() # Ensure environment variables are loaded first

import os # Import the os module
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.api.routes import router as api_router
from app.auth.routers import router as auth_router
from app.db import init_db

app = FastAPI(title="Reconbot API", version="1.0.0")

# Custom middleware for COOP and COEP
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response: Response = await call_next(request)
        response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
        response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
        return response

app.add_middleware(SecurityHeadersMiddleware)

# --- ✅ CORRECTED CORS CONFIGURATION ---
# Define production origins as a default
origins = [
    "https://coldemailai.in",
    "https://www.coldemailai.in",
]

# Get allowed origins from environment variable
# Example: "http://localhost:5173,http://127.0.0.1:5173"
allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
if allowed_origins_env:
    # Split the comma-separated string into a list and add to origins
    origins.extend([origin.strip() for origin in allowed_origins_env.split(",")])

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Use the dynamic origins list
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- END CORRECTION ---

@app.on_event("startup")
def on_startup():
    init_db()

# Include Routers with Prefixes
app.include_router(api_router, prefix="/api") # Correctly prefix the api_router
app.include_router(auth_router, prefix="/auth")

@app.get("/")
def root():
    return {"message": "Reconbot API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}