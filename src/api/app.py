"""
Main FastAPI application for serving NLP models.

This module initializes the FastAPI application and configures it
for serving NLP models through a RESTful API.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..utils.logging_utils import get_logger
from . import routes

# Initialize logger
logger = get_logger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Advanced NLP API",
    description="API for various NLP tasks including classification, named entity recognition, sentiment analysis, and summarization",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Include routers for different NLP tasks
app.include_router(
    routes.classification_router, prefix="/api/classification", tags=["Classification"]
)
app.include_router(
    routes.ner_router, prefix="/api/ner", tags=["Named Entity Recognition"]
)
app.include_router(
    routes.sentiment_router, prefix="/api/sentiment", tags=["Sentiment Analysis"]
)
app.include_router(
    routes.summarization_router, prefix="/api/summarization", tags=["Summarization"]
)
app.include_router(routes.model_hub_router, prefix="/api/models", tags=["Model Hub"])


# Add health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Check if the API is running."""
    return {"status": "healthy"}


# Add error handlers
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "message": str(exc)},
    )


# Startup event to load default models
@app.on_event("startup")
async def startup_event():
    """Load default models on startup."""
    logger.info("Initializing NLP API")
    try:
        from .model_registry import initialize_models

        initialize_models()
        logger.info("Default models initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing models: {e}", exc_info=True)
        # Don't fail startup, but log the error


# Shutdown event to clean up resources
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down NLP API")
    try:
        from .model_registry import cleanup_models

        cleanup_models()
        logger.info("Resources cleaned up successfully")
    except Exception as e:
        logger.error(f"Error cleaning up resources: {e}", exc_info=True)


# If running this file directly, start the API
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    uvicorn.run("app:app", host=host, port=port, reload=True)
