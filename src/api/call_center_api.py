"""
Call Center API
FastAPI backend for the call center chatbot with function calling capabilities.
"""

import os
import sys
import logging
import time
from typing import Dict, Any, List, Optional
from functools import lru_cache
from contextlib import asynccontextmanager

# Optional ngrok import
try:
    from pyngrok import ngrok, conf
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False
    ngrok = None
    conf = None

# Add the parent directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.models.call_center_llm import CallCenterLLMManager
from src.functions.function_caller import FunctionCaller
from src.utils.config import (
    API_HOST,
    API_PORT,
    MAX_CACHE_SIZE,
    CALL_CENTER_MODE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize managers lazily
@lru_cache(maxsize=1)
def get_llm_manager():
    """Get or initialize the call center LLM manager."""
    return CallCenterLLMManager()

@lru_cache(maxsize=1)
def get_function_caller():
    """Get or initialize the function caller."""
    return FunctionCaller()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the models at startup and gracefully shutdown."""
    logger.info("Application startup: Loading call center models...")
    get_llm_manager()
    get_function_caller()
    logger.info("Call center models loaded successfully.")
    yield
    logger.info("Application shutdown.")

# Initialize the FastAPI app
app = FastAPI(
    title="Call Center Chatbot API",
    description="API for AI-powered call center chatbot with function calling",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define API models
class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""
    query: str = Field(..., description="The customer's message")
    customer_id: Optional[str] = Field(None, description="Customer ID for personalized assistance")
    conversation_history: Optional[List[Dict[str, str]]] = Field([], description="Previous conversation messages")
    temperature: float = Field(0.7, description="Temperature for generation")
    top_p: float = Field(0.9, description="Top-p sampling parameter")

class FunctionCallRequest(BaseModel):
    """Request model for direct function calls."""
    function_name: str = Field(..., description="Name of the function to call")
    parameters: Dict[str, Any] = Field({}, description="Parameters for the function")

class ChatResponse(BaseModel):
    """Response model for the chat endpoint."""
    query: str = Field(..., description="The original query")
    response: str = Field(..., description="The assistant's response")
    function_calls: List[Dict[str, Any]] = Field(..., description="Function calls that were executed")
    elapsed_time: float = Field(..., description="Time taken to process the query")

class FunctionCallResponse(BaseModel):
    """Response model for function call endpoints."""
    function_name: str = Field(..., description="Name of the function that was called")
    parameters: Dict[str, Any] = Field(..., description="Parameters that were used")
    result: Dict[str, Any] = Field(..., description="Result of the function call")
    elapsed_time: float = Field(..., description="Time taken to execute the function")

class HealthResponse(BaseModel):
    """Response model for the health endpoint."""
    status: str = Field(..., description="Health status")
    llm_ready: bool = Field(..., description="Whether the LLM is ready")
    function_caller_ready: bool = Field(..., description="Whether the function caller is ready")
    available_functions: int = Field(..., description="Number of available functions")
    mode: str = Field(..., description="Operating mode")

class FunctionsResponse(BaseModel):
    """Response model for the functions endpoint."""
    functions: Dict[str, Any] = Field(..., description="Available functions and their schemas")

# Define API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of the call center API."""
    try:
        llm_manager = get_llm_manager()
        function_caller = get_function_caller()
        
        return {
            "status": "ok",
            "llm_ready": True,
            "function_caller_ready": True,
            "available_functions": len(function_caller.registry),
            "mode": "call_center" if CALL_CENTER_MODE else "unknown"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/functions", response_model=FunctionsResponse)
async def get_available_functions():
    """Get information about available functions."""
    try:
        function_caller = get_function_caller()
        functions = function_caller.get_available_functions()
        
        return {
            "functions": functions
        }
    except Exception as e:
        logger.error(f"Error getting functions: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting functions: {str(e)}")

@app.post("/function/call", response_model=FunctionCallResponse)
async def call_function(request: FunctionCallRequest):
    """Execute a function directly."""
    start_time = time.time()
    
    try:
        function_caller = get_function_caller()
        
        # Execute the function
        result = function_caller.execute_function(
            function_name=request.function_name,
            parameters=request.parameters
        )
        
        elapsed_time = time.time() - start_time
        
        return {
            "function_name": request.function_name,
            "parameters": request.parameters,
            "result": result,
            "elapsed_time": elapsed_time
        }
    except Exception as e:
        logger.error(f"Error calling function: {e}")
        raise HTTPException(status_code=500, detail=f"Error calling function: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a customer message and generate a response with function calling."""
    start_time = time.time()
    logger.info(f"Chat request: {request}")
    try:
        llm_manager = get_llm_manager()
        if request.customer_id:
            query = request.query + " customer_id: " + request.customer_id
        else:
            query = request.query
        # Generate response with function calling
        logger.info(f"Query: {query}")
        response, function_results = llm_manager.generate_response(
            query=query,
            conversation_history=request.conversation_history,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        elapsed_time = time.time() - start_time
        
        return {
            "query": request.query,
            "response": response,
            "function_calls": function_results,
            "elapsed_time": elapsed_time
        }
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

# Main function for running the API
def main():
    """Main function to run the API server."""
    import uvicorn
    
    # Check for ngrok configuration (only if ngrok is available)
    ngrok_token = os.getenv("NGROK_AUTHTOKEN")
    if NGROK_AVAILABLE and ngrok_token and ngrok and conf:
        logger.info("Setting up ngrok tunnel...")
        conf.get_default().auth_token = ngrok_token
        try:
            # Start ngrok tunnel
            tunnel = ngrok.connect(f"{API_HOST}:{API_PORT}")
            public_url = tunnel.public_url
            logger.info(f"🚀 Call Center API available at: {public_url}")
            logger.info(f"📋 Health check: {public_url}/health")
            logger.info(f"💬 Chat endpoint: {public_url}/chat")
            logger.info(f"🔧 Functions: {public_url}/functions")
            logger.info(f"📖 API docs: {public_url}/docs")
        except Exception as e:
            logger.warning(f"Failed to set up ngrok tunnel: {e}")
            logger.info("Falling back to local server...")
            logger.info(f"🚀 Call Center API available at: http://{API_HOST}:{API_PORT}")
    else:
        if ngrok_token and not NGROK_AVAILABLE:
            logger.warning("NGROK_AUTHTOKEN is set but pyngrok is not installed. Install with: pip install pyngrok")
        
        logger.info("Running locally without ngrok")
        logger.info(f"🚀 Call Center API available at: http://{API_HOST}:{API_PORT}")
        logger.info(f"📋 Health check: http://{API_HOST}:{API_PORT}/health")
        logger.info(f"💬 Chat endpoint: http://{API_HOST}:{API_PORT}/chat")
        logger.info(f"🔧 Functions: http://{API_HOST}:{API_PORT}/functions")
        logger.info(f"📖 API docs: http://{API_HOST}:{API_PORT}/docs")
    
    # Run the server
    uvicorn.run(
        "src.api.call_center_api:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main() 