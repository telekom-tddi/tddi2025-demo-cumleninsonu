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
import io

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

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.models.call_center_llm import CallCenterLLMManager
from src.functions.function_caller import FunctionCaller
from src.translator.translator import Translator
from src.tts import TextToSpeechService
from src.stt import SpeechToTextService
from src.utils.config import (
    API_HOST,
    API_PORT,
    MAX_CACHE_SIZE,
    CALL_CENTER_MODE
)


for handler in logging.root.handlers[:]:
    print(handler)
    logging.root.removeHandler(handler)

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

@lru_cache(maxsize=1)
def get_translator():
    """Get or initialize the translator."""
    return Translator()

@lru_cache(maxsize=1)
def get_tts_service():
    """Get or initialize the Text-to-Speech service."""
    return TextToSpeechService()

@lru_cache(maxsize=1)
def get_stt_service():
    """Get or initialize the Speech-to-Text service."""
    return SpeechToTextService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the models at startup and gracefully shutdown."""
    logger.info("Application startup: Loading call center models...")
    get_llm_manager()
    get_function_caller()
    get_translator()
    get_tts_service()
    get_stt_service()
    logger.info("Call center models loaded successfully.")
    yield
    logger.info("Application shutdown.")

# Initialize the FastAPI app
app = FastAPI(
    title="Call Center Chatbot API",
    description="AI ile desteklenen çağrı merkezi chatbot API'si.",
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

class TextToSpeechRequest(BaseModel):
    """Request model for the TTS endpoint."""
    text: str = Field(..., description="Text to be converted to speech")
    language: str = Field("tr", description="Language of the text")

class SpeechToTextResponse(BaseModel):
    """Response model for the STT endpoint."""
    text: str = Field(..., description="Transcribed text")
    elapsed_time: float = Field(..., description="Time taken to process the audio")
    confidence: Optional[float] = Field(None, description="Confidence score for transcription")
    detected_language: Optional[str] = Field(None, description="Detected language of the audio")
    segments: Optional[List[Dict[str, Any]]] = Field(None, description="Detailed transcription segments")

class TranslateRequest(BaseModel):
    """Request model for the translate endpoint."""
    text: str = Field(..., description="Text to be translated")
    source_lang: str = Field("tr", description="Source language of the text")
    target_lang: str = Field("en", description="Target language for translation")

class TranslateResponse(BaseModel):
    """Response model for the translate endpoint."""
    translated_text: str = Field(..., description="The translated text")
    original_text: str = Field(..., description="The original text")
    source_lang: str = Field(..., description="Source language")
    target_lang: str = Field(..., description="Target language")

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

@app.post("/tts")
async def text_to_speech(request: TextToSpeechRequest):
    """Convert text to speech and stream the audio back."""
    try:
        tts_service = get_tts_service()
        
        # Generate speech
        audio_bytes = tts_service.text_to_speech(request.text, language=request.language)
        
        if audio_bytes is None:
            raise HTTPException(status_code=500, detail="TTS conversion failed")
            
        # Stream the audio
        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")
        
    except Exception as e:
        logger.error(f"Error in TTS endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error in TTS: {str(e)}")

@app.post("/stt", response_model=SpeechToTextResponse)
async def speech_to_text(audio_file: UploadFile = File(...), language: str = "tr", detailed: bool = False):
    """Convert speech to text from an audio file."""
    start_time = time.time()
    
    try:
        stt_service = get_stt_service()
        
        # Read audio data directly into bytes
        audio_bytes = await audio_file.read()
        
        if detailed:
            # Use the new detailed transcription method
            result = stt_service.transcribe_audio_bytes(audio_bytes, language=language)
            
            return {
                "text": result.get("text", ""),
                "elapsed_time": result.get("processing_time", time.time() - start_time),
                "confidence": result.get("confidence", 0.0),
                "detected_language": result.get("detected_language", language),
                "segments": result.get("segments", [])
            }
        else:
            # Save the uploaded file temporarily for backward compatibility
            temp_audio_path = f"temp_{int(time.time())}_{audio_file.filename or 'audio.wav'}"
            with open(temp_audio_path, "wb") as f:
                f.write(audio_bytes)
                
            # Transcribe the audio file
            transcribed_text = stt_service.speech_to_text(temp_audio_path, language=language)
            
            # Clean up the temporary file
            os.remove(temp_audio_path)
            
            if not transcribed_text:
                raise HTTPException(status_code=500, detail="STT transcription failed")
                
            elapsed_time = time.time() - start_time
            
            return {
                "text": transcribed_text,
                "elapsed_time": elapsed_time
            }
        
    except Exception as e:
        logger.error(f"Error in STT endpoint: {e}")
        # Clean up temp file in case of error
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error in STT: {str(e)}")
@app.post("/translate", response_model=TranslateResponse)
async def translate_text(request: TranslateRequest):
    """Translate text from a source language to a target language."""
    try:
        translator = get_translator()
        
        if request.source_lang == "tr" and request.target_lang == "en":
            translated_text = translator.translate_tr_to_en(request.text)
        elif request.source_lang == "en" and request.target_lang == "tr":
            translated_text = translator.translate_en_to_tr(request.text)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported translation direction. Only 'tr' to 'en' and 'en' to 'tr' are supported."
            )
        
        return {
            "translated_text": translated_text,
            "original_text": request.text,
            "source_lang": request.source_lang,
            "target_lang": request.target_lang
        }
    except Exception as e:
        logger.error(f"Error in translation endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error in translation: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a customer message and generate a response with function calling."""
    start_time = time.time()
    try:
        llm_manager = get_llm_manager()
        translator = get_translator()
        isQueryTranslated = False
        # Translate query to English if it's in Turkish
        translated_query = translator.translate_tr_to_en(request.query)
        if translated_query != request.query:
            isQueryTranslated = True
        
        # Translate conversation history to English
        translated_history = []
        if request.conversation_history:
            for message in request.conversation_history:
                if message.get("role").lower() == "user":
                    translated_content = translator.translate_tr_to_en(message["content"])
                elif message.get("role").lower() == "assistant":
                    translated_content = translator.translate_tr_to_en(message["content"])
                else:
                    translated_content = message["content"]
                translated_history.append({"role": message["role"], "content": translated_content})

        if request.customer_id:
            query = translated_query + " customer_id: " + request.customer_id
        else:
            query = translated_query

        response, function_results = llm_manager.generate_response(
            query=query,
            conversation_history=translated_history,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Translate response back to Turkish
        if isQueryTranslated:
            final_response = translator.translate_en_to_tr(response)
        else:
            final_response = response
        
        elapsed_time = time.time() - start_time
        
        return {
            "query": request.query,
            "response": final_response,
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
            public_url_object = ngrok.connect(addr=API_PORT, proto="http", hostname=API_HOST)
            public_url = public_url_object.public_url
            logger.info(f"🚀 Call Center API available at: {public_url}")
            logger.info(f"📋 Health check: {public_url}/health")
            logger.info(f"💬 Chat endpoint: {public_url}/chat")
            logger.info(f"🔧 Functions: {public_url}/functions")
            logger.info(f"📖 API docs: {public_url}/docs")
        except Exception as e:
            logger.warning(f"Failed to set up ngrok tunnel: {e}")
            logger.info("Falling back to random server...")
            logger.info(f"🚀 Call Center API available at: http://{API_HOST}:{API_PORT}")
            public_url_object = ngrok.connect(addr=API_PORT, proto="http")
            public_url = public_url_object.public_url
    else:
        if ngrok_token and not NGROK_AVAILABLE:
            logger.warning("NGROK_AUTHTOKEN is set but pyngrok is not installed. Install with: pip install pyngrok")
        
        logger.info("Running locally without ngrok")
        logger.info(f"🚀 Call Center API available at: http://{API_HOST}:{API_PORT}")
        logger.info(f"📋 Health check: http://{API_HOST}:{API_PORT}/health")
        logger.info(f"💬 Chat endpoint: http://{API_HOST}:{API_PORT}/chat")
        logger.info(f"🔧 Functions: http://{API_HOST}:{API_PORT}/functions")
        logger.info(f"📖 API docs: http://{API_HOST}:{API_PORT}/docs")
    try:
        uvicorn.run(app, host="0.0.0.0", port=API_PORT)
    finally:
        if 'public_url_object' in locals() and public_url_object:
            ngrok.disconnect(public_url_object.public_url)
        ngrok.kill()
        logger.info("Ngrok tunnel closed.")

if __name__ == "__main__":
    main() 