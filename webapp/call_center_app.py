import os
import sys
import logging
from typing import Dict, Any
import time
import requests
import json
import io
import hashlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from audiorecorder import audiorecorder

from src.utils.config import (
    API_HOST,
    APP_TITLE,
    API_PORT,
    APP_DESCRIPTION,
    API_CONNECT_TIMEOUT,
    API_READ_TIMEOUT,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

resolved_host = "localhost" if API_HOST == "0.0.0.0" else API_HOST
API_URL = f"https://{resolved_host}"
if resolved_host == "localhost":
    API_URL = f"http://{resolved_host}:{API_PORT}"

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9

TTS_LANGUAGES = {
    "English": "en",
    "Turkish": "tr",
    "Spanish": "es",
    "German": "de",
}

def check_api_health() -> Dict[str, Any]:
    """Check the health of the call center API.
    
    Returns:
        Dictionary with health information
    """
    try:
        logger.info(f"Checking API health at {API_URL}/health")
        response = requests.get(
            f"{API_URL}/health",
            timeout=(API_CONNECT_TIMEOUT, min(API_READ_TIMEOUT, 10)),
        )
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"API health check failed with status code {response.status_code}")
            return {
                "status": "error",
                "llm_ready": False,
                "function_caller_ready": False,
                "available_functions": 0,
                "mode": "unknown"
            }
    except Exception as e:
        logger.error(f"Error checking API health: {e}")
        return {
            "status": "error",
            "llm_ready": False,
            "function_caller_ready": False,
            "available_functions": 0,
            "mode": "unknown"
        }

def get_available_functions() -> Dict[str, Any]:
    """Get available functions from the API.
    
    Returns:
        Dictionary with available functions
    """
    try:
        logger.info(f"Getting available functions from {API_URL}/functions")
        response = requests.get(
            f"{API_URL}/functions",
            timeout=(API_CONNECT_TIMEOUT, min(API_READ_TIMEOUT, 10)),
        )
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get functions with status code {response.status_code}")
            return {"functions": {}}
    except Exception as e:
        logger.error(f"Error getting functions: {e}")
        return {"functions": {}}

def transcribe_audio(audio_bytes: bytes) -> str:
    """Send audio to the STT API and get the transcript.
    
    Args:
        audio_bytes: The audio data to transcribe.
        
    Returns:
        The transcribed text.
    """
    try:
        stt_lang = st.session_state.get("stt_language", "tr")
        logger.info(f"Sending audio for transcription with language: {stt_lang}")
        files = {"audio_file": ("audio.wav", audio_bytes, "audio/wav")}
        params = {"language": stt_lang}
        response = requests.post(f"{API_URL}/stt", files=files, params=params, timeout=(API_CONNECT_TIMEOUT, API_READ_TIMEOUT))
        if response.status_code == 200:
            return response.json().get("text", "")
        else:
            logger.error(f"STT API failed with status code {response.status_code}")
            return ""
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return ""

def text_to_speech(text: str, language: str | None = None) -> bytes:
    """Send text to the TTS API and get the speech audio.
    
    Args:
        text: The text to convert to speech.
        language: Optional language code (e.g., 'en', 'tr')
        
    Returns:
        The audio data in bytes.
    """
    try:
        final_language = language or st.session_state.get("tts_language", "tr")
        logger.info(f"Sending text for speech synthesis with language {final_language}: '{text[:50]}...'")
        payload = {"text": text, "language": final_language}
        response = requests.post(
            f"{API_URL}/tts",
            json=payload,
            timeout=(API_CONNECT_TIMEOUT, API_READ_TIMEOUT),
            stream=True
        )
        if response.status_code == 200:
            return response.content
        else:
            logger.error(f"TTS API failed with status code {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error synthesizing speech: {e}")
        return None

def send_chat_message(
    query: str,
    customer_id: str = None,
    conversation_history: list = None,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P
) -> Dict[str, Any]:
    """Send a message to the call center chatbot.
    
    Args:
        query: Customer message
        customer_id: Optional customer ID
        conversation_history: Previous conversation messages
        temperature: Temperature for generation
        top_p: Top-p sampling parameter
        
    Returns:
        Dictionary with the response from the API
    """
    try:
        logger.info(f"Sending chat message: {query}")
        logger.info(f"API URL: {API_URL}/chat")
        
        payload = {
            "query": query,
            "temperature": temperature,
            "top_p": top_p
        }
        
        if customer_id:
            payload["customer_id"] = customer_id
            
        if conversation_history:
            payload["conversation_history"] = conversation_history
        
        response = requests.post(
            f"{API_URL}/chat",
            json=payload,
            timeout=(API_CONNECT_TIMEOUT, API_READ_TIMEOUT),
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Chat API failed with status code {response.status_code}")
            return {
                "query": query,
                "response": f"Error: API returned status code {response.status_code}",
                "function_calls": [],
                "elapsed_time": 0.0
            }
    except Exception as e:
        logger.error(f"Error sending chat message: {e}")
        return {
            "query": query,
            "response": f"Error: {str(e)}",
            "function_calls": [],
            "elapsed_time": 0.0
        }

def initialize_session_state():
    """Initialize the session state."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "api_health" not in st.session_state:
        st.session_state.api_health = check_api_health()
    
    if "available_functions" not in st.session_state:
        st.session_state.available_functions = get_available_functions()
    
    if "customer_id" not in st.session_state:
        st.session_state.customer_id = ""

    if "auto_tts" not in st.session_state:
        st.session_state.auto_tts = False

    if "tts_language" not in st.session_state:
        st.session_state.tts_language = "tr" 
    
    if "stt_language" not in st.session_state:
        st.session_state.stt_language = "tr"
    
    # Audio handling state
    if "last_audio_hash" not in st.session_state:
        st.session_state.last_audio_hash = None

def render_function_calls(function_calls: list):
    """Render function calls in an expandable section.
    
    Args:
        function_calls: List of function call results
    """
    if not function_calls:
        return
        
    with st.expander(f"🔧 Function Calls ({len(function_calls)})", expanded=False):
        for i, call in enumerate(function_calls):
            function_name = call.get("function", "Unknown")
            parameters = call.get("parameters", {})
            result = call.get("result", {})
            
            st.markdown(f"**{i+1}. {function_name}**")
            
            # Show parameters
            if parameters:
                st.markdown("**Parameters:**")
                st.json(parameters)
            
            # Show result
            if result:
                success = result.get("success", False)
                if success:
                    st.markdown("**Result:** ✅ Success")
                    if result.get("data"):
                        st.json(result["data"])
                else:
                    st.markdown("**Result:** ❌ Error")
                    st.error(result.get("error", "Unknown error"))
            
            if i < len(function_calls) - 1:
                st.divider()

def render_chat_message(message: Dict[str, Any], is_user: bool):
    """Render a chat message.
    
    Args:
        message: Message to render
        is_user: Whether the message is from the user
    """
    if is_user:
        st.chat_message("user").write(message["text"])
    else:
        with st.chat_message("assistant"):
            st.write(message["text"])
            
            # Show function calls if available
            if "function_calls" in message and message["function_calls"]:
                render_function_calls(message["function_calls"])
            
            # Show timing info
            if "elapsed_time" in message:
                st.caption(f"Response time: {message['elapsed_time']:.2f} seconds")

            # Speak button per message with better feedback
            speak_col, status_col = st.columns([1, 8])
            with speak_col:
                message_hash = abs(hash(message['text'][:100]))  # Use first 100 chars for hash
                if st.button("🔊", key=f"speak_{message_hash}", help="Speak this message"):
                    with st.spinner("🔊 Generating audio..."):
                        audio_response = text_to_speech(message["text"], language=st.session_state.get("tts_language"))
                        if audio_response:
                            render_audio_playback(audio_response)
                            with status_col:
                                st.success("✅ Audio generated", icon="🔊")
                        else:
                            with status_col:
                                st.error("❌ Audio generation failed", icon="🔊")

def render_chat_history():
    """Render the chat history."""
    for message in st.session_state.chat_history:
        render_chat_message(message, message["role"] == "user")

def render_audio_playback(audio_bytes: bytes):
    """Render an audio player for the given audio bytes."""
    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")

def render_health_info():
    """Render API health information."""
    health = st.session_state.api_health
    status = health.get("status", "unknown")
    llm_ready = health.get("llm_ready", False)
    function_caller_ready = health.get("function_caller_ready", False)
    available_functions = health.get("available_functions", 0)
    mode = health.get("mode", "unknown")
    
    status_color = "green" if status == "ok" else "red"
    llm_color = "green" if llm_ready else "red"
    function_color = "green" if function_caller_ready else "red"
    
    st.sidebar.markdown(
        f"""
        ### System Status
        API Status: :{status_color}[{status}]  
        LLM: :{llm_color}[{'Ready' if llm_ready else 'Not Ready'}]  
        Functions: :{function_color}[{'Ready' if function_caller_ready else 'Not Ready'}]  
        Available Functions: {available_functions}  
        Mode: {mode}
        """
    )

def render_customer_info():
    """Render customer information section."""
    st.sidebar.markdown("### Customer Information")
    
    customer_id = st.sidebar.text_input(
        "Customer ID",
        value=st.session_state.customer_id,
        placeholder="e.g., customer_001",
        help="Enter customer ID for personalized assistance"
    )
    
    if customer_id != st.session_state.customer_id:
        st.session_state.customer_id = customer_id
    
    # Show sample customer IDs
    st.sidebar.markdown("""
    **Sample Customer IDs:**
    - customer_001 (John Doe - Premium)
    - customer_002 (Jane Smith - Basic)  
    - customer_003 (Bob Johnson - Standard)
    """)

def render_available_functions():
    """Render available functions information."""
    functions = st.session_state.available_functions.get("functions", {})
    
    if functions:
        with st.sidebar.expander("🔧 Available Functions", expanded=False):
            for func_name, func_info in functions.items():
                st.markdown(f"**{func_name}**")
                st.caption(func_info.get("description", "No description"))
                
                # Show parameters
                parameters = func_info.get("parameters", {})
                if parameters:
                    for param_name, param_info in parameters.items():
                        required = " (required)" if param_info.get("required", False) else ""
                        st.text(f"  • {param_name}: {param_info.get('type', 'unknown')}{required}")
                
                st.markdown("---")

def render_settings():
    """Render settings controls."""
    st.sidebar.markdown("### Generation Settings")
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_TEMPERATURE,
        step=0.1,
        help="Controls randomness in response generation"
    )
    
    top_p = st.sidebar.slider(
        "Top-p",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_TOP_P,
        step=0.1,
        help="Controls diversity of response generation"
    )

    st.sidebar.markdown("### Audio Settings")
    auto_tts = st.sidebar.toggle("Auto speak assistant replies", value=st.session_state.auto_tts, help="Automatically convert assistant responses to speech")
    
    # TTS Language Selection
    current_lang_keys = list(TTS_LANGUAGES.keys())
    current_tts_lang = st.session_state.get("tts_language", "tr")
    current_tts_index = 0
    for i, lang_key in enumerate(current_lang_keys):
        if TTS_LANGUAGES[lang_key] == current_tts_lang:
            current_tts_index = i
            break
    
    tts_lang_label = st.sidebar.selectbox("TTS Language (Sesli okuma dili)", current_lang_keys, index=current_tts_index, help="Language for text-to-speech conversion")
    st.session_state.auto_tts = auto_tts
    st.session_state.tts_language = TTS_LANGUAGES[tts_lang_label]
    
    # STT Language Selection  
    current_stt_lang = st.session_state.get("stt_language", "tr")
    current_stt_index = 0
    for i, lang_key in enumerate(current_lang_keys):
        if TTS_LANGUAGES[lang_key] == current_stt_lang:
            current_stt_index = i
            break
    
    stt_lang_label = st.sidebar.selectbox("STT Language (Ses tanıma dili)", current_lang_keys, index=current_stt_index, help="Language for speech-to-text recognition")
    st.session_state.stt_language = TTS_LANGUAGES[stt_lang_label]
    
    # Audio diagnostics
    if st.sidebar.button("🔧 Test TTS"):
        # Dile göre test metni
        test_texts = {
            "Turkish": "Merhaba! Bu Türkçe sesli okuma testidir.",
            "English": "Hello! This is an English text-to-speech test.",
            "Spanish": "¡Hola! Esta es una prueba de texto a voz en español.",
            "German": "Hallo! Dies ist ein deutscher Text-zu-Sprache-Test."
        }
        test_text = test_texts.get(tts_lang_label, f"Test in {tts_lang_label}")
        
        with st.sidebar:
            with st.spinner("Testing TTS..."):
                test_audio = text_to_speech(test_text, language=st.session_state.tts_language)
                if test_audio:
                    st.success("✅ TTS working!")
                    st.audio(test_audio, format="audio/mp3")
                else:
                    st.error("❌ TTS test failed")
    
    # STT Test bilgilendirmesi
    st.sidebar.info("💡 STT test için: Mikrofon butonuna basıp konuşun")
    
    return {
        "temperature": temperature,
        "top_p": top_p
    }

def render_sidebar():
    """Render the sidebar."""
    st.sidebar.title("Call Center Assistant")
    st.sidebar.info(APP_DESCRIPTION)
    
    render_health_info()
    render_customer_info()
    render_available_functions()
    settings = render_settings()
    
    if st.sidebar.button("🔄 Refresh Status"):
        st.session_state.api_health = check_api_health()
        st.session_state.available_functions = get_available_functions()
        st.rerun()
    
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    return settings

def render_quick_actions():
    """Render quick action buttons."""
    st.markdown("### 🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Check My Account"):
            if st.session_state.customer_id:
                quick_message = f"Can you show me my account information for customer {st.session_state.customer_id}?"
            else:
                quick_message = "Can you show me account information? My customer ID is customer_001"
            return quick_message
    
    with col2:
        if st.button("📦 Available Packages"):
            return "What packages do you have available?"
    
    with col3:
        if st.button("💳 Check Billing"):
            if st.session_state.customer_id:
                quick_message = f"Can you check my billing status for customer {st.session_state.customer_id}?"
            else:
                quick_message = "Can you check my billing status? My customer ID is customer_001"
            return quick_message
    
    return None

def main():
    """Main entry point for the Streamlit app."""
    # Set page config
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📞",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Render the title
    st.title(f"{APP_TITLE} 📞")
    st.markdown("*AI-powered customer service assistant with intelligent function calling*")
    
    # Render the sidebar and get settings
    settings = render_sidebar()
    
    # Check if API is healthy
    if st.session_state.api_health.get("status") != "ok":
        st.error("⚠️ API is not responding. Please check the backend service.")
        st.stop()
    
    # Render quick actions
    quick_action_message = render_quick_actions()
    
    # Render chat history
    render_chat_history()
    
    # Audio recorder (mic button)
    audio = audiorecorder("🎤 Click to record", "🔴 Recording...")
    
    # Handle audio input with better state management
    audio_transcription = None
    if audio and len(audio) > 0:
        # Check if this is a new recording by comparing with previous state
        audio_bytes_io = io.BytesIO()
        audio.export(audio_bytes_io, format="wav")
        new_audio_data = audio_bytes_io.getvalue()
        
        # Only process if this is new audio data
        if ("last_audio_hash" not in st.session_state or 
            hashlib.md5(new_audio_data).hexdigest() != st.session_state.get("last_audio_hash")):
            
            st.session_state.last_audio_hash = hashlib.md5(new_audio_data).hexdigest()
            
            with st.spinner("🎤 Transcribing audio..."):
                audio_transcription = transcribe_audio(new_audio_data)
                if audio_transcription:
                    st.success(f"🎤 Transcribed: {audio_transcription[:100]}{'...' if len(audio_transcription) > 100 else ''}")
                else:
                    st.error("❌ Failed to transcribe audio. Please try again.")

    # Handle quick action or user input
    user_query = quick_action_message or audio_transcription or st.chat_input("Type your message here...")
    
    # Process user input
    if user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "text": user_query
        })
        
        # Show user message
        st.chat_message("user").write(user_query)
        
        # Display thinking message
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("_Processing your request..._")
            
            # Prepare conversation history for API
            conversation_history = []
            for msg in st.session_state.chat_history[:-1]:  # Exclude current message
                conversation_history.append({
                    "role": msg["role"],
                    "content": msg["text"]
                })
            
            # Send message to API
            start_time = time.time()
            response = send_chat_message(
                query=user_query,
                customer_id=st.session_state.customer_id if st.session_state.customer_id else None,
                conversation_history=conversation_history,
                temperature=settings["temperature"],
                top_p=settings["top_p"]
            )
            
            # Clear thinking message
            thinking_placeholder.empty()
            
            # Show the response
            st.write(response["response"])

            # Handle TTS - either auto or manual
            tts_audio_content = None
            if st.session_state.auto_tts:
                payload_lang = st.session_state.tts_language
                try:
                    logger.info(f"Auto-TTS enabled, language={payload_lang}")
                    tts_audio_content = text_to_speech(response["response"], language=payload_lang)
                    if tts_audio_content:
                        render_audio_playback(tts_audio_content)
                        logger.info("Auto-TTS audio played successfully")
                    else:
                        logger.error("Auto-TTS failed to generate audio")
                        st.warning("⚠️ Auto-TTS failed. Try using the speak button.")
                except Exception as e:
                    logger.error(f"Auto-TTS error: {e}")
                    st.warning(f"⚠️ Auto-TTS error: {str(e)}")
            
            # Always show manual speak button (only if auto-TTS is off or failed)
            if not st.session_state.auto_tts or not tts_audio_content:
                speak_col, status_col = st.columns([1, 3])
                with speak_col:
                    if st.button("🔊 Speak reply", key=f"speak_latest_{int(time.time()*1000)}"):
                        with st.spinner("🔊 Generating audio..."):
                            manual_audio = text_to_speech(response["response"], language=st.session_state.get("tts_language"))
                            if manual_audio:
                                render_audio_playback(manual_audio)
                                with status_col:
                                    st.success("✅ Audio generated successfully")
                            else:
                                with status_col:
                                    st.error("❌ Failed to generate audio")
            
            # Show function calls if available
            if response.get("function_calls"):
                render_function_calls(response["function_calls"])
            
            elapsed_time = response.get("elapsed_time", time.time() - start_time)
            st.caption(f"Response time: {elapsed_time:.2f} seconds")
        
        # Add assistant message to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "text": response["response"],
            "function_calls": response.get("function_calls", []),
            "elapsed_time": elapsed_time
        })

if __name__ == "__main__":
    main() 