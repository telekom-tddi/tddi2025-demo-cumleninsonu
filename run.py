#!/usr/bin/env python3
"""
Run script for the Call Center Chatbot System.
This script provides a convenient way to run the different components of the call center system.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

# Add the current directory to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Filter out env directories for watchfiles
ignore_dirs = [
    "test-env",
    "venv",
    "data",
    "__pycache__",
    ".git"
]

def setup_environment():
    """Set up the environment for the call center system."""
    # Create cache directory
    cache_dir = os.path.join(BASE_DIR, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    print("✅ Cache directory created")
    
    # Check if required environment variables are set
    required_vars = ["GOOGLE_GENAI_API_KEY", "GOOGLE_API_KEY"]  # Either one is fine
    missing_vars = []
    
    # Check if at least one Google API key is set
    if not os.getenv("GOOGLE_GENAI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        missing_vars.append("GOOGLE_GENAI_API_KEY or GOOGLE_API_KEY")
    
    if missing_vars:
        print(f"⚠️  Warning: Missing environment variables: {', '.join(missing_vars)}")
        print("   Please set these in your .env file or environment")
        print("   GOOGLE_GENAI_API_KEY or GOOGLE_API_KEY is required for Google GenAI API access")
    else:
        print("✅ Environment variables configured")

def test_functions():
    """Test the call center functions."""
    print("🧪 Testing call center functions...")
    
    try:
        # Import and test basic function calling
        from src.functions.function_caller import FunctionCaller
        from src.functions.call_center_functions import get_available_packages, get_customer_info
        
        # Test function caller
        function_caller = FunctionCaller()
        print(f"   ✅ Function caller initialized with {len(function_caller.registry)} functions")
        
        # Test a simple function
        result = get_available_packages()
        if result["success"]:
            print(f"   ✅ get_available_packages: {len(result['data'])} packages available")
        
        # Test customer lookup
        result = get_customer_info("customer_001")
        if result["success"]:
            print(f"   ✅ get_customer_info: Found customer {result['data']['name']}")
        
        print("✅ Function tests completed successfully")
        
    except Exception as e:
        print(f"❌ Function test failed: {e}")
        return False
    
    return True

def test_translator():
    """Test the translator."""
    print("🗣️ Testing translator...")
    try:
        from src.translator.translator import Translator
        translator = Translator()
        
        english_text = "Hello, how are you?"
        turkish_text = translator.translate_en_to_tr(english_text)
        if turkish_text:
            print(f"   ✅ EN->TR Translation successful: '{english_text}' -> '{turkish_text}'")
        else:
            print("   ❌ EN->TR Translation failed.")
            return False

        turkish_text_2 = "Bugün hava çok güzel."
        english_text_2 = translator.translate_tr_to_en(turkish_text_2)
        if english_text_2:
             print(f"   ✅ TR->EN Translation successful: '{turkish_text_2}' -> '{english_text_2}'")
        else:
            print("   ❌ TR->EN Translation failed.")
            return False
            
        print("✅ Translator tests completed successfully")
        return True
    except Exception as e:
        print(f"❌ Translator test failed: {e}")
        return False

def run_call_center_api():
    """Run the call center FastAPI backend."""
    print("🚀 Starting Call Center API...")
    
    # Use uvicorn directly to run the call center API
    cmd = ["uvicorn", "src.api.call_center_api:app", "--host", "0.0.0.0", "--port", "8000"]
    print(f"Running: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=dict(os.environ, PYTHONPATH=BASE_DIR))

def run_call_center_webapp():
    """Run the call center Streamlit web application."""
    print("🌐 Starting Call Center Web Interface...")
    cmd = ["streamlit", "run", "webapp/call_center_app.py", "--server.port", "8501"]
    print(f"Running: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=dict(os.environ, PYTHONPATH=BASE_DIR))

def check_health():
    """Check the health of running services."""
    import requests
    import time
    
    print("🔍 Checking service health...")
    
    # Wait a moment for services to start
    time.sleep(2)
    
    # Check API health
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            mode = health_data.get("mode", "unknown")
            functions = health_data.get("available_functions", 0)
            print(f"   ✅ API is healthy (mode: {mode}, functions: {functions})")
        else:
            print(f"   ⚠️  API returned status {response.status_code}")
    except Exception as e:
        print(f"   ❌ API health check failed: {e}")
    
    # Check if Streamlit is running (basic check)
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("   ✅ Web interface is accessible")
        else:
            print(f"   ⚠️  Web interface returned status {response.status_code}")
    except Exception as e:
        print(f"   ❌ Web interface check failed: {e}")

def clean_cache():
    """Clean corrupted cache files and lock files."""
    print("🧹 Cleaning cache...")
    
    cache_dir = os.path.join(BASE_DIR, ".cache")
    locks_dir = os.path.join(cache_dir, ".locks")
    
    try:
        # Check if cache exists
        if not os.path.exists(cache_dir):
            print("   ✅ No cache directory found")
            return
        
        # Remove lock files
        if os.path.exists(locks_dir):
            import shutil
            print("   🗑️  Removing lock files...")
            shutil.rmtree(locks_dir, ignore_errors=True)
            print("   ✅ Lock files removed")
        
        # Count cached models
        model_dirs = [d for d in os.listdir(cache_dir) if d.startswith("models--")]
        if model_dirs:
            print(f"   📁 Found {len(model_dirs)} cached model(s)")
            for model_dir in model_dirs:
                model_path = os.path.join(cache_dir, model_dir)
                size = sum(os.path.getsize(os.path.join(dirpath, filename))
                          for dirpath, dirnames, filenames in os.walk(model_path)
                          for filename in filenames) / (1024**3)  # GB
                print(f"      - {model_dir}: {size:.2f} GB")
        else:
            print("   📭 No cached models found")
        
        print("   ✅ Cache cleanup completed")
        
    except Exception as e:
        print(f"   ❌ Cache cleanup failed: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run the Call Center Chatbot System")
    
    # Call center commands
    parser.add_argument("--setup", action="store_true", help="Set up the call center environment")
    parser.add_argument("--test", action="store_true", help="Test call center functions")
    parser.add_argument("--api", action="store_true", help="Run the call center API")
    parser.add_argument("--webapp", action="store_true", help="Run the call center web interface")
    parser.add_argument("--all", action="store_true", help="Run complete call center system")
    parser.add_argument("--health", action="store_true", help="Check service health")
    parser.add_argument("--clean-cache", action="store_true", help="Clean corrupted cache and lock files")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help and run the default call center system
    if not any(vars(args).values()):
        print("🤖 Call Center Chatbot System")
        print("=" * 50)
        parser.print_help()
        print("\n" + "=" * 50)
        print("No arguments provided. Starting complete call center system...")
        args.all = True
    # Test functions
    if args.test or args.all:
        if not test_functions():
            print("❌ Function tests failed. Please check your setup.")
            return
        if not test_translator():
            print("❌ Translator tests failed. Please check your setup.")
            return
    
    # Health check only
    if args.health:
        check_health()
        return
    
    # Clean cache only
    if args.clean_cache:
        clean_cache()
        return
    
    # Run services
    processes = []
    try:
        # Call center system
        if args.api or args.all:
            api_process = run_call_center_api()
            processes.append(api_process)
            print("⏳ Waiting for API to start...")
            time.sleep(5)  # Give API more time to load models
        
        if args.webapp or args.all:
            webapp_process = run_call_center_webapp()
            processes.append(webapp_process)
            time.sleep(2)
        
        
        # Show status and URLs
        if processes:
            print("\n" + "=" * 60)
            print("🎉 Call Center Chatbot System is running!")
            print("=" * 60)
            
            if args.api or args.all:
                print("📋 Call Center API: http://localhost:8000")
                print("📖 API Documentation: http://localhost:8000/docs")
            
            if args.webapp or args.all:
                print("💬 Call Center Chat: http://localhost:8501")
            
            print("=" * 60)
            
            # Run health check
            if args.all or args.webapp:
                check_health()
            
            print("\n✅ Press Ctrl+C to stop all services")
            print("\n💡 Try these sample queries:")
            print("   • 'Check my account for customer_001'")
            print("   • 'What packages are available?'")
            print("   • 'I want to upgrade to premium'")
            print("   • 'Check my billing status'")
            
            # Keep running until interrupted
            for p in processes:
                p.wait()
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        for p in processes:
            try:
                p.terminate()
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        print("✅ All services stopped")

if __name__ == "__main__":
    main() 